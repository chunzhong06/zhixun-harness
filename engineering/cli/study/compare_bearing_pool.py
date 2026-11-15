"""比较一次官方续训的样本来源。候选、父版及实际发布版的预测全部封存后才给服务设备计分。"""
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow.public_maintenance import assemble_public, run_prepared
from harness.contracts import BatchContext, Sample, digest
from harness.data.official_bearing import read_official_bearing
from harness.models.official_rul_cpu import RulPilotTools
from harness.runtime.engine import Harness
from harness.runtime.storage import LocalStore, atomic_json
from read_kaist_windows import prepare
from unseen_sources import bearing_data
from verify_unseen_scores import remaining_minutes
from run_rul_maintenance import PROTOCOL, partition

OUT = ROOT/"results/experiments/bearing-pool"
ARCHIVE = ROOT/"data/official/bearing/cpu-b-rul-20261110-v1.zip"
SEEDS = (17,29,43,71,101)


def load(p):
    return json.loads(p.read_bytes())


def rosters(package, explore):
    official = sorted(package.train,key=lambda s:s.sample_id)
    old = partition(package,load(PROTOCOL))[0]["fit"]
    if len(official)!=7032 or len(old)!=72:
        raise ValueError("官方训练池身份不符")
    records = {"old":tuple(old)}
    for seed in SEEDS:
        indices = np.random.default_rng(seed).choice(len(official),72,replace=False)
        records[f"uniform-{seed}"] = tuple(official[i] for i in indices)
    x = np.asarray([r.values for r in official])
    scale = x.std(axis=0)
    scale[scale==0] = 1
    by_unit = defaultdict(list)
    for r in explore:
        if r.role!="explore":
            raise ValueError("匹配参照只接受已结束的探索设备")
        by_unit[r.unit_id].append(r)
    if len(by_unit)!=2:
        raise ValueError("匹配必须使用预登记的两台设备")
    anchors = []
    for uid in sorted(by_unit):
        rows = sorted(by_unit[uid],key=lambda r:r.time_end)
        anchors.append([rows[i] for i in np.linspace(0,len(rows)-1,36,dtype=int)])
    used = np.zeros(len(official),dtype=bool)
    matches = []
    for pair in zip(*anchors,strict=True):
        for anchor in pair:
            dist = np.sum(np.square((x-np.asarray(anchor.values))/scale),axis=1)
            dist[used] = np.inf
            idx = int(dist.argmin())
            used[idx] = True
            matches.append(official[idx])
    records["matched"] = tuple(matches)
    for rows in records.values():
        if len({r.sample_id for r in rows})!=72 or any(r.role!="fit" for r in rows):
            raise ValueError("清单须有 72 条唯一官方训练记录")
    return records


def material():
    archive, _, _ = bearing_data()
    prior_test, _, _ = bearing_data(full_test=True)
    visible = [replace(r,available_at=0) for rows in archive.values() for r in rows if r.role in ("explore","adopt")]
    service = [replace(r,label=None,available_at=0) for rows in prior_test.values() for r in rows]
    windows = prepare()["rows"]
    times = [datetime.fromisoformat(r["time"]) for r in windows]
    if len(times)!=129 or times!=sorted(set(times)):
        raise ValueError("KAIST 的测量日期不完整")
    for record,t in zip(windows,times,strict=True):
        if "invalid_nonfinite_values" in record:
            continue
        minutes = (t-times[0]).total_seconds()/60
        service.append(Sample("KAIST/"+record["file"],1,tuple(record["features"]),None,"serve","KAIST",0,
                              "KAIST/single",minutes,minutes))
    if {s.unit_id for s in visible}&{s.unit_id for s in service}:
        raise ValueError("确认与评分设备重叠")
    return visible,service


def run(arm, rows, visible, service):
    folder = OUT/arm
    driver = {"mode":"rule"}
    base, official = assemble_public("bearing",ROOT/"data/official",folder,"fixed",driver)
    tools = RulPilotTools("bearing",ARCHIVE,base.tools.package,rows,(),None,base.tools.runner,
                          batches=official,reuse_identical_fit=True)
    tools.values = {r.sample_id:r.values for r in [*visible,*service]}
    engine = Harness(base.task,base.config,tools,base.planner,base.store,allow_real_execution=True)
    batch = BatchContext(base.task.task_id,official[0].batch_id,0,(*rows,*visible,*service))
    with LocalStore(folder/"study-lock").exclusive():
        report = run_prepared("bearing",folder,"fixed",1,driver,engine,[batch])["rounds"][0]
        trial, = report["trials"]
        if trial["status"]!="completed":
            raise ValueError("未取得唯一真实候选")
        models = load(folder/"state.json")["models"]
        def reference(mid):
            return models[mid]["artifact"]
        parent = reference(report["current"])
        candidate = reference(trial["model"])
        fit_request = tools.runner.artifacts.read(candidate)[0]["request"]
        if (fit_request["fit_ids"]!=sorted(r.sample_id for r in rows) or fit_request["epochs"]!=1
                or fit_request["parent"]!=parent):
            raise ValueError("实际训练行、遍数或父权重不符")
        inputs = [{"sample_id":r.sample_id,"values":list(r.values)} for r in service]
        incumbent = tools.predict(parent,inputs)
        proposal = tools.predict(candidate,inputs)
        issued = load(folder/report["submitted"])["payload"]["predictions"]
        adopted = report["current"]!=report["selected"]
        if issued!=(proposal if adopted else incumbent):
            raise ValueError("发布预测与所选真实权重不符")
        atomic_json(folder/"paired.json",{"original":incumbent,"candidate":proposal,"served":issued,"adopted":adopted,
                    "fit_ids":fit_request["fit_ids"],"input_sha":digest(inputs),"outcome":report["outcome"]})


def metrics(predictions,ids,truth):
    errors = np.asarray([predictions[i]-truth[i] for i in ids])
    return {"rmse":float(np.sqrt(np.mean(errors**2))),"bias":float(np.mean(errors)),"n":len(ids)}


def score(names):
    pairs = {arm:load(OUT/arm/"paired.json") for arm in names}
    original = pairs["old"]["original"]
    for v in pairs.values():
        if v["original"]!=original or v["input_sha"]!=pairs["old"]["input_sha"]:
            raise ValueError("对照的起点或测量内容不同")
    truth = remaining_minutes("full-test-manifest.json")
    # 仅用原文件名的时钟复算，候选训练与版本决定没有这个映射。
    events = {sid:datetime.strptime(sid,"KAIST/LogFile_%Y-%m-%d-%H-%M-%S.csv")
              for sid in original if sid.startswith("KAIST/")}
    clock_rows = prepare()["rows"]
    end = max(datetime.strptime(r["file"],"LogFile_%Y-%m-%d-%H-%M-%S.csv") for r in clock_rows)
    truth.update({sid:(end-t).total_seconds()/60 for sid,t in events.items()})
    cohorts = {"KAIST":sorted(events),"PRONOSTIA":sorted(i for i in original if i.startswith("PHM12/"))}
    expected = {"KAIST/"+r["file"] for r in clock_rows if "features" in r}
    if set(cohorts["KAIST"])!=expected or len(cohorts["PRONOSTIA"])!=586 or not expected:
        raise ValueError("外部预测数量不匹配")
    result = {}
    for name,ids in cohorts.items():
        units = sorted({s.split("/")[1] if name=="PRONOSTIA" else "single" for s in ids})
        groups = {u:[s for s in ids if name=="KAIST" or s.split("/")[1]==u] for u in units}
        table = {"original":metrics(original,ids,truth),"arms":{}}
        for arm,v in pairs.items():
            table["arms"][arm] = {"candidate":metrics(v["candidate"],ids,truth),"served":metrics(v["served"],ids,truth),
                "adopted":v["adopted"],"outcome":v["outcome"],
                "by_unit":{u:{"candidate":metrics(v["candidate"],g,truth),"served":metrics(v["served"],g,truth)}
                           for u,g in groups.items()}}
        result[name] = table
    atomic_json(OUT/"scores.json",result)
    for c,v in result.items():
        print(c, {a:(round(r["candidate"]["rmse"],2),round(r["served"]["rmse"],2),r["adopted"]) for a,r in v["arms"].items()})


if __name__=="__main__":
    visible, service = material()
    arms = rosters(read_official_bearing(ARCHIVE),[r for r in visible if r.role=="explore"])
    for arm, rows in arms.items():
        run(arm,rows,visible,service)
    score(arms)
