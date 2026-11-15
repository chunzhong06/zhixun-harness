"""从官方原包答案和交付回执独立计分，并核实各组的输入、预算及真实工作进程费用。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from math import fsum, isfinite, sqrt
from pathlib import Path
from zipfile import ZipFile
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow.public_maintenance import PACKAGES, PROTOCOL, SERIES, gas_service_windows, partition_rul, windows
from harness.contracts import digest
from harness.data.official_gas import read_official_gas
from harness.evaluation.gas_pilot import partition as gas_partition
from harness.evaluation.metrics import macro_f1
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import atomic_json

BASE = ROOT/"results/experiments/local-defaults"
ARMS = ("original","rule","fixed","selected","paper")


def read(path):
    return json.loads(path.read_bytes())


def official(task):
    if task=="gas":
        file,checksum,reader = "cpu-a-guide-20261110-v1.zip",PACKAGE_SHA,read_official_gas
    else:
        file,checksum,reader = PACKAGES[task]
    archive = ROOT/"data/official"/task/file
    if sha256(archive.read_bytes()).hexdigest()!=checksum:
        raise ValueError("官方包身份变化")
    package = reader(archive)
    if task=="gas":
        batches, held = gas_partition(package,read(ROOT/"engineering/configs/experiments/official-gas-pilot.json"))
        batches = gas_service_windows(batches,held)
    else:
        groups,assignment = partition_rul(package,read(PROTOCOL))
        batches = windows(package,assignment,groups["fit"],read(SERIES),task)
    return package,checksum,batches


def metric(task,answers,predictions):
    if task=="gas":
        return macro_f1((answers[i],p) for i,p in predictions.items())
    return sqrt(fsum((answers[i]-p)**2 for i,p in predictions.items())/len(predictions))


def cost(folder, checksum, fit):
    times = defaultdict(list)
    for file in (folder/"jobs").glob("*/state.json"):
        job,packet = read(file),read(file.with_name("input.json"))
        request = packet["request"]
        if (job["status"]!="completed" or job["nonce"]!=packet["nonce"] or job["request_id"]!=digest(request)
                or request["package_sha256"]!=checksum or not isfinite(job["cpu_seconds"])):
            raise ValueError("工作进程状态、原包或费用回执不符")
        if "fit_ids" in request and not set(request["fit_ids"])<=fit:
            raise ValueError("候选混入训练池以外的数据")
        times[request["kind"]].append(job["cpu_seconds"])
    return {"physical_builds":len(times["continue"]),"worker_cpu_seconds":fsum(fsum(v) for v in times.values()),
            "by_operation":{k:{"calls":len(v),"cpu_seconds":fsum(v)} for k,v in times.items()}}


def score(task, *, first_engine_filter=False):
    package, checksum, batches = official(task)
    truth = {s.sample_id:s.label for s in package.validation}
    expected = {b.batch_id:{s.sample_id for s in b.samples if s.role=="serve" and s.available_at==b.ordinal} for b in batches}
    all_fit = {s.sample_id for b in batches for s in b.samples if s.role=="fit"}
    if len(expected)!=10 or any(s.label is not None for b in batches for s in b.samples if s.role=="serve"):
        raise ValueError("十轮服务分片或标签隔离错误")
    results, streams = {}, {}
    controls = None
    initial_sha = package.checkpoint_sha256.removeprefix("sha256:") if task=="gas" else package.checkpoint_sha
    arms = ("fixed", "paper", "first-engine-filter") if first_engine_filter else ARMS
    for arm in arms:
        folder = BASE/(f"paper-{task}" if arm=="paper" else arm)/task
        if arm=="paper" and task=="gas":
            folder = BASE/"paper-gas-round-budget"/task
        report, state = read(folder/"run.json"),read(folder/"state.json")
        if report["completed_rounds"]!=10 or not read(folder/report["swanlab"])["verified"]:
            raise ValueError("交付与跟踪回执不齐")
        cfg = dict(state["config"])
        cfg.pop("maintenance_trigger")
        if cfg["max_trials"]!=1 or cfg["max_plan_calls"]!=1:
            raise ValueError("每轮预算不一致")
        signatures = digest([cfg,state["samples"]])
        if controls is None:
            controls = signatures
        if controls!=signatures:
            raise ValueError("输入、角色、标签时序或运行上限不同")
        store = ModelArtifacts(folder/"models")
        root, = (v for v in state["models"].values() if v["parent_id"] is None)
        if store.read(root["artifact"])[0]["sha256"]!=initial_sha:
            raise ValueError("起始权重不符")
        predicted, scores, adopted = {}, [], []
        for r,b in zip(report["rounds"],batches,strict=True):
            payload = read(folder/r["submitted"])["payload"]
            p = payload["predictions"]
            if (r["batch"]!=b.batch_id or r["selected"]!=payload["model_id"]
                    or p.keys()!=expected[b.batch_id] or p.keys()&predicted.keys() or len(r["trials"])>1):
                raise ValueError("轮次、在用版或预测身份不一致")
            op = state["operations"][f"{b.batch_id}:serve:{r['selected']}"]
            job = folder/"jobs"/op["execution"]["request_id"]
            values = read(job/"state.json")["value"]
            values = values["predictions"] if "gas_repairs" in values else values
            if p!=values:
                raise ValueError("交接预测与实际工作进程不同")
            store.read(r["model_artifact"])
            predicted.update(p)
            scores.append(metric(task,truth,p))
            if r["current"]!=r["selected"]:
                adopted.append(r["batch"])
        totals = cost(folder,checksum,all_fit)
        calls = []
        if arm=="paper":
            calls = read(folder/"driver-exchanges.json")["exchanges"]
            if [v["batch_id"] for v in calls]!=list(expected) or any(v["status"]!="completed" for v in calls):
                raise ValueError("模型提案未全部完成")
        streams[arm] = predicted
        results[arm] = {"score":metric(task,truth,predicted),"round_scores":scores,
                        "worst_round":(min if task=="gas" else max)(scores),"rows":len(predicted),
                        "adopted":adopted,"model_proposals":len(calls),"proposal_elapsed_seconds":fsum(v["elapsed_seconds"] for v in calls),
                        "issued_sha256":digest(predicted),**totals}
    for name,p in streams.items():
        results[name]["different_from_fixed"] = sum(v!=streams["fixed"][i] for i,v in p.items())
    target = BASE/"scores"/(task+".json")
    if first_engine_filter:
        # 比较真实交接给同一台设备的预测，保留总体收益的代价及首轮行动是否复现。
        devices = defaultdict(list)
        for batch in batches:
            for row in batch.samples:
                if row.role == "serve" and row.available_at == batch.ordinal:
                    devices[row.unit_id].append(row.sample_id)
        for name, predictions in streams.items():
            results[name]["by_device"] = {
                unit: metric(task, truth, {i: predictions[i] for i in ids})
                for unit, ids in sorted(devices.items())
            }
        target = BASE/"scores"/"engine-first-filter.json"
    atomic_json(target,{"task":task,"input_and_budget":controls,"package_sha256":checksum,"arms":results})
    print(task, {k:round(v["score"],6) for k,v in results.items()})


def verify_package():
    release = ROOT/"results/engineering/contest-reproduction"
    zip_path = release/"智训引擎-公开运行.zip"
    receipt = read(release/"checks/public-maintenance-entry.json")
    if sha256(zip_path.read_bytes()).hexdigest()!=receipt["bundle_sha256"]:
        raise ValueError("源码包与解压验收回执不符")
    with ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("engineering/") and z.read(name)!=(ROOT/name).read_bytes():
                raise ValueError(f"当前代码与交付包不同 {name}")
    results = {}
    for task in ("gas","bearing","engine"):
        folder = BASE/"selected"/task
        rounds = read(folder/"run.json")["rounds"]
        metrics = read(BASE/"scores"/(task+".json"))["arms"]["selected"]
        if len(rounds)!=10 or len(receipt["tasks"][task])!=10:
            raise ValueError("默认入口未完整运行")
        checked = {}
        for saved,dist in zip(rounds,receipt["tasks"][task],strict=True):
            p = read(folder/saved["submitted"])["payload"]["predictions"]
            raw = json.dumps(p,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
            if (dist["batch"]!=saved["batch"] or dist["trial_count"]!=len(saved["trials"])
                    or dist["predictions_sha256"]!=sha256(raw).hexdigest()):
                raise ValueError("解压包默认输出与已评分的实际交接不一致")
            checked.update(p)
        if digest(checked)!=metrics["issued_sha256"]:
            raise ValueError("交付预测与计分绑定不符")
        results[task] = {"rows":len(checked),"rounds":10,"score":metrics["score"],"issued_sha256":digest(checked)}
    atomic_json(BASE/"delivery.json",{"zip_sha256":receipt["bundle_sha256"],"tasks":results,
                "environment":"Windows 解包默认命令，与分段恢复后十轮交接逐值相等","container_built":receipt["container_built"]})
    print("交付包与三组评分预测逐条一致")


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("gas","bearing","engine","all","package"))
    args = parser.parse_args()
    if args.task=="package":
        verify_package()
    else:
        for name in (("gas","bearing","engine") if args.task=="all" else (args.task,)):
            score(name)
