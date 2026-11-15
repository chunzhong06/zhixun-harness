"""正常三任务入口，从四轮断点恢复，使用已交接预测而非训练分数验收。"""
import json
import subprocess
import sys

import numpy as np

from score_gas_probability_fusion import BASE, ROOT, digest, read, atomic_json, metrics, read_official_gas, ARCHIVE
from score_gas_served_scale import DEST as STUDY
from score_gas_feedback_selection import DEST as SELECTED
from compare_group_objectives import official
from score_local_defaults import metric
from engineering.cli.workflow.gas_observer_predict import infer
from harness.data.official_gas import prediction_label, prediction_request

DEST = STUDY / "integrated"
PARENT = BASE / "whole-maintenance/engine-continued-integrated"


def execute(task, rounds, with_options=True, *, destination=DEST, mode="boundary-served-scale"):
    cmd = [sys.executable,"-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),"public-run",task,
           "--rounds",str(rounds),"--output",str(destination)]
    if with_options:
        cmd += ["--gas-feedback","32","--gas-field-mode",mode,"--engine-continued-condition"]
    run = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
    if run.returncode:
        raise RuntimeError(run.stderr[-1900:])


def load_issued(folder, task):
    r = read(folder/task/"run.json")
    files = (["accepted/"+event["request"]+".json" for event in r["accepted"]] if "accepted" in r
             else [event["submitted"] for event in r["rounds"]])
    records = [read(folder/task/f) for f in files]
    issued = {}
    for f,rec in zip(files,records,strict=True):
        if rec["request_id"] != f.split("/")[-1][:-5]:
            raise ValueError("存储的请求号不符")
        p = rec["payload"]
        if "identity" in r and digest([r["identity"],p]) != rec["request_id"]:
            raise ValueError("回读的内容不符")
        if issued.keys() & p["predictions"].keys():
            raise ValueError("发出多次服务预测")
        issued.update(p["predictions"])
    return files, records, issued


def main():
    if (DEST/"scores.json").exists():
        raise ValueError("已有本次整套结果")
    tasks = ("gas","bearing","engine")
    execute("all",4)
    prefix = {t:load_issued(DEST,t)[0] for t in tasks}
    execute("all",10)
    gas_files,records,issued = load_issued(DEST,"gas")
    execute("gas",10,with_options=False)
    if load_issued(DEST,"gas")[0] != gas_files:
        raise ValueError("省略参数恢复后改变交付")
    result = {}
    for t in tasks:
        files,_,data = load_issued(DEST,t)
        if files[:len(prefix[t])] != prefix[t] or len(files)!=10:
            raise ValueError("整套恢复更改历史预测或缺少轮次")
        if t != "gas":
            _,_,before = load_issued(PARENT,t)
            if data != before:
                raise ValueError("其他任务出现逐条退步或变化")
            answers = {r.sample_id:r.label for r in official(t)[0].validation}
            result[t] = {"score":metric(t,answers,data),"rows":len(data),"identical":True}
    pkg = read_official_gas(ARCHIVE)
    source = {r.sample_id:r for r in pkg.validation}
    frozen = read(STUDY/"seed-17/边界/run.json")
    query = read(SELECTED/"seed-17/边界/run.json")
    actual = read(DEST/"gas/run.json")
    different, different_queries = [], []
    for j,(old,r) in enumerate(zip(frozen["requests"],records,strict=True)):
        p = r["payload"]
        keys = sorted(p["predictions"])
        weights = DEST/"gas/snapshots"/(p["snapshot"]+".json")
        predicted = [prediction_label(a) for a in infer(weights,ROOT/"data/official",
                     (prediction_request(list(source[k].values)) for k in keys))]
        if predicted != [p["predictions"][k] for k in keys]:
            raise ValueError("单条推理未复现批交付")
        q = read(STUDY/"seed-17/边界/accepted"/(old+".json"))["payload"]
        ref = dict(zip(q["ids"],q["predictions"]["全部服务中途"],strict=True))
        different.extend(k for k in keys if p["predictions"][k]!=ref[k])
        expected = query["feedback"][j]["ids"] if j < 9 else []
        if set(actual["accepted"][j]["returned"]) != set(expected):
            different_queries.append(j+1)
    ids = list(issued)
    table = metrics([source[k].label for k in ids],[issued[k] for k in ids])
    result["gas"] = {"score":table["macro"],"rows":len(ids),"worst":table["worst"],"per_class":table["recall"],
                     "screen_prediction_differences":different,"screen_query_different_rounds":different_queries,
                     "single_row_verified":len(ids),"feedback":sum(len(e["returned"]) for e in actual["accepted"])}
    atomic_json(DEST/"scores.json",result)
    print(json.dumps({"整套实际交付":result},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
