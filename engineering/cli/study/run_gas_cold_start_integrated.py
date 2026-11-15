"""从原三任务入口实际发放、接续并重载逐行推理，验收保留首轮的交付候选。"""
import json
import subprocess
import sys

from score_gas_probability_fusion import ROOT,BASE,read,atomic_json
from score_integrated_observer import main as score
from compare_group_objectives import official
from engineering.cli.workflow.gas_observer_predict import infer
from harness.data.official_gas import prediction_request,prediction_label

DEST=BASE/"whole-maintenance/gas-cold-start/integrated"


def accepted(task):
    run=read(DEST/task/"run.json")
    if "accepted" in run:return [r["request"] for r in run["accepted"]]
    return [r["submitted"] for r in run["rounds"]]


def execute(through):
    command=[sys.executable,"-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),"public-run","all",
        "--gas-feedback","32","--gas-field-mode","pooled-midway-cold","--engine-input-repair",
        "--rounds",str(through),"--output",str(DEST)]
    r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
    if r.returncode:raise RuntimeError(r.stderr[-1200:])
    print(json.dumps({"stage/rounds":through}),flush=True)


def main():
    if (DEST/"scores.json").exists():raise ValueError("已完成的完整交付请直接复用")
    execute(4);prefix={t:accepted(t) for t in ("gas","bearing","engine")}
    execute(10);final={t:accepted(t) for t in prefix}
    if any(final[t][:len(prefix[t])]!=prefix[t] for t in prefix):raise ValueError("接续更换了已交接预测")
    execute(10)
    if any(accepted(t)!=final[t] for t in final):raise ValueError("完成后恢复改变了已发预测")
    score(DEST)
    pkg=official("gas")[0];values={r.sample_id:r.values for r in pkg.validation}
    folder=DEST/"gas";report=read(folder/"run.json");checked=0
    for j,event in enumerate(report["accepted"]):
        payload=read(folder/"accepted"/(event["request"]+".json"))["payload"];keys=sorted(payload["predictions"])
        snapshot=folder/"snapshots"/(payload["snapshot"]+".json")
        outputs=[prediction_label(v) for v in infer(snapshot,ROOT/"data/official",
            (prediction_request(list(values[k])) for k in keys))]
        if outputs!=[payload["predictions"][k] for k in keys]:raise ValueError("单条推理不等于实际交付")
        expected="observer" if j==0 else ("pooled-source" if j<5 else "pooled-separate")
        if payload["selected_tool"]!=expected:raise ValueError("实际采用配方不符")
        checked+=len(keys)
    checks={"resume_preserves_issued":True,"gas_single_row_matches":True,"gas_rows":checked,
            "source": "actual public-run all and original JSON single-row endpoint", "independent_validation_pending":True}
    atomic_json(DEST/"delivery-checks.json",checks)
    results=read(DEST/"scores.json")
    print(json.dumps({"delivery":{k:v["score"] for k,v in results["tasks"].items()},
        "verification":{"resume_preserves_issued":True,"single_row_matches":True},"independent_validation_pending":True}),flush=True)


if __name__=="__main__":main()
