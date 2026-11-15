"""实际运行三任务总入口及同目录恢复，再比较原推荐版的每一条交付。

仅验证由训练侧已选定的权重，不依据本轮服务答案更换候选。
"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/"engineering/src")]
from harness.contracts import digest
from harness.runtime.storage import atomic_json
from score_integrated_observer import main as score

BASE = ROOT/"results/experiments/local-defaults/whole-maintenance/bearing-hybrid-readout"
OLD = ROOT/"results/experiments/local-defaults/fixed-tree-references/setting-reference/automatic/integrated"


def read(path):
    return json.loads(path.read_bytes())


def predictions(folder, task):
    report=read(folder/task/"run.json")
    if "accepted" in report:
        paths=[folder/task/"accepted"/(r["request"]+".json") for r in report["accepted"]]
    else:
        paths=[folder/task/r["submitted"] for r in report["rounds"]]
    merged={}
    for path in paths:
        payload=read(path)["payload"]["predictions"]
        if merged.keys()&payload.keys():
            raise ValueError("总入口重复交付同一测点")
        merged.update(payload)
    return merged


def main():
    training=read(BASE/"training.json")
    weights=BASE/"models"/(training["source_selected"]+".npz")
    if sha256(weights.read_bytes()).hexdigest()!=training["weights"][training["source_selected"]]:
        raise ValueError("源侧选定的模型已变")
    folder=BASE/"integrated"
    command=[sys.executable,"-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),"public-run","all",
             "--gas-feedback","32","--engine-input-repair","--bearing-checkpoint",str(weights),"--output",str(folder)]
    subprocess.run(command,cwd=ROOT,check=True)
    score(folder)
    before={task:predictions(folder,task) for task in ("gas","bearing","engine")}
    # 第二遍恢复同一交接目录；不能产生第二组答案或修改已接收的预测。
    subprocess.run(command,cwd=ROOT,check=True)
    after={task:predictions(folder,task) for task in before}
    changes={}
    for task,p in after.items():
        old=predictions(OLD,task)
        if p.keys()!=old.keys() or p!=before[task]:
            raise ValueError("样本身份或恢复后的预测不同")
        changes[task]=sum(abs(v-old[k])>1e-7 for k,v in p.items())
        if task!="bearing" and changes[task]:
            raise ValueError("更换轴承权重影响了另一任务")
    issued=read(BASE/"issued.json")["cases"]["public"]
    expected=dict(zip(issued["ids"],issued["predictions"][training["source_selected"]],strict=True))
    if after["bearing"].keys()!=expected.keys() or max(abs(v-expected[k]) for k,v in after["bearing"].items())>1e-7:
        raise ValueError("整套入口未复现冻结的轴承候选")
    report={"source_selected":training["source_selected"],"weights_sha256":training["weights"][training["source_selected"]],
            "same_directory_resume_preserves_predictions":True,"different_from_current":changes,
            "prediction_digests":{k:digest(v) for k,v in after.items()},"tasks":read(folder/"scores.json")["tasks"]}
    atomic_json(folder/"comparison.json",report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
