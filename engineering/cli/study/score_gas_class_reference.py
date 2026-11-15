"""固定之前实际取得的回标，比较校准、重训与根据历史交付证据发布的增量。"""
from pathlib import Path
import json
from hashlib import sha256
import sys

import numpy as np
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from engineering.cli.workflow.field_classification import checked
from harness.contracts import digest
from harness.methods.gas_class_reference import ClassReference
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from score_gas_query_risk import BASE, DEST as PRIOR, STREAMS, read

DEST = PRIOR.parent / "gas-class-reference"
MODES = ("原刻度", "独立刻度", "固定四分一", "固定各半", "固定四分三", "固定校准", "只用现场", "现场证据")


def run(site, relative, query="均匀覆盖"):
    folder = BASE / relative
    spec = read(folder / "stream.json")
    src = checked(folder / spec["source"], spec["source_sha256"])
    classes = spec["classes"]
    source = np.array([r["values"] for r in src])
    learner = ClassReference(source, [r["label"] for r in src])
    frozen = PRIOR / site / query / "17"
    previous = read(frozen / "run.json")
    identity = digest([spec, previous["identity"], sha256(Path(__file__).read_bytes()).hexdigest(),
                       sha256((ROOT/"engineering/src/harness/methods/gas_class_reference.py").read_bytes()).hexdigest()])
    out = DEST / site / query
    receiver = LocalAcceptance()
    receiver.output = out / "accepted"
    history_x, history_y, seen, past, requests = [], [], [], [], []
    labels_set = list(range(1, classes + 1))
    for j, (b, event) in enumerate(zip(spec["batches"], previous["accepted"], strict=True)):
        rows = checked(folder / b["inputs"], b["inputs_sha256"])
        keys = [r["id"] for r in rows]
        values = np.array([r["values"] for r in rows])
        old = read(frozen / "accepted" / (event["request"] + ".json"))["payload"]
        if digest([previous["identity"], old]) != event["request"] or old["ids"] != keys:
            raise ValueError("旧回执与当前行不符")
        source_x = np.asarray(history_x).reshape(-1, values.shape[1])
        source_y = np.asarray(history_y, dtype=int)
        seen.append(values)
        pred = {k: learner.predict(np.vstack(seen), source_x, source_y, values, k).argmax(axis=1)+1 for k in MODES}
        if not np.array_equal(pred["原刻度"], old["prediction"]):
            raise ValueError("原训练与同回标旧回执预测不符")
        chosen = "独立刻度"
        if past:
            y = np.concatenate([v[0] for v in past])
            rates = {k: f1_score(y, np.concatenate([v[1][k] for v in past]), labels=labels_set,
                                 average="macro", zero_division=0) for k in MODES}
            # 同分优先共用较简单的刻度；所有历史输出来自取回那批答案之前。
            chosen = max(("独立刻度", *[k for k in MODES if k != "独立刻度"]), key=rates.get)
        pred["历史发布选择"] = pred[chosen]
        payload = {"ids": keys, "predictions": {k:v.tolist() for k,v in pred.items()}, "发布选择": chosen,
                   "已取得回标": len(history_y), "原查询回执": event["request"]}
        key = digest([identity, payload])
        if not receiver.submit(key,payload)["accepted"]:
            raise ValueError("新方案尚未交接")
        requests.append(key)
        picked = event["selected"]
        if picked:
            truth = checked(folder / b["feedback"], b["feedback_sha256"])
            ix = [keys.index(k) for k in picked]
            ans = [truth[k] for k in picked]
            history_x.extend(values[ix].tolist())
            history_y.extend(ans)
            past.append((ans,{k: p[ix] for k,p in pred.items()}))
    truth = read(folder / "evaluation.json")
    issued = {k:[] for k in pred}
    actual = []
    for key in requests:
        p = read(receiver.output / (key+".json"))["payload"]
        if digest([identity,p]) != key:
            raise ValueError("交接后预测改变")
        actual.extend(truth[k] for k in p["ids"])
        for k in issued:
            issued[k].extend(p["predictions"][k])
    table = {k: {"F1": float(f1_score(actual,v,labels=labels_set,average="macro",zero_division=0)),
                 "最差召回": float(recall_score(actual,v,labels=labels_set,average=None,zero_division=0).min())} for k,v in issued.items()}
    atomic_json(out/"run.json",{"requests": requests, "identity":identity, "scores":table, "回标":len(history_y)})
    return table


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已完成此阶段")
        results={site:run(site,relative) for site,relative in STREAMS.items()}
        means={k:{metric:float(np.mean([t[k][metric] for t in results.values()])) for metric in ("F1","最差召回")}
               for k in next(iter(results.values()))}
        fixed=max(MODES[:-1],key=lambda k:means[k]["F1"])
        candidate=max(("现场证据","历史发布选择"),key=lambda k:means[k]["F1"])
        delta=means[candidate]["F1"]-means[fixed]["F1"]
        decision={"固定":fixed,"现场":candidate,"现场增量":delta,
                  "继续配论文":delta>=.01 and means[candidate]["最差召回"]>=means[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":results,"means":means,"decision":decision})
        print(json.dumps({"同回标汇总":means,"取舍":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
