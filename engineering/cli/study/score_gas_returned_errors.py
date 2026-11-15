"""逐批重放已登记的真实查询，发出本批预测后仅反馈当时选中的八条答案。"""
import json
from pathlib import Path
import numpy as np

from score_gas_rank_readout import DEST as PRIOR, BASE, STREAMS, QUERIES, QUERY_RUNS, checked, read, LocalAcceptance, LocalStore, atomic_json, digest, recall_score, f1_score
from harness.methods import gas_returned_errors as method
from harness.methods.gas_returned_errors import TOOLS, PRIMARY, ReturnedErrors

DEST = PRIOR.parent / "gas-returned-errors"


def evaluate(name, relative, query):
    src = BASE / relative
    spec = read(src / "stream.json")
    frozen = PRIOR / name / query
    run = read(frozen / "run.json")
    qdir = QUERY_RUNS / name / query / "17"
    qrun = read(qdir / "run.json")
    identity = digest([run["identity"], qrun["identity"], Path(method.__file__).read_text(encoding="utf8"),
                       Path(__file__).read_text(encoding="utf8")])
    receiver = LocalAcceptance()
    folder = DEST / name / query
    receiver.output = folder / "accepted"
    model = ReturnedErrors(spec["classes"])
    events = []
    for i,(b, old_key, selection) in enumerate(zip(spec["batches"], run["requests"], qrun["accepted"], strict=True)):
        old = read(frozen / "accepted" / (old_key + ".json"))["payload"]
        qkey = selection["request"]
        qp = read(qdir / "accepted" / (qkey + ".json"))["payload"]
        inputs = checked(src / b["inputs"], b["inputs_sha256"])
        ids = [r["id"] for r in inputs]
        if (digest([run["identity"], old]) != old_key or digest([qrun["identity"], qp]) != qkey or
                ids != old["ids"] or ids != qp["ids"] or old["已见回标"] != sum(map(len,model.y))):
            raise ValueError("旧回执或当前可见的标签数不同")
        x = np.array([r["values"] for r in inputs])
        votes = np.column_stack([old["预测"][k] for k in TOOLS])
        new, saved, chosen = model.predict(x, votes)
        payload = {"ids": ids, "预测": new, "选择": method.CORRECTIONS[chosen], "已见回标":old["已见回标"], "原预测回执":old_key}
        key = digest([identity, payload])
        if not receiver.submit(key, payload)["accepted"]:
            raise ValueError("修正预测未交接")
        events.append(key)
        selected = selection["selected"]
        if len(selected) != (min(8, len(ids)) if i < len(spec["batches"]) - 1 else 0) or len(set(selected)) != len(selected):
            raise ValueError("查询数量与冻结预算不符")
        if selected:
            returned = checked(src / b["feedback"], b["feedback_sha256"])
            ix = [ids.index(k) for k in selected]
            model.receive(x[ix], votes[ix], np.array([returned[k] for k in selected]), saved[ix])
    truth = read(src / "evaluation.json")
    y, columns, seen = [], {}, set()
    for key in events:
        p = read(receiver.output / (key + ".json"))["payload"]
        if digest([identity, p]) != key or seen.intersection(p["ids"]):
            raise ValueError("评分的预测错位或重复")
        seen.update(p["ids"])
        y.extend(truth[j] for j in p["ids"])
        for k, v in p["预测"].items():
            columns.setdefault(k, []).extend(v)
    if seen != set(truth):
        raise ValueError("逐批交付未覆盖所有服务测量")
    classes = list(range(1, spec["classes"] + 1))
    scores = {k: {"F1": float(f1_score(y,v,labels=classes,average="macro",zero_division=0)),
                  "最差召回": float(recall_score(y,v,labels=classes,average=None,zero_division=0).min())} for k,v in columns.items()}
    atomic_json(folder/"run.json", {"identity":identity,"requests":events,"scores":scores,"回标":sum(map(len,model.y))})
    return scores


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("这份开发回放已经完成")
        streams = {s:{q:evaluate(s,path,q) for q in QUERIES} for s,path in STREAMS.items()}
        old = read(PRIOR/"scores.json")
        means = {q:dict(old["means"][q]) for q in QUERIES}
        for q in QUERIES:
            for k in next(iter(streams.values()))[q]:
                means[q][k] = {m:float(np.mean([v[q][k][m] for v in streams.values()])) for m in ("F1","最差召回")}
        score = lambda arm: means[arm[0]][arm[1]]["F1"]
        fixed = [old["decision"]["最强可行固定"]]
        for q in ("随机","边界","均匀覆盖"):
            fixed.extend((q,k) for k in streams[next(iter(streams))][q] if k != PRIMARY)
        paper = max((("BADGE",k) for k in means["BADGE"] if k != "各流回看最强单通道"), key=score)
        fixed = max(fixed,key=score)
        main = ("均匀覆盖",PRIMARY)
        gain = score(main)-max(score(fixed),score(paper))
        decision = {"主候选":main,"固定":fixed,"同工具论文":paper,"现场F1":score(main),"固定F1":score(fixed),
                    "论文F1":score(paper),"最小领先":gain,"进独立":gain>=.01 and means[main[0]][main[1]]["最差召回"]>=means[fixed[0]][fixed[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":decision})
        print(json.dumps({"实际交付":decision},ensure_ascii=False))


if __name__ == "__main__":
    run()
