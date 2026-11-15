"""以现场回标训练局部修正。按批次提交预测，成功后才申请固定数量的标签。"""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from engineering.cli.workflow.field_classification import checked
from engineering.cli.workflow import gas_query_stream
from harness.contracts import digest
from harness.methods import gas_returned_errors, gas_response_votes, gas_rank_readout, gas_class_reference, feedback_query_risk
from harness.models import field_classification
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

OUTPUT = "相近读数残差"


def run(manifest, out, budget=8, seed=17, through=None):
    src, spec = manifest.parent, json.loads(manifest.read_bytes())
    source = checked(src / spec["source"], spec["source_sha256"])
    batches = spec["batches"]
    stop = len(batches) if through is None else through
    if not 0 <= stop <= len(batches) or budget < 0:
        raise ValueError("批次或回标预算无效")
    deps = (gas_returned_errors, gas_response_votes, gas_rank_readout, gas_class_reference, feedback_query_risk,
            field_classification, gas_query_stream)
    identity = digest([spec, budget, seed, Path(__file__).read_text(encoding="utf8"),
                       *[Path(m.__file__).read_text(encoding="utf8") for m in deps]])
    sx = np.asarray([r["values"] for r in source])
    sy = np.asarray([r["label"] for r in source])
    classes = spec["classes"]
    if set(sy) != set(range(1, classes+1)):
        raise ValueError("源端类别不完整")
    learner = gas_class_reference.ClassReference(sx, sy)
    svm = SVC(C=4, gamma="scale", class_weight="balanced", random_state=17).fit(learner.source, sy)
    rank = gas_rank_readout.RankReadout(sx, sy)
    corrector = gas_returned_errors.ReturnedErrors(classes)
    store = LocalStore(out / "checkpoint")
    receiver = LocalAcceptance()
    receiver.output = out / "accepted"
    with store.exclusive():
        state = store.load() or {"identity": identity, "events": []}
        if state["identity"] != identity or len(state["events"]) > stop:
            raise ValueError("输入版本与断点不同")
        observed, history, seen = [], [], {r["id"] for r in source}
        if len(seen) != len(source):
            raise ValueError("源端编号重复")
        for i,b in enumerate(batches[:stop]):
            rows = checked(src / b["inputs"], b["inputs_sha256"])
            ids = [r["id"] for r in rows]
            if len(set(ids)) != len(ids) or seen.intersection(ids) or any(set(r) != {"id","values"} for r in rows):
                raise ValueError("批次重复或输入带有额外字段")
            seen.update(ids)
            x = np.asarray([r["values"] for r in rows])
            observed.append(x)
            pool = np.vstack(observed)
            scale, base, px = gas_query_stream.train(source, history, classes)
            probs = base.predict_proba(scale.transform(x))
            hy = np.asarray([r["y"] for r in history], dtype=int)
            fixed = {name: learner.predict(pool, px, hy, x, name).argmax(1)+1 for name in ("原刻度", "独立刻度", "只用现场")}
            fixed.update(gas_response_votes.kernel_models(sx, sy, x, px, hy, pool, svm))
            bank = rank.predict(x, pool, px, hy)
            half = bank.shape[1] // 2
            hard = .4 * np.eye(classes)[fixed["固定合训核"] - 1]
            fixed["现场秩等权与合训核"] = 1 + (bank[:,half:].mean(1) + hard).argmax(1)
            fixed["两域秩固定各半与合训核"] = 1 + (bank.mean(1) + hard).argmax(1)
            votes = np.column_stack([fixed[k] for k in gas_returned_errors.TOOLS])
            corrected, scores, _ = corrector.predict(x, votes)
            payload = {"ids": ids, "prediction": corrected[OUTPUT], "已见回标": len(history)}
            key = digest([identity, payload])
            replay = i < len(state["events"])
            if replay:
                event = state["events"][i]
                stored = json.loads((receiver.output / (key + ".json")).read_bytes())
                if event["request"] != key or stored["request_id"] != key or stored["payload"] != payload:
                    raise ValueError("恢复后无法复现之前已发的分类")
            elif not receiver.submit(key,payload)["accepted"]:
                raise ValueError("批次未交付")
            selected = []
            if i + 1 < len(batches) and budget:
                local = StandardScaler().fit(x)
                view = np.clip(local.transform(x), -5, 5)
                past = np.clip(local.transform(px), -5, 5) if len(px) else px
                ix,_ = feedback_query_risk.select("均匀覆盖", ids, view, probs, budget, seed + i, past, [], [])
                selected = [ids[j] for j in ix]
                if len(set(selected)) != min(budget,len(ids)):
                    raise ValueError("实际索取数量有误")
                returned = checked(src / b["feedback"], b["feedback_sha256"])
                labels = np.asarray([returned[k] for k in selected],dtype=int)
                if replay and (selected != event["selected"] or labels.tolist() != event["labels"]):
                    raise ValueError("已返回反馈的内容或身份变化")
                corrector.receive(x[ix], votes[ix], labels, scores[ix])
                history.extend({"x": x[j].tolist(),"y": int(y)} for j,y in zip(ix,labels,strict=True))
            else:
                labels = np.empty(0,dtype=int)
            if not replay:
                state["events"].append({"request":key,"selected":selected,"labels":labels.tolist()})
                store.save(state)
        report = {"identity":identity, "events":state["events"], "returned":len(history)}
        atomic_json(out / "run.json", report)
        return report


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--stream",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--budget",type=int,default=8)
    cli.add_argument("--seed",type=int,default=17)
    cli.add_argument("--through",type=int)
    args = cli.parse_args()
    run(args.stream, args.output, args.budget, args.seed, args.through)
