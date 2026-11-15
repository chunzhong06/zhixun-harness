"""可恢复的逐批气体实验入口。预测持久化后才能请求限额内的标签。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from engineering.cli.workflow.field_classification import checked
from harness.contracts import digest
from harness.methods import feedback_query_risk as queries
from harness.models.field_classification import weights
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json


def train(source, history, classes):
    sx = np.asarray([r["values"] for r in source])
    sy = np.asarray([r["label"] for r in source])
    if set(sy) != set(range(1, classes + 1)):
        raise ValueError("源训练缺少声明的类别")
    x = np.asarray([r["x"] for r in history]).reshape(-1, sx.shape[1])
    hy = np.array([r["y"] for r in history], dtype=int)
    scale = StandardScaler().fit(sx)
    sample_weight = weights(sy, len(sy))
    if len(hy):
        sample_weight = np.r_[sample_weight, weights(hy, len(sy))]
    model = LogisticRegression(C=1, max_iter=1000, random_state=17).fit(
        scale.transform(np.vstack((sx, x))), np.r_[sy, hy], sample_weight=sample_weight)
    return scale, model, x


def run(manifest, out, method, budget, seed=17, through=None):
    root = manifest.parent
    spec = json.loads(manifest.read_bytes())
    batches = spec["batches"]
    stop = len(batches) if through is None else through
    if method not in queries.POLICIES or not 0 <= stop <= len(batches) or budget < 0:
        raise ValueError("无效的现场运行条件")
    source = checked(root / spec["source"], spec["source_sha256"])
    identity = digest([spec, method, seed, budget, sha256(Path(__file__).read_bytes()).hexdigest(),
                       sha256(Path(queries.__file__).read_bytes()).hexdigest()])
    receiver = LocalAcceptance()
    receiver.output = out / "accepted"
    store = LocalStore(out / "checkpoint")
    with store.exclusive():
        state = store.load() or {"identity": identity, "accepted": [], "history": []}
        if state["identity"] != identity or len(state["accepted"]) > stop:
            raise ValueError("接续条件与已发记录不同")
        source_ids = {r["id"] for r in source}
        if len(source_ids) != len(source):
            raise ValueError("源训练行重复")
        seen = set(source_ids)
        for event in state["accepted"]:
            record = json.loads((receiver.output / (event["request"] + ".json")).read_bytes())
            if digest([identity, record["payload"]]) != event["request"]:
                raise ValueError("已发记录身份改变")
            seen.update(record["payload"]["ids"])
        for i in range(len(state["accepted"]), stop):
            batch = batches[i]
            rows = checked(root / batch["inputs"], batch["inputs_sha256"])
            keys = [r["id"] for r in rows]
            if any(set(r) != {"id", "values"} for r in rows) or len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError("源和各批服务输入的身份重叠或含答案")
            x = np.array([r["values"] for r in rows])
            history = state["history"]
            scale, clf, past_x = train(source, history, spec["classes"])
            z = scale.transform(x)
            probabilities = clf.predict_proba(z)
            pred = clf.classes_[probabilities.argmax(axis=1)]
            payload = {"batch": batch["batch"], "ids": keys, "prediction": pred.tolist(), "已回标": len(history)}
            key = digest([identity, payload])
            reply = receiver.submit(key, payload)
            if not reply["accepted"] or reply["request_id"] != key:
                raise ValueError("接收端未确认")
            ix, trace = [], {}
            if i < len(batches) - 1 and budget:
                local = StandardScaler().fit(x)
                view = z if method == "BADGE" else np.clip(local.transform(x), -5, 5)
                past = np.clip(local.transform(past_x), -5, 5) if len(past_x) else past_x
                ix, trace = queries.select(method, keys, view, probabilities, budget, seed + i,
                                           past, [r["p"] for r in history], [r["y"] for r in history])
                if len(set(ix)) != min(budget, len(x)) or any(k < 0 or k >= len(x) for k in ix):
                    raise ValueError("查询越界或重复")
                answers = checked(root / batch["feedback"], batch["feedback_sha256"])
                for j in ix:
                    history.append({"id": keys[j], "x": x[j].tolist(), "p": probabilities[j].tolist(), "y": answers[keys[j]]})
            seen.update(keys)
            state["accepted"].append({"request": key, "selected": [keys[j] for j in ix], **trace})
            store.save(state)
        report = {"identity": identity, "accepted": state["accepted"], "returned": len(state["history"])}
        atomic_json(out / "run.json", report)
        return report


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--stream", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--method", choices=queries.POLICIES, required=True)
    cli.add_argument("--budget", required=True, type=int)
    cli.add_argument("--seed", type=int, default=17)
    cli.add_argument("--through", type=int)
    opts = cli.parse_args()
    run(opts.stream, opts.output, opts.method, opts.budget, opts.seed, opts.through)
