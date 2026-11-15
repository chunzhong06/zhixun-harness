"""在整日留出的 UCI 家庭测量上运行与四阵列相同的作者程序和本地模型传输。"""
from hashlib import sha256
import json

import numpy as np
from river.drift.binary import DDM
from sklearn.metrics import recall_score

from score_author_new_arrays import AwaitingResponse, IssuedForest, ObserverErrors, atomic_json, author_loop
from score_home_forward import CLASSES, EXPECTED, OUT, ROOT, load_recordings, macro


PROTOCOL = ROOT / "engineering/configs/experiments/home-author-completion.json"
NAMES = ("current", "logistic", "forest", "lda", "svm")


def main():
    x, y, train, days, _ = load_recordings()
    keys = [ids for _, ids in days]
    result = json.loads((OUT/"scores.json").read_bytes())
    if result["source_hash"] != EXPECTED or result["source_ids"] != train or max(map(len, keys)) > 32:
        raise ValueError("两组源数据或标签上限不同")
    base = OUT / "hllm"
    h, method, calls = author_loop("hllm", output=OUT, columns=list(x.columns),
                                   context="Identify why residential gas sensor stimulus classification may have degraded.")
    model = IssuedForest(keys, base / "paper")
    status = "complete"
    detail = None
    try:
        method(h, model, x.loc[train], y.loc[train], [x.loc[ids] for ids in keys],
               [y.loc[ids] for ids in keys], None, DDM(drift_threshold=.8,warm_start=3), 0, 0)
    except AwaitingResponse as missing:
        status, detail = "waiting_driver", str(missing)
    except (ValueError, SyntaxError, TypeError, KeyError) as error:
        status, detail = "run_failed", f"{type(error).__name__}: {error}"
    table = {}
    if status == "complete":
        if [r["keys"] for r in model.events] != keys:
            raise ValueError("作者没有给全部服务日期交付")
        observer = ObserverErrors(NAMES, 3)
        full = {"author": [], "author-calibrated": []}
        truth = []
        for j, (ids, released) in enumerate(zip(keys, model.events, strict=True), 1):
            source = OUT / f"issued-{j:02}.json"
            if sha256(source.read_bytes()).hexdigest() != result["issued_sha256"][j-1]:
                raise ValueError("冻结的项目输出变化")
            issued = json.loads(source.read_bytes())
            votes = {a: np.asarray(issued["predictions"][a]) for a in NAMES}
            votes["current"] = np.asarray(released["predictions"])
            chosen = observer.predict(votes, "uniform").tolist()
            atomic_json(base / f"issued-{j:02}.json", {"ids": ids, "author": released["predictions"],
                        "author-calibrated": chosen, "returned_after": issued["returned_after"]})
            idx = [ids.index(k) for k in issued["returned_after"]]
            if idx:
                observer.update(y.loc[issued["returned_after"]], {k: v[idx] for k, v in votes.items()})
            full["author"].extend(released["predictions"])
            full["author-calibrated"].extend(chosen)
            truth.extend(y.loc[ids].tolist())
        table = {a: {"macro_f1": macro(truth, p), "recall": recall_score(truth, p, labels=CLASSES,
                       average=None, zero_division=0).tolist()} for a, p in full.items()}
    summary = {"source_sha256": EXPECTED, "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
               "status": status, "detail": detail, "calls": calls, "issued_days": len(model.events),
               "fit_attempts": model.fit_attempts, "successful_fits": model.fit_count, "scores": table}
    atomic_json(base / "scores.json", summary)
    print(status, detail, len(model.events), table, flush=True)


if __name__ == "__main__":
    main()
