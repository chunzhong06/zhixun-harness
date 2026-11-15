"""让作者的同一冻结气体修补器覆盖此次比较的全部服务观测。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from zipfile import ZipFile

from compare_gas_patching import (ROOT, PACKAGE, SOURCE, author_predict, compile_bridge,
                                  dependencies, export_weights, score, atomic_json)
from harness.data.official_gas import LABELS


def main():
    folder = ROOT/"results/experiments/luna-comparison"
    # 只读固定对照的服务编号；预测先写入，再和标签匹配。
    baseline = [json.loads((folder/"gas/rounds/original"/f"{n:02d}.json").read_bytes())[
        "served_predictions"] for n in range(1,11)]
    ids = [k for block in baseline for k in block]
    jars = dependencies(SOURCE.parent.parent/"runtime-dependencies")
    dest = folder/"gas-patching"
    dest.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest) as temp, ZipFile(PACKAGE) as z:
        work = Path(temp)
        parent, weights = work/"checkpoint.npz", work/"weights.bin"
        parent.write_bytes(z.read("checkpoint.npz"))
        export_weights(parent,weights)
        train = [json.loads(line) for line in z.read("public/train.jsonl").splitlines()]
        valid = {r["id"]:r for r in map(json.loads,z.read("public/validation.jsonl").splitlines())}
        if len(set(ids)) != 414 or not set(ids) <= valid.keys():
            raise ValueError("待评分编号与逐轮对照不一致")
        preds,times = author_predict(work,compile_bridge(work,jars),weights,[r["inputs"]["features"] for r in train],
                        [LABELS.index(r["answer"]) for r in train],[valid[k]["inputs"]["features"] for k in ids])
    issued = dict(zip(ids,preds.tolist(),strict=True))
    atomic_json(dest/"issued.json",issued)
    labels = [LABELS.index(valid[k]["answer"])+1 for k in ids]
    diff = sum(issued[k]!=p for block in baseline for k,p in block.items())
    summary = {"source_commit":SOURCE.parent.name,"single_fit_rows":len(train),"rows":len(ids),
               "macro_f1":score(labels,preds.tolist()),"different_from_static":diff,"times_ns":times}
    atomic_json(dest/"scores.json",summary)
    print(summary)


if __name__ == "__main__":
    main()
