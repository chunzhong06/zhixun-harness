"""从当版官方赛题 ZIP 和 SDK 已保存的逐行预测，复算完整公开集指标。"""
import argparse
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np


OFFICIAL = {
    "7a992b92d92c807869c3eccf25b3bf82dcdeba9174e161d6e8b61d82c3ec8851": "A",
    "68db6a72c29bac11a4f86e97d9c1f524d35695464fd5f3f6d131f1431bc69d68": "B",
    "6cb8784e2cab7ee93453e756abf54c168721c667ca0c2c1327f5faae96e4211f": "C",
}


def score(archive, predictions):
    with archive.open("rb") as stream:
        task = OFFICIAL.get(file_digest(stream, "sha256").hexdigest())
    if task is None:
        raise ValueError("赛题 ZIP 的 SHA-256 与本版三项官方包的记录不符")
    with ZipFile(archive) as bundle:
        with bundle.open("public/validation.jsonl") as entries:
            rows = [json.loads(line) for line in entries]
        with np.load(BytesIO(bundle.read("checkpoint.npz")), allow_pickle=False) as source:
            meta = json.loads(str(source["metadata"]))
    with np.load(predictions, allow_pickle=False) as stored:
        ids = [row["id"] for row in rows]
        np.testing.assert_array_equal(stored["ids"], ids)
        values = stored["prediction"]
    expected_shape = (len(rows), len(meta["classes"])) if task == "A" else (len(rows),)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"{task} 预测应为有限数值，形状应为 {expected_shape}")
    answers = [row["answer"] for row in rows]
    if task == "A":
        answers = [meta["classes"].index(label) for label in answers]
    # 每个 CLI 进程只复算一个官方包，防止三个包内同名的 baselines 模块串用。
    sys.path.insert(0, str(archive))
    from baselines.train import metrics
    if str(archive) not in metrics.__code__.co_filename:
        raise RuntimeError("指标函数的来源与输入的官方包不符")
    frozen = SimpleNamespace(meta=meta, predict=lambda _: values)
    return {"task":task,"rows":len(rows),**metrics(frozen, {"x":None,"y":np.asarray(answers)})}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="本版赛方原始 ZIP")
    parser.add_argument("predictions", type=Path, help="含 ids 和 prediction 的 SDK 逐行预测 NPZ")
    args = parser.parse_args()
    print(json.dumps(score(args.archive.resolve(), args.predictions), ensure_ascii=False))
