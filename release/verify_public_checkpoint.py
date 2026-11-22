"""在比赛镜像内，用同一官方检查点逐行复核 SDK 输出与原版指标。"""
import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from score_public import OFFICIAL


def check(archive, samples):
    with archive.open("rb") as stream:
        profile = OFFICIAL.get(file_digest(stream,"sha256").hexdigest())
    if profile is None:
        raise ValueError("输入归档与本版官方包记录不符")
    sys.path.insert(0, "/app")
    from contest_harness.cli import load_submission
    submission = load_submission()
    sys.path.insert(0, str(archive))
    from baselines.model import Model
    from baselines.predict import predict
    from baselines.train import metrics
    if str(archive) not in predict.__code__.co_filename:
        raise RuntimeError("未取得指定官方 ZIP 内的预测函数")
    with ZipFile(archive) as z:
        rows = [json.loads(line) for line in z.open("public/validation.jsonl")]
        raw = z.read("checkpoint.npz")
        expected = json.loads(z.read("baseline-info.json"))["checkpoint_sha256"].removeprefix("sha256:")
    if sha256(raw).hexdigest() != expected:
        raise ValueError("官方起始权重摘要不一致")
    if samples:
        rows = [rows[i] for i in np.linspace(0, len(rows)-1, min(samples,len(rows)), dtype=int)]
    values = []
    with TemporaryDirectory(dir="/tmp") as tmp:
        folder = Path(tmp)
        path = folder/"checkpoint.npz"
        path.write_bytes(raw)
        official = Model.load(path)
        for row in rows:
            answer = predict(official, row["inputs"])["outputs"]
            sample = SimpleNamespace(task=official.meta["task"], dataset_key=profile, inputs=row["inputs"])
            actual = submission.predict(sample, folder)["outputs"]
            if profile == "A":
                labels = official.meta["classes"]
                for output in (actual,answer):
                    np.testing.assert_array_equal([p["label"] for p in output],labels)
                actual = [p["score"] for p in actual]
                answer = [p["score"] for p in answer]
            else:
                actual, answer = actual["value"], answer["value"]
            np.testing.assert_allclose(actual, answer, rtol=0, atol=1e-10, err_msg=row["id"])
            values.append(actual)
    predictions = np.array(values)
    frozen = SimpleNamespace(meta=official.meta, predict=lambda _: predictions)
    truth = [r["answer"] for r in rows]
    if profile == "A":
        truth = [official.meta["classes"].index(label) for label in truth]
    scores = metrics(frozen, {"x":None, "y":np.array(truth)})
    result = dict(task=profile, rows=len(rows), **scores)
    if profile == "C":
        result["above_235"] = int(np.sum(predictions>235))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--samples", type=int, default=0, help="等距抽样条数，0 表示全部公开行")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("样本数不能为负")
    check(args.archive.resolve(), args.samples)
