"""用作者原始 Patching 核验共同起点。依赖取固定版本，本入口不训练修补器。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.runtime.storage import atomic_json

PAPER_COMMIT = "02efdb34e4548ef8f11dec7473ad6e77e3f3d0ae"
SOURCE = ROOT / "references/implementations/SebWagner/Patching" / PAPER_COMMIT / "upstream-source.zip"
PACKAGE = ROOT / "data/official/gas/cpu-a-guide-20261110-v1.zip"
JARS = {
    "moa-2017.06.jar": "7fbf93f96267041d985a0dad2e7557de96c7756e037c0de8652876409c2d325a",
    "weka-dev-3.7.12.jar": "6325b7f3324312d03051da1534ed70a2aac8ed36ddf073fa5cbca068dfe5da1e",
    "java-cup-0.11a.jar": "9afcfd0996dcc9a933e66749988428ad964d8c1b678107fe688a6fa55325e17e",
    "pentaho-package-manager-1.0.8.jar": "c46e613a5a50246e3b61c3614df8a129462f039782a1e9882097a0ba3941e4ac",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_bridge(work, jars):
    """作者源文件直接编译；Weka 适配器单独编译，便于区分上游行为与接入改动。"""
    if sha(SOURCE) != "2053e7e4893c4143040aabeed5180b2bdf50168b3b6a0bfae51b227bc3f11311":
        raise ValueError("作者原件发生变化")
    classes = work / "classes"
    classes.mkdir()
    sources = []
    with zipfile.ZipFile(SOURCE) as archive:
        for entry in archive.namelist():
            if entry.endswith(".java"):
                src = work / Path(entry).name
                src.write_bytes(archive.read(entry))
                sources.append(src)
    sources.append(Path(__file__).parent / "batchwise_patching/OfficialGasBase.java")
    subprocess.run(["javac", "-proc:none", "--release", "8", "-encoding", "UTF-8",
                    "-cp", os.pathsep.join(map(str, jars)), "-d", str(classes), *map(str, sources)], check=True, cwd=work)
    return os.pathsep.join([str(classes), *map(str, jars)])


def dependencies(jar_dir):
    if sha(PACKAGE) != "7a992b92d92c807869c3eccf25b3bf82dcdeba9174e161d6e8b61d82c3ec8851":
        raise ValueError("官方气体包与已核准版本不一致")
    jars = [(jar_dir / name).resolve() for name in JARS]
    for jar in jars:
        if sha(jar) != JARS[jar.name]:
            raise ValueError(f"依赖版本不一致：{jar.name}")
    return jars


def export_weights(path, export):
    with np.load(path, allow_pickle=False) as params, export.open("wb") as stream:
        for name in ("mean", "scale", "w1", "b1", "w2", "b2"):
            stream.write(params[name].astype(">f8").tobytes(order="C"))


def audit(jar_dir):
    jars = dependencies(jar_dir)
    sys.path.insert(0, str(PACKAGE))
    from baselines.data import flatten
    from baselines.model import Model

    with TemporaryDirectory(prefix="gas-patching-start-") as tmp, zipfile.ZipFile(PACKAGE) as archive:
        work = Path(tmp)
        cp = compile_bridge(work, jars)
        checkpoint = work / "checkpoint.npz"
        checkpoint.write_bytes(archive.read("checkpoint.npz"))
        model = Model.load(checkpoint)
        weights = work / "weights.bin"
        export_weights(checkpoint, weights)

        train = [json.loads(row) for row in archive.read("public/train.jsonl").splitlines()]
        valid = [json.loads(row) for row in archive.read("public/validation.jsonl").splitlines()]
        inputs = []
        for row in train + valid:
            paths, values = flatten(row["inputs"])
            if paths != model.meta["paths"] or len(values) != 128:
                raise ValueError("气体输入与起点权重的字段顺序不同")
            inputs.append(values)
        features = np.asarray(inputs, dtype=np.float64)
        official = model.predict(features).argmax(axis=1)
        data = work / "features.bin"
        data.write_bytes(struct.pack(">I", len(inputs)) + features.astype(">f8").tobytes())
        pred = work / "author-output.bin"
        subprocess.run(["java", "-Xmx512m", "-cp", cp, "OfficialGasBase", str(weights),
                        str(work / "base.model"), str(data), str(pred)], check=True, cwd=work)
        observed = np.frombuffer(pred.read_bytes(), dtype="uint8")
        if not np.array_equal(official, observed):
            raise ValueError("关闭修补时作者入口与官方结果不一致")
        names = model.meta["classes"]
        # 仅训练集标签用于盘点错误区域能拿到的监督；不打开评价批来选参数。
        errors = {}
        for row, guess in zip(train, official):
            if row["answer"] != names[guess]:
                key = row["answer"] + " -> " + names[guess]
                errors[key] = errors.get(key, 0) + 1
        return {"author_commit": PAPER_COMMIT, "author_zip_sha256": sha(SOURCE),
                "bridge_sha256": sha(Path(__file__).parent / "batchwise_patching/OfficialGasBase.java"),
                "package_sha256": sha(PACKAGE), "parent_sha256": sha(checkpoint), "jars": JARS,
                "train_rows": len(train), "validation_rows": len(valid), "prediction_disagreements": 0,
                "paired_prediction_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                "training_errors_by_direction": errors, "patch_trained": False, "driver_called": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar-dir", type=Path, default=SOURCE.parent.parent / "runtime-dependencies")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = audit(args.jar_dir)
    if args.output:
        atomic_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False))
