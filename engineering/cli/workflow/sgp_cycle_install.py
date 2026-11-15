"""用共同标气装机，或从未知混合物的原始加热周期输出浓度及分类。"""

from argparse import ArgumentParser, Namespace
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.models.paired_cycle_teacher import INVERSE, PRIMARY, install_profiles


def read_cycles(path, calibration=False):
    with np.load(path, allow_pickle=False) as record:
        keys = {"cycle_ids", "cycles", "mixture_ids"}
        if calibration:
            keys.add("concentrations")
        if set(record.files) != keys:
            raise ValueError("原始数据包的字段与当前操作不符")
        mixture_ids = record["mixture_ids"]
        cycle_ids = record["cycle_ids"]
        cycles = record["cycles"]
        if (mixture_ids.ndim != 1 or not len(mixture_ids) or cycle_ids.ndim != 1
                or mixture_ids.dtype.kind not in "iu" or cycle_ids.dtype.kind not in "iu"
                or len(set(mixture_ids)) != len(mixture_ids) or set(cycle_ids) != set(mixture_ids)):
            raise ValueError("混合物身份与测量记录不一致")
        if cycles.shape != (len(cycle_ids), 4, 1440) or not np.isfinite(cycles).all():
            raise ValueError("每个周期需要四路传感器的1440个有效读数")
        groups = []
        for uid in mixture_ids:
            rows = cycles[cycle_ids == uid].astype(float).reshape(-1, 4, 12, 120)
            if len(rows) not in (3, 4):
                raise ValueError("每种混合物需要三或四个完整的保留周期")
            # 每段高温50个读数，低温70个读数，保留低温后半与前半的差。
            features = np.stack((rows[:, :, :, :50].mean(3), rows[:, :, :, 50:].mean(3),
                                 rows[:, :, :, 85:].mean(3) - rows[:, :, :, 50:85].mean(3)), axis=3)
            groups.append(features.reshape(len(rows), -1))
        y = record["concentrations"] if calibration else None
        if calibration and (y.shape != mixture_ids.shape or not np.isfinite(y).all()):
            raise ValueError("需为每种源标气提供唯一浓度")
    return mixture_ids, groups, y


def install(args):
    ids, source, y = read_cycles(args.source, calibration=True)
    target_ids, target, _ = read_cycles(args.install)
    index = {uid: i for i, uid in enumerate(ids)}
    if len(target_ids) < 5 or not set(target_ids) <= set(index):
        raise ValueError("装机至少需要五种在源端有已知浓度的共同配方")
    paired = np.asarray([index[uid] for uid in target_ids])
    pack = install_profiles(source, y, target, paired)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.artifact.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        joblib.dump(pack, temporary)
        with temporary.open("rb") as stream:
            identity = file_digest(stream, "sha256").hexdigest()
        temporary.replace(args.artifact)
    finally:
        temporary.unlink(missing_ok=True)
    return {"sha256": identity, "selected_strength": pack["selected"], "threshold": pack["threshold"]}


def predict(args):
    with args.artifact.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != args.sha256:
            raise ValueError("发布权重与给定身份不符")
        stream.seek(0)
        pack = joblib.load(stream)
    ids, groups, _ = read_cycles(args.input)
    fixed = pack["models"][1.].predict(groups)[INVERSE]
    selected = pack["models"][pack["selected"]].predict(groups)[PRIMARY]
    limit = pack["threshold"]
    return {"mixture_ids": ids.tolist(), "fixed": fixed.tolist(), "adaptive": selected.tolist(),
            "fixed_class": (1 + (fixed > limit)).tolist(), "adaptive_class": (1 + (selected > limit)).tolist()}


def predict_release(args):
    """按发布清单应用装机试验的取舍。现场不需要手选模型或读取标气真值。"""
    with args.artifact.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != args.sha256:
            raise ValueError("发布清单校验失败")
        stream.seek(0)
        release = json.load(stream)
    folder = args.artifact.parent
    calibration = release["calibration"]
    values = predict(Namespace(artifact=folder / calibration["path"], sha256=calibration["sha256"], input=args.input))
    weights = {k: {**v, "path": str((folder / v["path"]).resolve())} for k,v in release["networks"].items()}
    runner = os.getenv("ZHIXUN_TORCH_PYTHON", str(Path(sys.executable).parents[1] / "aegis_env/python.exe"))
    proc = subprocess.run([runner, "-B", str(Path(__file__).with_name("sgp_network_predict.py")),
                           "--input", str(args.input), "--weights", json.dumps(weights)],
                          stdout=subprocess.PIPE, text=True, encoding="utf-8", check=True)
    estimates = json.loads(proc.stdout)
    corrected = np.asarray(values["fixed"])
    result = {"mixture_ids": values["mixture_ids"]}
    for mode in ("fixed", "adaptive"):
        policy = release[mode]
        if policy["network"] is None:
            concentration = corrected
        else:
            pred = np.asarray(estimates[policy["network"]])
            concentration = (pred + corrected) / 2 if policy["blend"] else pred
        result[mode] = concentration.tolist()
        result[mode + "_class"] = (1 + (concentration > release["threshold"])).tolist()
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    setup = commands.add_parser("install")
    setup.add_argument("--source", required=True, type=Path)
    setup.add_argument("--install", required=True, type=Path)
    setup.add_argument("--artifact", required=True, type=Path)
    serve = commands.add_parser("predict")
    serve.add_argument("--artifact", required=True, type=Path)
    serve.add_argument("--sha256", required=True)
    serve.add_argument("--input", required=True, type=Path)
    combo = commands.add_parser("predict-release")
    combo.add_argument("--artifact", required=True, type=Path)
    combo.add_argument("--sha256", required=True)
    combo.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    actions = {"install": install, "predict": predict, "predict-release": predict_release}
    print(json.dumps(actions[args.action](args), allow_nan=False))
