"""核对原 NASA 测试轨迹能否检验已经冻结的两个官方格式模型。默认仅读输入及权重身份。"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"engineering/src")]
from engineering.cli.study.evaluate_rul_baseline import PACKAGES, input_paths
from harness.data.cmapss_input import REGISTERED_SHA256, read_cmapss, test_answers
from harness.data.official_engine import SUBSETS
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.storage import LocalStore, atomic_json

RAW = ROOT/"data/public/cmapss/CMAPSSData.zip"
FIT = ROOT/"results/experiments/official-engine/fit-comparison/fixed"
PREFLIGHT = ROOT/"results/inspections/engine/NASA终点输入核验.json"
OUTPUT = ROOT/"results/experiments/official-engine/nasa-endpoints"


def inspect_inputs():
    reference, expected, _ = PACKAGES["engine"]
    package = ROOT/"data/official/engine"/reference
    if sha256(package.read_bytes()).hexdigest() != expected:
        raise ValueError("官方包身份变化")
    fit = json.loads((FIT/"result.json").read_bytes())
    manifest, tuned = ModelArtifacts(FIT/"models").read(fit["candidate"])
    with ZipFile(package) as z:
        initial = z.read("checkpoint.npz")
        weights = {"original": initial, "balanced": tuned}
        hashes = {k: sha256(v).hexdigest() for k,v in weights.items()}
        if (hashes["original"] != manifest["request"]["checkpoint_sha256"] or
                hashes["balanced"] != manifest["sha256"] or manifest["request"]["package_sha256"] != expected or
                fit["parent"] != manifest["request"]["parent"]):
            raise ValueError("均衡版的构建来源或权重不匹配")
        for blob in weights.values():
            with np.load(io.BytesIO(blob), allow_pickle=False) as w:
                meta = json.loads(w["metadata"].item())
                if meta["task"] != "regression" or list(map(json.loads, meta["paths"])) != input_paths("engine"):
                    raise ValueError("两版权重须按 21 项传感器、3 项运行设置接收原始输入")
        tracks = {s: read_cmapss(RAW, subset=s, partition="test") for s in SUBSETS}
        source_train = {(s,t.unit): np.asarray(t.values, dtype="<f8") for s in SUBSETS
                        for t in read_cmapss(RAW, subset=s, partition="train")}
        end_features = {s: {np.asarray(t.values[-1], dtype="<f8").tobytes() for t in ts} for s,ts in tracks.items()}
        overlap = Counter()
        seen = defaultdict(set)
        split_units = defaultdict(set)
        for partition in ("train", "validation"):
            with z.open("public/"+partition+".jsonl") as stream:
                for line in stream:
                    r = json.loads(line)
                    m, x = r["meta"], r["inputs"]
                    s, u, c = m["source_subset"], m["unit"], m["cycle"]
                    unit = s,u
                    values = np.array(x["settings"] + x["sensors"], dtype="<f8")
                    if (m["source_split"] != "train" or unit not in source_train or
                            c in seen[unit] or not 1 <= c <= len(source_train[unit]) or
                            not np.array_equal(values, source_train[unit][c-1])):
                        raise ValueError("官方行未与指定 NASA 训练设备逐周期对应")
                    seen[unit].add(c)
                    split_units[partition].add(unit)
                    if values.tobytes() in end_features[s]:
                        overlap[s] += 1
        if split_units["train"] & split_units["validation"] or not all(split_units.values()):
            raise ValueError("官方两份数据未按完整设备分开")
    # 原始 train/test 各自从 1 编设备号。按同子集的实测轨迹比对，含改号后完全相同的前缀。
    starts = defaultdict(list)
    for (s,_), values in source_train.items():
        starts[s,values[0].tobytes()].append(values)
    duplicate_prefix = []
    fingerprints = {}
    for s,ts in tracks.items():
        digest = sha256()
        for t in ts:
            values = np.asarray(t.values, dtype="<f8")
            if any(np.array_equal(v[:len(values)], values) for v in starts[s,values[0].tobytes()]):
                duplicate_prefix.append(t.unit_id)
            digest.update(f"{s}:{t.unit}:{len(values)}\n".encode())
            digest.update(values[-1].tobytes())
        fingerprints[s] = digest.hexdigest()
    receipt = {"official_archive": expected, "raw_archive": REGISTERED_SHA256, "weights_sha256": hashes,
               "matched_official_rows": sum(map(len, seen.values())),
               "official_devices": {k: len(v) for k,v in split_units.items()},
               "source_test_devices": {s: len(ts) for s,ts in tracks.items()}, "endpoint_hashes": fingerprints,
               "test_endpoints_matching_official_rows_by_subset": {s:overlap[s] for s in SUBSETS},
               "test_prefixes_matching_train": duplicate_prefix,
               "metric": "最后一个已测周期，每台设备一个答案；主结果为所有设备等权 RMSE，另报四个子集的 RMSE。",
               "decision": "固定均衡版只有在整体 RMSE 比原版至少下降 1%、四个子集各自的 RMSE 均不恶化，才支持进一步验证跨设备采用。其余情况保留原版为跨来源参照。只用一次结果，不在这组终点调参。"}
    return receipt, tracks, weights, package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="需要新增寿命实测授权；使用已冻结的两个模型，各预测一次")
    enabled = parser.parse_args().predict
    receipt, tracks, weights, package = inspect_inputs()
    if receipt["test_prefixes_matching_train"] or any(receipt["test_endpoints_matching_official_rows_by_subset"].values()):
        raise ValueError("测试终点或轨迹重复，需重新判断验证范围")
    if not enabled:
        atomic_json(PREFLIGHT, receipt)
        return
    if not PREFLIGHT.exists() or json.loads(PREFLIGHT.read_bytes()) != receipt:
        raise ValueError("本次输入与事前核验不一致")
    with LocalStore(OUTPUT).exclusive():
        evaluate(tracks, weights, package)


def evaluate(tracks, weights, package):
    """两版共用同一台设备的同一个终点。先保存预测，重启只从匹配的回执续评分。"""
    if (OUTPUT/"scores.json").exists():
        raise ValueError("已使用这组终点作过评价")
    sys.path.insert(0, str(package))
    from baselines.model import Model

    # 输入和已发布预测都绑定到预检回执。中断后只重建尚未发布的预测，之后才读取答案。
    key = sha256(PREFLIGHT.read_bytes()).hexdigest()
    issued = OUTPUT/"issued.json"
    if issued.exists():
        record = json.loads(issued.read_bytes())
        if record["input"] != key:
            raise ValueError("已有预测来自另一份预检")
        predictions = record["predictions"]
    else:
        flat = [t for ts in tracks.values() for t in ts]
        x = np.asarray([(*t.values[-1][3:], *t.values[-1][:3]) for t in flat])
        predictions = {}
        with TemporaryDirectory(prefix="nasa-endpoints-") as tmp:
            for k,v in weights.items():
                path = Path(tmp)/(k+".npz")
                path.write_bytes(v)
                ans = Model.load(path).predict(x)
                if ans.shape != (len(flat),) or not np.isfinite(ans).all():
                    raise ValueError("终点预测数量或数值无效")
                predictions[k] = dict(zip((t.unit_id for t in flat), ans.tolist(), strict=True))
        atomic_json(issued, {"input":key, "predictions":predictions})
    truth = {s:test_answers(RAW, s, ts) for s,ts in tracks.items()}
    total = {i:v for group in truth.values() for i,v in group.items()}
    if set(predictions) != set(weights) or any(set(p) != set(total) for p in predictions.values()):
        raise ValueError("预测、答案或冻结模型编号不符")
    scores = {k:{s:float(np.sqrt(np.mean([(p[i]-v)**2 for i,v in vals.items()])))
                 for s, vals in {**truth, "pooled":total}.items()} for k,p in predictions.items()}
    atomic_json(OUTPUT/"scores.json", {"input":key, "issued":sha256(issued.read_bytes()).hexdigest(), "rmse":scores})


if __name__ == "__main__":
    main()
