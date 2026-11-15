"""对首次纳入评分的 PHM08 截断设备，实测冻结模型在三十个循环内的预测变化。"""
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from check_released_engine import ARMS, DEST, ROOT, STUDY, ModelArtifacts, LocalStore, atomic_json, np, read
from engine_endpoint_check import PACKAGES, RAW, REGISTERED_SHA256
from unseen_sources import PHM08_TRAIN_SHA, fingerprints, track_rows

DATA = ROOT / "data/public/phm08"
OUTPUT = DEST / "decline30"
HORIZON = 30
SOURCES = {
    "test": "0109aa91930961e86f2e972ebb2887896281c18192db70c3ce77185b080e3850",
    "final_test": "8f6a857757a0bc0a2d04de0239b40e74fab72e313ade00342669c7364b475a7c",
}


def checked_bytes(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"{path} 来源摘要不符")
    return data


def paired_inputs():
    """先去除已见轨迹。模型只读当前行，文件名、设备编号及终点索引仅由评价侧持有。"""
    seen = set()
    if sha256(RAW.read_bytes()).hexdigest() != REGISTERED_SHA256:
        raise ValueError("旧 C-MAPSS 档案身份变化")
    with ZipFile(RAW) as old:
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            for split in ("train", "test"):
                for _, track in track_rows(old.read(f"{split}_{subset}.txt").splitlines()):
                    seen.update(fingerprints(track))
    previous = checked_bytes(DATA / "train.txt", PHM08_TRAIN_SHA)
    for _, track in track_rows(previous.splitlines()):
        seen.update(fingerprints(track))
    matrices, ids, exclusions = [], [], {}
    for split, checksum in SOURCES.items():
        excluded = {}
        for number, track in track_rows(checked_bytes(DATA / f"{split}.txt", checksum).splitlines()):
            name = f"{split}/{number}"
            marks = fingerprints(track)
            overlap = not marks.isdisjoint(seen)
            seen.update(marks)
            if overlap or len(track) <= HORIZON:
                excluded[name] = "overlap" if overlap else "too_short"
                continue
            # NASA 原始列为工况在前；赛方推理数组依次为所有传感器和三项工况。
            pair = track[[-1-HORIZON,-1]]
            matrices.append(np.column_stack((pair[:, 3:], pair[:, :3])))
            ids.append((name, len(track)-HORIZON, len(track)))
        exclusions[split] = excluded
    if any(not any(name.startswith(split+"/") for name, _, _ in ids) for split in SOURCES):
        raise ValueError("某一来源没有合格的设备")
    return ids, np.ascontiguousarray(np.vstack(matrices), dtype="<f8"), exclusions


def released_weights():
    folders = {arm: STUDY / "engine" / arm for arm in ARMS}
    once = ROOT / "results/experiments/local-defaults/proposal-frequency/streams/phm08"
    folders.update({arm+"_once":once / arm for arm in ("mature", "simple")})
    expected = read(DEST / "issued.json")["source"]["weights"]
    weights = {}
    for arm, folder in folders.items():
        rounds = read(folder / "run.json")["rounds"]
        last = rounds[-1]
        if len(rounds) != 10 or read(folder / last["submitted"])["payload"]["model_id"] != last["selected"]:
            raise ValueError(f"{arm} 末版与实际交接不符")
        _, blob = ModelArtifacts(folder / "models").read(last["model_artifact"])
        if arm in ARMS and sha256(blob).hexdigest() != expected[arm]:
            raise ValueError(f"{arm} 与旧终点对照的末版不同")
        weights[arm] = blob
    return weights


def issue():
    ids, inputs, exclusions = paired_inputs()
    weights = released_weights()
    package_name, package_sha, _ = PACKAGES["engine"]
    package = ROOT / "data/official/engine" / package_name
    checked_bytes(package, package_sha)
    sys.path.insert(0, str(package))
    from baselines.model import Model
    source = {"sources": SOURCES, "old_data": [PHM08_TRAIN_SHA, REGISTERED_SHA256],
              "input_tensor_sha256": sha256(inputs.tobytes()).hexdigest(), "package_sha256":package_sha,
              "weights": {k:sha256(v).hexdigest() for k,v in weights.items()}, "horizon":HORIZON}
    predictions = {}
    with TemporaryDirectory(prefix="engine-decline-") as temp:
        for arm, blob in weights.items():
            path = Path(temp) / f"{arm}.npz"
            path.write_bytes(blob)
            values = Model.load(path).predict(inputs)
            if values.shape != (len(ids)*2,) or not np.isfinite(values).all():
                raise ValueError(f"{arm} 没有逐行输出有限预测")
            predictions[arm] = values.reshape(-1,2).tolist()
    receipt = {"source": source, "devices": [list(row) for row in ids],
               "excluded": exclusions, "predictions": predictions}
    sealed = OUTPUT / "issued.json"
    if sealed.exists() and read(sealed) != receipt:
        raise ValueError("同一实验已有不同的封存预测")
    atomic_json(sealed, receipt)
    return sealed


def score(sealed):
    receipt = read(sealed)
    ids = receipt["devices"]
    true_change = np.array([late-early for _,early,late in ids])
    if np.any(true_change != HORIZON):
        raise ValueError("设备间隔与事前协议不符")
    table = {}
    for group in (*SOURCES, "pooled"):
        keep = np.array([group=="pooled" or name.startswith(group+"/") for name,_,_ in ids])
        arms = {}
        fixed = np.array(receipt["predictions"]["fixed"])[keep]
        fixed_abs = np.abs(fixed[:,0]-fixed[:,1]-true_change[keep])
        for arm, pairs in receipt["predictions"].items():
            early, late = np.array(pairs)[keep].T
            delta = early-late
            error = delta-true_change[keep]
            absolute = np.abs(error)
            arms[arm] = {"n":len(delta), "rmse":float(np.sqrt(np.mean(error**2))),
                         "mae":float(absolute.mean()), "mean_predicted_drop":float(delta.mean()),
                         "worst_absolute_error":float(absolute.max()),
                         "better_than_fixed":int(np.sum(absolute<fixed_abs)),
                         "worse_than_fixed":int(np.sum(absolute>fixed_abs))}
        table[group] = arms
    cleared = [arm for arm in receipt["predictions"]
               if all(table[s][arm]["rmse"] <= .99*table[s]["fixed"]["rmse"] for s in SOURCES)]
    atomic_json(OUTPUT / "scores.json", {"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),
                "target_drop":HORIZON, "constant_output_rmse":HORIZON, "elapsed_clock_rmse":0,
                "results":table, "cleared_relative_threshold":cleared})
    print({group:{arm:round(v["rmse"],3) for arm,v in arms.items()} for group,arms in table.items()})


if __name__ == "__main__":
    with LocalStore(OUTPUT).exclusive():
        score(issue())
