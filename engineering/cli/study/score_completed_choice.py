"""以参考设备失效为反馈时钟，冻结选择规则后给两个台架的新测量实际发放预测。"""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, digest, official, read, remaining_minutes, summary
from compare_public_selection import match_bearing_window
from engineering.cli.workflow.bearing_phase_predict import infer
from harness.data.official_bearing import CHANNELS, MEASURES
from harness.models.feedback_choice import HistoricalToolChoice
from score_completed_calibration import five
from score_unseen_bearing_windows import window_answers, window_inputs
from score_unseen_xjtu_devices import LOCAL, UNITS
from unseen_sources import FEMTO, bearing_data


PROTOCOL = ROOT / "engineering/configs/experiments/bearing-completed-choice.json"
DEST = OUT / "completed-choice"
PHASE = OUT / "life-scale/live/pronostia"


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def femto(offset):
    units, _, _ = bearing_data()
    known = remaining_minutes()
    # 失效时刻按采集计时器；同一分钟内的微秒抖动不用于提前开放反馈。
    failures = {"PHM12/" + r["unit"]: (r["end"] - 1) / 6 for r in read(FEMTO / "manifest.json")}
    history = [(r.unit_id, r.sample_id, match_bearing_window(r).values, known[r.sample_id])
               for r in five([row for group in units.values() for row in group])]
    rounds, metadata, ends = window_inputs(offset)
    service = [(key, "PHM12/" + metadata[key][0], (metadata[key][1] - 1) / 6, values)
               for group in rounds.values() for key, values in group]
    manifests = [FEMTO / "manifest.json", FEMTO / f"offset{offset}-manifest.json"]
    return failures, history, service, lambda: window_answers(metadata, ends), manifests


def xjtu(package, manifest):
    history = [(r.unit_id, r.sample_id, r.values, r.label) for r in five(package.validation)]
    failures = {}
    for row in package.validation:
        end = row.label + row.time_end - 1
        if failures.setdefault(row.unit_id, end) != end:
            raise ValueError("同设备的失效时刻不一致")
    data = read(manifest)
    service = []
    for entry in data["rows"]:
        key = entry["member"]
        p = Path(key)
        service.append((key, p.parent.name, int(p.stem) - 1, entry["features"]))
    if set(data["ends"]) != UNITS or set(failures) != {f"Bearing{i}_4" for i in (1, 2, 3)}:
        raise ValueError("反馈与服务设备不符合冻结协议")
    if manifest.name == "fifth-offset6.json":
        old = read(LOCAL / "fifth-devices.json")
        if {k for k, *_ in service} & {r["member"] for r in old["rows"]} or old["ends"] != data["ends"]:
            raise ValueError("新批次与原测量相交或终点变化")
    return failures, history, service, lambda: {k: data["ends"][u] - t - 1 for k, u, t, _ in service}, [manifest]


def forecast(inputs, tree, median, original):
    x = np.asarray(inputs)
    requests = ({"task": "regression", "inputs": {"features": {
        c: dict(zip(MEASURES, row[i*len(MEASURES):(i+1)*len(MEASURES)].tolist(), strict=True))
        for i, c in enumerate(CHANNELS)}}} for row in x)
    phase = np.asarray([record["outputs"]["value"] for record in infer(PHASE, requests)])
    direct = tree_values(tree, "bearing", x)
    values = {"original": original.predict(x), "phase": phase, "median": np.full(len(x), median),
              "tree": direct, "mean-of-three": (phase + median + direct) / 3}
    if any(v.shape != (len(x),) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("固定工具返回非有限值或行数不符")
    return values


def issue(site, stream, predict, identities, protocol):
    failures, history, service, answers, sources = stream
    order = protocol["tools"]
    decision = HistoricalToolChoice(order, protocol["default"])
    devices = {k: u for k, u, _, _ in service}
    if len(devices) != len(service) or set(devices.values()) & set(failures) or set(devices) & {r[1] for r in history}:
        raise ValueError("服务与参考设备重叠或测量重复")
    groups = {unit: [r for r in history if r[0] == unit] for unit in failures}
    if any(len(rows) != 5 for rows in groups.values()):
        raise ValueError("每台必须恰好五条已完成记录")
    # 批量算各固定工具的对照输出。选择器每次只收到刚开放的五条记录所对应的误差。
    service.sort(key=lambda r: (r[2], r[0]))
    predictions = predict([x for _, _, _, x in service])
    released = set()
    events = []
    choices = {}
    times = {}
    emitted = {a: {} for a in (*predictions, "once", "ongoing")}
    for i, (key, unit, now, _) in enumerate(service):
        ready = sorted(u for u, end in failures.items() if end < now and u not in released)
        if ready:
            losses = []
            records = {}
            for u in ready:
                rows = groups[u]
                estimates = predict([r[2] for r in rows])
                labels = np.asarray([r[3] for r in rows])
                losses.append({a: float(np.mean((estimates[a] - labels)**2)) for a in order})
                records[u] = {"failure_minute": failures[u], "ids": [r[1] for r in rows], "mse": losses[-1]}
            released.update(ready)
            means = decision.update(losses)
            events.append({"service_minute": now, "received": records, "mean_mse": means,
                           "once": decision.first, "ongoing": decision.active})
        choices[key] = decision.active
        times[key] = now
        for arm, v in predictions.items():
            emitted[arm][key] = float(v[i])
        emitted["once"][key] = emitted[decision.first][key]
        emitted["ongoing"][key] = emitted[decision.active][key]
    path = DEST / site / "issued.json"
    atomic_json(path, {"protocol_sha256": fingerprint(PROTOCOL), "sources": {p.name: fingerprint(p) for p in sources},
                       "models": identities, "reference_failures": failures, "events": events, "choice": choices,
                       "service_minutes": times, "device": devices, "predictions": emitted})
    # 交接文件落盘后评估者才打开服务答案；选择器没有标签参数。
    actual = read(path)["predictions"]
    truth = answers()
    if set(truth) != set(devices):
        raise ValueError("终点评分不覆盖同一批预测")
    scores = summary("bearing", truth, {"parent": actual["original"], **actual}, devices)
    del scores["parent"]
    fixed = min(order, key=lambda a: scores[a]["score"])
    accepted = (scores["ongoing"]["score"] <= .99 * min(scores[fixed]["score"], scores["once"]["score"])
                and scores["ongoing"]["worst_group"] <= 1.01 * scores[fixed]["worst_group"])
    result = {"issued_sha256": fingerprint(path), "arms": scores, "best_fixed": fixed,
              "meets_gate": accepted, "devices": len(set(devices.values())), "rows": len(service),
              "choices": dict(Counter(choices.values())), "feedback_rows": 5 * len(released),
              "training_calls": 0, "paid_calls": 0}
    atomic_json(path.with_name("scores.json"), result)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--femto-offset", type=int, choices=(48, 40), default=48)
    parser.add_argument("--xjtu-manifest", choices=("fifth-offset6.json", "fifth-devices.json"), default="fifth-offset6.json")
    args = parser.parse_args()
    package, checksum, batches = official("bearing")
    fit = [r for r in batches[0].samples if r.role == "fit"]
    median = float(np.median([r.label for r in fit]))
    trees, receipts = frozen("bearing")
    receipt = receipts["catboost"]
    if receipt["package"] != checksum or {r.unit_id for r in fit} & UNITS:
        raise ValueError("树的源包不符或训练设备与服务设备相交")
    protocol = read(PROTOCOL)
    identities = {"official": checksum, "tree": receipt["sha256"], "phase_run": fingerprint(PHASE / "run.json"),
                  "median": median}
    streams = {"femto": femto(args.femto_offset), "xjtu": xjtu(package, LOCAL / args.xjtu_manifest)}
    archive, = (ROOT / "data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "B") as trainer:
        path = Path(tmp) / "original.npz"
        path.write_bytes(package.checkpoint)
        original = trainer.Model.load(path)
        def predict(inputs):
            return forecast(inputs, trees["catboost"], median, original)
        results = {site: issue(site, stream, predict, identities, protocol) for site, stream in streams.items()}
    for site, result in results.items():
        print(site, result["rows"], result["choices"], {a: round(v["score"], 3) for a, v in result["arms"].items()})
    atomic_json(DEST / "scores.json", {"site_results": results, "adopt": all(x["meets_gate"] for x in results.values())})


if __name__ == "__main__":
    main()
