"""从每个原始航班重走工作流入口。按六组回执评分，另用独立进程核对JSONL调用。"""
from hashlib import sha256
import json
import subprocess
import sys

import h5py
import joblib
import numpy as np

from score_flight_segment_precision import BASE, DEST as PRIOR, file_hash, receipt, LocalStore, LocalAcceptance, digest, atomic_json, measures
from score_flight_life_pairs import PREP, dataset_path, read_flights
from score_flight_complement import ROOT
from engineering.cli.workflow.flight_input_predict import infer_segments

DEST = BASE / "flight-segment-entry"
RULE = "直接用树方差倒数"
CLI = ROOT / "engineering/cli/workflow/flight_input_predict.py"


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("当前权重已有发放核对")
        fit = BASE / "flight-segment-training"
        learner = fit / "segment.joblib"
        if file_hash(learner) != json.loads((fit / "scores.json").read_bytes())["model"]:
            raise ValueError("训练产物身份不一致")
        bundle = {"prep": joblib.load(PREP).prep, "learner": joblib.load(learner)}
        artifact = DEST / "segment-release.joblib"
        joblib.dump(bundle, artifact)
        sha = file_hash(artifact)
        verify_entry(DEST, PRIOR, RULE, artifact, sha, "segment-precision", infer_segments)


def verify_entry(dest, prior_dir, rule, artifact, sha, method, infer):
    """同一完整航班重走实际入口，保留原回执及命令行复核。调用方持有目标目录锁。"""
    release = joblib.load(artifact)
    prior = json.loads((prior_dir / "scores.json").read_bytes())["sites"]
    receiver = LocalAcceptance()
    receiver.output = dest / "accepted"
    result, samples, reference = {}, [], []
    for site, info in prior.items():
        published = receipt(prior_dir, site, info)
        issued = {}
        with h5py.File(dataset_path(site)) as h:
            aux = h["A_test"][:, :2]
            ends = np.r_[0, np.flatnonzero(np.any(np.diff(aux, axis=0), axis=1)) + 1, len(aux)]
            for i, (start, end) in enumerate(zip(ends[:-1], ends[1:], strict=True)):
                sensors = h["X_s_test"][start:end]
                settings = h["W_test"][start:end]
                req = {"inputs": {"sensors": sensors, "settings": settings}}
                response, = infer(release, [req])
                uid = tuple(map(int, aux[start]))
                if uid in issued:
                    raise ValueError("航班重复")
                issued[uid] = response["outputs"]["value"]
                if i == len(ends) // 2:
                    samples.append({"inputs": {"sensors": sensors.tolist(), "settings": settings.tolist()}})
                    reference.append(issued[uid])
        ids = published["ids"]
        if set(issued) != {tuple(x) for x in ids}:
            raise ValueError("交付航班错位")
        values = [issued[tuple(k)] for k in ids]
        payload = {"ids": ids, "predictions": {rule: values}, "模型SHA256": sha}
        key = digest([site, payload])
        if not receiver.submit(key, payload)["accepted"]:
            raise ValueError("新入口交接失败")
        actual = receipt(dest, site, {"request": key})["predictions"][rule]
        np.testing.assert_allclose(actual, published["predictions"][rule], atol=1e-8, rtol=0)
        with h5py.File(dataset_path(site)) as h:
            labels = read_flights(h, "test", answers=True)
        y = np.array([labels[tuple(k)] for k in ids])
        result[site] = {"request": key, "航班数": len(ids), **measures(y, np.array(actual), np.asarray(ids)[:, 0])}
    lines = "\n".join(json.dumps(x, allow_nan=False) for x in samples)
    process = subprocess.run([sys.executable, "-B", str(CLI), "--method", method,
                              "--artifact", str(artifact), "--sha256", sha], input=lines, text=True,
                             capture_output=True, check=True, cwd=ROOT)
    values = [json.loads(v)["outputs"]["value"] for v in process.stdout.splitlines()]
    np.testing.assert_allclose(values, reference, atol=1e-8, rtol=0)
    total = {"机队等权误差": float(np.mean([v["rmse"] for v in result.values()])),
             "原始航班": sum(v["航班数"] for v in result.values()), "子进程样本": len(values)}
    atomic_json(dest / "scores.json", {"sites": result, "summary": total, "权重SHA256": sha,
                "入口SHA256": sha256(CLI.read_bytes()).hexdigest()})
    print(json.dumps({"逐航班发布": total}, ensure_ascii=False))

if __name__ == "__main__":
    run()
