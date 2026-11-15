"""对冻结的同一权重及输入核对六类概率，计量已加载模型逐条预测的开销。"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BASELINE_CPU_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT/"engineering/src"))
import numpy as np
from harness.models.gas_views import pair_model
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import atomic_json


def measure(package, control, output):
    if sha256(package.read_bytes()).hexdigest() != PACKAGE_SHA:
        raise ValueError("气体包与冻结模型不对应")
    sys.path.insert(0, str(package.resolve()))
    from baselines.model import Model
    state = json.loads((control/"state.json").read_bytes())
    result = json.loads((control/"result.json").read_bytes())
    ref = state["models"][state["current"]]["artifact"]
    checkpoint = control/"models"/ref["key"]/"pipeline.bin"
    model = pair_model(Model).load(checkpoint)
    with ZipFile(package) as z:
        rows = {r["id"]:r["inputs"]["features"] for r in map(json.loads,z.read("public/validation.jsonl").splitlines())}
    ids = list(result["holdout"]["predictions"])
    inputs = [np.asarray([rows[s]], dtype=np.float64) for s in ids]
    times = []
    fingerprint = None
    for repeat in range(8):
        start = perf_counter()
        probabilities = np.concatenate([model.predict(x) for x in inputs])
        elapsed = perf_counter()-start
        assert np.array_equal(probabilities.argmax(1)+1, [result["holdout"]["predictions"][s] for s in ids])
        run_hash = sha256(probabilities.astype("<f8").tobytes()).hexdigest()
        if repeat:
            assert run_hash == fingerprint
            times.append(elapsed)
        fingerprint = run_hash
    report = {"rows": len(ids), "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
              "source_sha256":sha256((ROOT/"engineering/src/harness/models/gas_views.py").read_bytes()).hexdigest(),
              "sample_order_sha256":sha256(json.dumps(ids).encode()).hexdigest(), "probabilities_sha256":fingerprint,
              "seconds":times, "median_seconds":float(np.median(times)), "numpy":np.__version__}
    atomic_json(output, report)
    print(report["median_seconds"], fingerprint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    parser.add_argument("--control", type=Path, default=ROOT/"results/experiments/official-gas/pilot-4/pair-shape-blend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measure(args.package, args.control, args.output)
