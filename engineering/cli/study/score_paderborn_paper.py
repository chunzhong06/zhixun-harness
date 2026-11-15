"""按作者 B17、20 频带、种子 0、含工况的原入口重放，保存退化阶段的实际归一化预测。"""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from harness.runtime.storage import atomic_json
from prepare_paderborn_author import DATA, REV, ROOT, destination


def main():
    manifest = json.loads((DATA / "manifest.json").read_bytes())
    out = ROOT / "results/experiments/local-defaults/whole-maintenance/paderborn-forward/paper"
    # 论文入口以工作目录定位论文数据。仅在临时目录复现作者布局，源文件保持固定摘要。
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        for row in manifest["files"]:
            path = row["path"]
            src = destination(path)
            if sha256(src.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"论文源输入与采集时不同 {path}")
            target = work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, target)
            except OSError:
                shutil.copyfile(src, target)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        args = ["main.py", "B17", "20", "0", "op_feature"]
        start = perf_counter()
        completed = subprocess.run([sys.executable, "-B", *args], cwd=work, env=env, check=True,
                                   capture_output=True, text=True)
        seconds = perf_counter() - start
        author_score = float(completed.stdout.strip().splitlines()[-1])
        path, = (work / "all_yhats").rglob("y_hat.npy")
        predicted = np.load(path, allow_pickle=False)
        n = len(predicted)
        truth = (n - np.arange(n) - 1) / n
        mae = float(np.abs(predicted - truth).mean())
        if not np.isclose(author_score, mae, atol=1e-12, rtol=0):
            raise ValueError("作者打印值与逐条预测复算不同")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out / "B17.npy")
    count = len(np.load(DATA / "fft/B17.npy", mmap_mode="r"))
    atomic_json(out / "scores.json", {"revision": REV, "args": args, "metric": "normalized_degradation_only_MAE",
                "mae": mae, "rmse": float(np.sqrt(np.mean((predicted - truth)**2))), "rows": n, "full_rows": count,
                "degradation_start_index": count - n, "training_devices": 16, "wall_seconds": seconds,
                "predictions_sha256": sha256((out / "B17.npy").read_bytes()).hexdigest()})
    print("B17", n, mae, flush=True)


if __name__ == "__main__":
    main()
