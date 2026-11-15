"""在 CPU 上从未知混合物的原始周期读取已发布的卷积预测。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from harness.models.sgp_cycle_network import CycleNetwork, normalize


def predictions(raw, weights):
    torch.set_num_threads(1)
    with np.load(raw, allow_pickle=False) as data:
        if set(data.files) != {"mixture_ids", "cycles", "cycle_ids"}:
            raise ValueError("服务输入只接收测量记录和分组编号")
        ids, codes, cycles = (data[k] for k in ("mixture_ids", "cycle_ids", "cycles"))
        if cycles.shape != (len(codes), 4, 1440) or not np.isfinite(cycles).all():
            raise ValueError("服务周期不完整")
        x = normalize(cycles)
    estimates = {}
    for name, item in weights.items():
        with Path(item["path"]).open("rb") as stream:
            if sha256(stream.read()).hexdigest() != item["sha256"]:
                raise ValueError("卷积权重校验失败")
            stream.seek(0)
            pack = torch.load(stream, map_location="cpu", weights_only=True)
        model = CycleNetwork().eval()
        model.load_state_dict(pack["weights"])
        with torch.no_grad():
            cycle_values = torch.cat([model(batch) for batch in x.split(128)]).numpy() * pack["scale"] + pack["center"]
        estimates[name] = [float(cycle_values[codes == i].mean(dtype=np.float64)) for i in ids]
    return estimates


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    print(json.dumps(predictions(args.input, json.loads(args.weights)), allow_nan=False))
