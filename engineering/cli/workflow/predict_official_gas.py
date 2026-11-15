"""按官方 JSONL 协议预测。支持原始 NPZ 和 Harness 发布的自包含局部修正版本。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"engineering/src"))
from harness.models.official_gas_cpu import PACKAGE_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    args = parser.parse_args()
    if sha256(args.package.read_bytes()).hexdigest() != PACKAGE_SHA:
        raise ValueError("官方气体包的代码版本不符")
    sys.path.insert(0, str(args.package.resolve()))
    from baselines.model import Model
    from baselines.data import json_value
    from baselines.predict import predict
    from harness.models.gas_views import pair_model
    model = pair_model(Model).load(args.checkpoint)
    if model.meta.get("profile") != "A":
        raise ValueError("需要气体检查点")
    for line in sys.stdin:
        if line.strip():
            item = json_value(line)
            if not isinstance(item, dict) or set(item) - {"task","inputs"} or item.get("task","classification") != "classification":
                raise ValueError("需要官方气体分类输入")
            print(json.dumps(predict(model, item["inputs"]), allow_nan=False))


if __name__ == "__main__":
    main()
