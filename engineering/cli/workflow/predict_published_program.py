"""以官方单行JSON接口运行已绑定父版的可编辑预测程序。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

from harness.data.official_gas import LABELS, prediction_request
from harness.integrations.published_parent import program_predict, read


def main(argv):
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/official"))
    args = ap.parse_args(argv)
    manifest = read(args.program / "program.json")
    task = manifest["binding"]["task"]
    requests = [json.loads(line) for line in sys.stdin if line.strip()]
    if task == "gas":
        values = []
        for item in requests:
            if set(item) - {"task", "inputs"} or item.get("task", "classification") != "classification" or set(item["inputs"]) != {"features"}:
                raise ValueError("需要官方分类输入")
            values.append(prediction_request(item["inputs"]["features"])["inputs"]["features"])
    else:
        from engineering.cli.workflow.engine_input_predict import input_rows
        values = [row[0] for row in input_rows(requests)]
    if not values: return
    predictor = program_predict
    if "feature_view" in manifest:
        if task != "engine": raise ValueError("该发布视图只支持发动机")
        from harness.integrations.engine_parent_view import program_predict as engine_predict
        predictor = engine_predict
    predictions, probabilities = predictor(args.program, np.asarray(values), args.data_root)
    for i, value in enumerate(predictions):
        output = ({"outputs": [{"label": label, "score": float(score)} for label, score in zip(LABELS, probabilities[i], strict=True)]}
                  if task == "gas" else {"outputs": {"value": float(value)}})
        print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)
