"""从正式父版逐轮真实推理，通过同一生成程序接口重发气体和发动机完整预测。"""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.workflow import public_maintenance as public
from harness.data.official_gas import read_official_gas
from harness.integrations.published_parent import augment, bind, parent_outputs, program_predict, read, reference_source, validate
from harness.models.field_classification import macro
from harness.runtime.storage import LocalStore, atomic_json


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--integrated", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/official")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    scores = {}; reports = {}; programs = []
    with LocalStore(args.output).exclusive(), TemporaryDirectory(dir=args.output, prefix="temporary-") as temporary:
        if (args.output / "scores.json").exists(): raise ValueError("不覆盖已有等价实测")
        # 原推理接口和隔离子进程的临时解包也归入本次目录，并由各自上下文关闭。
        os.environ["TEMP"] = temporary; os.environ["TMP"] = temporary
        import tempfile
        tempfile.tempdir = temporary
        for task in ("gas", "engine"):
            run = args.integrated / task; report = read(run / "run.json")
            state = read(run / "regular/state.json"); samples = state["samples"]
            fit_ids = sorted(k for k, row in samples.items() if row["role"] == "fit")[:64]
            if not fit_ids or any(samples[k]["label"] is None for k in fit_ids): raise ValueError("缺少合法源训练行")
            train_x = np.asarray([samples[k]["values"] for k in fit_ids]); train_y = np.asarray([samples[k]["label"] for k in fit_ids])
            ids_all = []; issued_all = []; expected_all = []; maximum_probability_difference = 0.
            for ordinal, event in enumerate(report["accepted"], 1):
                binding = bind(run, task, ordinal)
                payload = read(run / "accepted" / (event["request"] + ".json"))["payload"]
                ids = sorted(payload["predictions"])
                if set(ids) & set(fit_ids) or any(samples[k]["label"] is not None or samples[k]["role"] != "serve" for k in ids):
                    raise ValueError("父版输入不是合法无答案服务样本")
                raw = np.asarray([samples[k]["values"] for k in ids])
                expected = np.asarray([payload["predictions"][k] for k in ids])
                folder = args.output / task / f"round-{ordinal}"; folder.mkdir(parents=True)
                data = folder / "input.npz"
                augmented = augment(binding, raw, args.data_root)
                if not np.allclose(augmented[:, -1], expected, rtol=0, atol=1e-10): raise ValueError("原正式入口没有复现父版交付")
                np.savez_compressed(data, train_x=augment(binding, train_x, args.data_root), train_y=train_y,
                                    train_ids=np.asarray(fit_ids), predict_x=augmented, predict_ids=np.arange(len(ids)))
                program = folder / "program"
                pred_ids, fitted_prediction = fit_source(reference_source(task), data, program)
                if not np.array_equal(pred_ids, np.arange(len(ids))): raise ValueError("生成程序预测顺序改变")
                manifest = {"binding": binding, "source_sha256": sha256((program / "source.py").read_bytes()).hexdigest(),
                            "weights_sha256": sha256((program / "model.joblib").read_bytes()).hexdigest(),
                            "source_training_ids": fit_ids, "purpose": "父版能力等价，无新方案或提分主张"}
                atomic_json(program / "program.json", manifest)
                actual, probabilities = program_predict(program, raw, args.data_root)
                if not np.allclose(actual, expected, rtol=0, atol=1e-10) or not np.allclose(actual, fitted_prediction, rtol=0, atol=1e-10):
                    raise ValueError("同一可编辑接口未保持完整父版能力")
                if task == "gas":
                    difference = float(np.max(np.abs(probabilities - augmented[:, -7:-1])))
                    maximum_probability_difference = max(maximum_probability_difference, difference)
                    if difference != 0: raise ValueError("气体六类分数改变")
                # 真正走一条官方JSON命令入口，检验格式装配及模型重新加载。
                if ordinal == 1:
                    request = ({"task": "classification", "inputs": {"features": raw[0].tolist()}} if task == "gas" else
                               {"task": "regression", "inputs": {"sensors": raw[0, :21].tolist(), "settings": raw[0, 21:].tolist()}})
                    result = subprocess.run([sys.executable, "-B", str(ROOT / "engineering/cli/workflow/run_workflow.py"),
                        "published-program-predict", "--program", str(program.resolve()), "--data-root", str(args.data_root.resolve())],
                        input=json.dumps(request) + "\n", capture_output=True, text=True, encoding="utf-8", timeout=60)
                    if result.returncode: raise ValueError(result.stderr[-1200:])
                    response = json.loads(result.stdout)
                    value = 1 + int(np.argmax([r["score"] for r in response["outputs"]])) if task == "gas" else response["outputs"]["value"]
                    if not np.isclose(value, expected[0], rtol=0, atol=1e-10): raise ValueError("官方JSON入口改变预测")
                    atomic_json(folder / "entry-check.json", {"request": request, "response": response, "matched": True})
                    stale = deepcopy(binding); stale["ordinal"] += 1
                    try: validate(stale)
                    except ValueError: pass
                    else: raise ValueError("父版错配未被拒绝")
                np.savez_compressed(folder / "issued.npz", ids=np.asarray(ids), prediction=actual,
                                    parent_prediction=expected, probabilities=probabilities if probabilities is not None else np.empty((len(ids), 0)))
                programs.append({"task": task, "round": ordinal, "program": str(program), "parent_identity": binding["identity"], "rows": len(ids)})
                ids_all.extend(ids); issued_all.extend(actual.tolist()); expected_all.extend(expected.tolist())
            # 完整交付封存以后才用官方答案评分；生成程序只见源端64行训练答案。
            archive = args.data_root / task / ("cpu-a-guide-20261110-v1.zip" if task == "gas" else public.PACKAGES[task][0])
            package = read_official_gas(archive) if task == "gas" else public.PACKAGES[task][2](archive)
            answers = {r.sample_id: r.label for r in (*package.train, *package.validation)}
            y = np.asarray([answers[k] for k in ids_all]); prediction = np.asarray(issued_all)
            if task == "gas":
                matrix = np.zeros((6, 6), dtype=int); np.add.at(matrix, (y.astype(int) - 1, prediction.astype(int) - 1), 1)
                scores[task] = macro(matrix)
            else: scores[task] = float(np.sqrt(np.mean((y - prediction) ** 2)))
            reports[task] = {"rows": len(ids_all), "rounds": len(report["accepted"]), "equal_parent_predictions": bool(np.array_equal(prediction, expected_all)),
                             "maximum_probability_difference": maximum_probability_difference, "json_entry_verified": True, "stale_parent_rejected": True}
            print(json.dumps({"task": task, "score": scores[task], "issued": len(ids_all)}, ensure_ascii=False), flush=True)
        result = {"scope": "官方已有开发交付的完整父版能力等价；无新Luna提案，不构成性能提升或陌生任务验收", "scores": scores, "tasks": reports, "programs": programs}
        atomic_json(args.output / "scores.json", result)
        print(json.dumps({"gas_macro_f1": scores["gas"], "engine_rmse": scores["engine"], "parent_capabilities_preserved": True}, ensure_ascii=False))


if __name__ == "__main__": main()
