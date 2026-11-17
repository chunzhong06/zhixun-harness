"""绑定已交付父版，把完整父版输出作为可编辑学习器的输入；不复制其预处理实现。"""
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np

from harness.core.contracts import digest
from harness.adapters.research.generated_regressor import load_module

ROOT = __import__("harness.paths", fromlist=["workspace_root"]).workspace_root()


def read(path): return json.loads(path.read_bytes())


def bind(run, task, ordinal):
    """父版属于实际接收回执，而不是当前目录中任意一个模型。"""
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT) or task not in ("gas", "engine"):
        raise ValueError("需要项目内已完成的气体或发动机交付")
    report = read(run / "run.json")
    if not 1 <= ordinal <= len(report["accepted"]): raise ValueError("父版轮次不存在")
    event = report["accepted"][ordinal - 1]
    receipt = run / "accepted" / (event["request"] + ".json")
    accepted = read(receipt); payload = accepted["payload"]
    if accepted["request_id"] != event["request"] or digest([report["identity"], payload]) != event["request"]:
        raise ValueError("父版交付身份不符")
    files = [run / "run.json", receipt]
    if task == "gas":
        snapshot = run / "snapshots" / (payload["snapshot"] + ".json")
        record = read(snapshot)
        files += [snapshot, snapshot.parent / "models" / (record["weights"] + ".npz")]
        parent = {"snapshot": snapshot.relative_to(ROOT).as_posix()}
    else:
        if report["policy"] != "continued-condition-native":
            raise ValueError("当前发动机接入针对已实测的六十遍原生权重")
        parent = {"native_artifact": report["native_artifact"], "native_sha256": report["native_sha256"]}
        files.append(run / "source-combinations.json")
    # 同一份权重配上变化后的输入处理并非同一个父版；直接消费者也参与绑定。
    owners = ["src/harness/learning/models/official_rul_cpu.py",
              "src/harness/learning/models/model_artifacts.py"]
    if task == "gas":
        owners += ["src/harness/learning/inference/gas_observer_predict.py",
                   "src/harness/learning/models/field_classification.py",
                   "src/harness/learning/inference/gas_views.py",
                   "src/harness/learning/models/gas_feedback_bank.py",
                   "src/harness/learning/models/observer_errors.py"]
    else:
        owners += ["src/harness/learning/inference/engine_input_predict.py",
                   "src/harness/learning/models/full_condition_engine.py",
                   "src/harness/learning/models/source_combinations.py"]
    files.extend(ROOT / name for name in owners)
    body = {"version": 1, "task": task, "run": run.relative_to(ROOT).as_posix(), "ordinal": ordinal,
            "receipt_id": event["request"], "parent": parent, "raw_width": 128 if task == "gas" else 24,
            "parent_columns": ["parent_probability_" + str(i) for i in range(1, 7)] + ["parent_label"] if task == "gas" else ["parent_value"],
            "files": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files},
            "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    return {**body, "identity": digest(body)}


def validate(binding):
    if binding.get("identity") != digest({k: v for k, v in binding.items() if k != "identity"}):
        raise ValueError("父版绑定内容变化")
    if binding["adapter_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("父版适配器实现已变化")
    for name, checksum in binding["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("父版文件身份变化：" + name)


def parent_outputs(binding, raw, data_root):
    """复用原有单行交付接口，因而保留工况修复、标准化、类别顺序和原权重。"""
    validate(binding)
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != binding["raw_width"] or not np.isfinite(raw).all():
        raise ValueError("输入宽度或数值不符合父版契约")
    if binding["task"] == "gas":
        from harness.learning.inference.gas_observer_predict import infer
        requests = ({"task": "classification", "inputs": {"features": row.tolist()}} for row in raw)
        outputs = list(infer(ROOT / binding["parent"]["snapshot"], Path(data_root), requests))
        probabilities = np.asarray([[x["score"] for x in row["outputs"]] for row in outputs])
        values = probabilities.argmax(axis=1) + 1
        return np.column_stack((probabilities, values))
    from harness.learning.inference.engine_input_predict import infer
    requests = ({"task": "regression", "inputs": {"sensors": row[:21].tolist(), "settings": row[21:].tolist()}} for row in raw)
    outputs = infer(ROOT / binding["run"], binding["ordinal"], Path(data_root), requests)
    return np.asarray([row["outputs"]["value"] for row in outputs])[:, None]


def augment(binding, raw, data_root):
    raw = np.asarray(raw, dtype=float)
    return np.column_stack((raw, parent_outputs(binding, raw, data_root)))


def program_predict(folder, raw, data_root):
    """源代码、权重和父版三者必须一起匹配；每次只向生成模型传当前一行。"""
    folder = Path(folder)
    manifest = read(folder / "program.json"); binding = manifest["binding"]
    source, weights = folder / "source.py", folder / "model.joblib"
    if sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"] or sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("可编辑程序或权重已变化")
    load_module(source); model = joblib.load(weights)
    x = augment(binding, raw, data_root); values = []; probabilities = []
    for row in x:
        value = np.asarray(model.predict(row[None, :]), dtype=float)
        if value.shape != (1,) or not np.isfinite(value).all(): raise ValueError("预测须为单个有限值")
        if binding["task"] == "gas":
            label = int(value[0])
            if value[0] != label or not 1 <= label <= 6: raise ValueError("分类结果须为1至6的整数")
            if hasattr(model, "predict_proba"):
                p = np.asarray(model.predict_proba(row[None, :]), dtype=float)
                if p.shape != (1, 6) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.):
                    raise ValueError("分类分数须按1至6排列并构成概率")
                if int(p.argmax()) + 1 != label: raise ValueError("类别与六类分数不一致")
                probabilities.append(p[0])
            else:
                probabilities.append(np.eye(6)[label - 1])
            values.append(label)
        else:
            values.append(max(0., float(value[0])))
    return np.asarray(values), np.asarray(probabilities) if probabilities else None


def reference_source(task):
    """无改动源码保持父版所有输出；后续驱动可在同一输入契约上替换学习器。"""
    classification = task == "gas"
    return '''from sklearn.base import BaseEstimator

class PreserveParent(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, -1]
''' + ('''
    def predict_proba(self, X):
        return X[:, -7:-1]
''' if classification else "") + '''
def build_model():
    return PreserveParent()
'''
