"""核对本地CPU后端、输入请求与未拟合对象保存重载；不fit、不预测、不联网。"""
import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.contracts import ActionPlan
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.training_tools import SklearnBackend, TrainingModelTools
from harness.models.model_artifacts import ModelArtifacts
from harness.data.package_input import read_package


def inspect_backend(manifest):
    """prepare只核对合法fit行；构造/编码不执行学习或模型推理。"""
    import joblib
    package = read_package(manifest, batch_index=0)
    batch, backend = package.batches[0], SklearnBackend()
    rows = tuple(s for s in batch.samples if s.role == "fit")
    checked = []
    temp = ROOT/"scratch"
    temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="backend-", dir=temp) as folder:
        tools = TrainingModelTools(package.task, backend, ModelArtifacts(Path(folder)/"models"))
        processors = ["none", "standard", "robust"]
        if package.task.condition_features:
            processors.append("condition_standard")
        for predictor in ("linear", "forest", "catboost"):
            for processor in processors:
                components = PipelineRecipe(processor=processor, predictor=predictor)
                action = ActionPlan("recent", pipeline=components)
                request = tools.prepare(package.task, batch.batch_id, action, rows)
                pipeline = backend._pipeline(package.task, components)
                data = BytesIO()
                joblib.dump(pipeline, data)
                data.seek(0)
                restored = joblib.load(data)
                if type(restored.named_steps["predictor"]) is not type(pipeline.named_steps["predictor"]):
                    raise ValueError("对象重载类型改变")
                checked.append({"recipe": asdict(components), "request_data_digest": request["data_digest"],
                                "serialized_bytes": len(data.getvalue()), "state": "unfitted"})
        try:
            backend.fit(package.task, batch.batch_id, ActionPlan("recent"), rows)
        except RuntimeError as exc:
            blocked = str(exc) == "真实训练与模型执行尚未启用"
        else:
            raise RuntimeError("执行门禁失效")
    return {"manifest": str(Path(manifest).resolve()), "task": asdict(package.task),
        "backend": backend.specification, "recipes": checked, "execution_blocked": blocked,
        "model_fitted": False, "model_predict_called": False, "network_called": False,
        "serialization_scope": "未拟合的本项目CPU流水线配置；真实权重一致性由后续真实运行核对"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_backend(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CPU后端构造、请求与未拟合对象重载已核对；实际执行保持关闭。")


if __name__ == "__main__":
    main()
