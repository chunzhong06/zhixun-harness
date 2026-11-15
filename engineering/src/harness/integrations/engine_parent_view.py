"""重用已交付发动机父版的冻结输入处理，向新学习器提供相同工况刻度。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from harness.contracts import digest
from harness.integrations.published_parent import ROOT, read, validate
from harness.models.full_condition_engine import FullConditionEngine
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _trainer
from harness.models.source_combinations import SourceCombinations


def transform(binding, raw, data_root):
    """不重新拟合刻度；返回标准化输入及完整父版预测，训练侧允许批量求值。"""
    validate(binding)
    if binding["task"] != "engine": raise ValueError("该输入视图只属于发动机父版")
    from engineering.cli.workflow import public_maintenance as public
    folder = ROOT / binding["run"]
    report = read(folder / "run.json")
    lookup = SourceCombinations.restore(read(folder / "source-combinations.json"))
    manifest, blob = ModelArtifacts(folder / "references").read(report["native_artifact"])
    spec = manifest["request"]
    if spec["repair"] != lookup.identity or digest([spec, manifest["sha256"]]) != report["reference_identity"]:
        raise ValueError("父版权重和输入处理不匹配")
    archive = Path(data_root) / "engine" / public.PACKAGES["engine"][0]
    from hashlib import sha256
    if sha256(archive.read_bytes()).hexdigest() != spec["package"]:
        raise ValueError("原模型源码包身份变化")
    with TemporaryDirectory() as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "C") as trainer:
        condition, center, model, checksum = FullConditionEngine.load_bundle(blob, trainer)
        if checksum != report["native_sha256"]: raise ValueError("实际父版权重变化")
        features = center.transform(condition.transform(lookup.transform(raw)))
        prediction = np.maximum(0., model.predict(features * model.scale + model.mean))
    return features, prediction


def program_predict(folder, raw, data_root):
    """按发布时固定的视图重载源码和权重；逐条正式输入不重估任何统计量。"""
    from hashlib import sha256
    import joblib
    from harness.integrations.generated_regressor import load_module
    from harness.integrations.published_parent import augment
    folder = Path(folder); manifest = read(folder / "program.json")
    if manifest["view_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("发布的输入视图实现变化")
    for name, key in (("source.py", "source_sha256"), ("model.joblib", "weights_sha256")):
        if sha256((folder / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("发布源码或权重变化")
    view = manifest["feature_view"]
    if view == "native":
        features, parent = transform(manifest["binding"], raw, data_root)
        x = np.column_stack((features, parent))
    elif view == "raw": x = augment(manifest["binding"], raw, data_root)
    else: raise ValueError("未登记的输入视图")
    load_module(folder / "source.py"); model = joblib.load(folder / "model.joblib")
    values = []
    for row in x:
        value = np.asarray(model.predict(row[None, :]), dtype=float)
        if value.shape != (1,) or not np.isfinite(value).all(): raise ValueError("预测须为单个有限值")
        values.append(max(0., float(value[0])))
    return np.asarray(values), None
