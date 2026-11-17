"""将已选拟合资料交给训练后端，保存完整流水线并按原版本重载预测。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import Protocol
import math
import platform

from harness.core.contracts import ActionPlan, Sample, TaskSpec, digest
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.core.recipe import CATALOG_VERSION, CATBOOST_DEFAULT_DEPTH
from harness.learning.models.corrections import CorrectedPipeline, combine, inside, validate_predicates


def gas_continuation_argv(action: ActionPlan, *, checkpoint, initial_parent, training, validation, stage, output):
    """构造随包训练器参数。调用方绑定实际父权重和合法训练/探索文件，再决定是否执行。"""
    if type(action.epochs) is not int or not 1 <= action.epochs <= 10000:
        raise ValueError("官方续训需要正整数训练轮数")
    return ["python", "-m", "baselines.train", "--profile", "A", "--train", str(training),
            "--validation", str(validation), "--from-baseline" if initial_parent else "--resume",
            str(checkpoint), "--epochs", str(action.epochs), "--stage-id", stage, "--output", str(output)]


class TrainingBackend(Protocol):
    """后端拥有处理器与预测器的构造及编码；每次fit创建独立完整对象。

    specification应包含配置、实现版本和依赖版本，且在本次运行保持稳定。
    evidence_kind区分真实模型与工程替身；predict只接收身份和特征。
    """
    evidence_kind: str

    @property
    def specification(self) -> dict: ...

    def fit(self, task: TaskSpec, batch_id: str, action: ActionPlan,
            rows: tuple[Sample, ...], *, initialize: bool) -> object: ...

    def dump(self, pipeline: object) -> bytes: ...

    def load(self, content: bytes) -> object: ...

    def predict(self, pipeline: object, inputs: tuple[dict, ...]) -> dict: ...


class TrainingModelTools:
    """实现既有ModelTools接口，数据选择与采用决定仍归Harness。

    训练只使用控制器提供的fit行；近期/历史配方共用本入口。artifact保存
    相对仓库的版本键，搬移运行时需一并搬移模型目录。每次预测从该版本
    重载完整对象，不在预测时重新拟合处理器或改写已发布文件。
    """
    def __init__(self, task: TaskSpec, backend: TrainingBackend, artifacts: ModelArtifacts):
        self.task, self.backend, self.artifacts = task, backend, artifacts
        self.evidence_kind = backend.evidence_kind
        self._specification = deepcopy(backend.specification)
        self.identity = "training-model-tools-v3:" + digest({
            "task": asdict(task), "backend": self._specification,
            "evidence_kind": self.evidence_kind,
        })

    @property
    def capabilities(self):
        return {"local_corrections": hasattr(self.backend, "fit_correction")}

    def _check_backend(self):
        if self.backend.specification != self._specification or self.backend.evidence_kind != self.evidence_kind:
            raise ValueError("训练后端配置或版本在运行期间发生变化")

    def prepare(self, task: TaskSpec, batch_id: str, action: ActionPlan,
                rows: tuple[Sample, ...], *, initialize: bool = False, base=None) -> dict:
        """准备可核查请求，不调用训练库；时间可用性已由批次接入层检查。"""
        self._check_backend()
        if task.signature != self.task.signature or not batch_id or not rows:
            raise ValueError("训练任务不匹配或拟合资料为空")
        if action.epochs is not None:
            raise ValueError("此后端按配方重新拟合，续训动作需要接受父检查点的工具")
        ActionPlan.parse(asdict(action))
        if len({s.sample_id for s in rows}) != len(rows):
            raise ValueError("拟合清单包含重复身份")
        for row in rows:
            if row.role != "fit":
                raise ValueError("训练工具只接受拟合资料")
            row.validate(task, row.available_at)
        from harness.adapters.datasets.data_conditions import training_condition
        condition=training_condition(task,rows,action)
        if not condition["ready"]:
            raise ValueError("拟合资料不满足当前预测器条件："+condition["reason"])
        request = {
            "task": asdict(task), "task_signature": task.signature,
            "batch_id": batch_id, "action": action.to_record(), "initialize": initialize,
            "backend": deepcopy(self._specification), "evidence_kind": self.evidence_kind,
            "training_versions": [f"{s.sample_id}@{s.revision}" for s in rows],
            "data_digest": digest([asdict(s) for s in rows]),
        }
        if action.correction:
            action.correction.validate_task(task)
            if initialize or not base or not self.capabilities.get("local_corrections"):
                raise ValueError("修正需要已保存的基础模型与对应后端")
            bundle, predicates = base["bundle"], base["predicates"]
            validate_predicates(predicates, len(task.features))
            if (bundle["model_id"] != action.correction.base_model_id or bundle.get("correction")
                    or bundle["task_signature"] != task.signature
                    or bool(predicates) != bool(action.correction.region_id)):
                raise ValueError("修正基础版本或范围不匹配；当前只允许一层")
            if len(rows) < 2 or any(s.sample_id in bundle["training_ids"] or not inside(s.values, predicates) for s in rows):
                raise ValueError("修正拟合需要至少两条范围内且未参与基础拟合的新资料")
            manifest, _ = self.artifacts.read(bundle["artifact"])
            prior = manifest["request"]
            if (prior["backend"] != self._specification or prior["task_signature"] != task.signature
                    or sorted(prior["training_versions"]) != sorted(bundle["training_versions"])):
                raise ValueError("基础模型内容、依赖或拟合来源不匹配")
            request["base"] = deepcopy(base)
            request["composition_work"] = {"base_prediction_rows": len(rows) if action.correction.mode == "residual" else 0,
                                            "embedded_base": True}
        elif base is not None:
            raise ValueError("普通重建不接收基础产物")
        return request

    def produce(self, request, action, rows):
        """由同一受控构建操作完成基础预测、修正拟合和整体保存，费用统一登记。"""
        if action.correction:
            _, content = self.artifacts.read(request["base"]["bundle"]["artifact"])
            base = self.backend.load(content)
            pipeline = self.backend.fit_correction(self.task, request["batch_id"], action, rows,
                                                    base, request["base"]["predicates"])
        else:
            pipeline = self.backend.fit(self.task, request["batch_id"], action, rows, initialize=request["initialize"])
        return self.backend.dump(pipeline)

    def build(self, task: TaskSpec, batch_id: str, action: ActionPlan,
              rows: tuple[Sample, ...], *, initialize: bool = False, base=None) -> dict:
        request = self.prepare(task, batch_id, action, rows, initialize=initialize, base=base)

        def produce():
            return self.produce(request, action, rows)

        return self.artifacts.create(request, produce)

    def predict(self, artifact: dict, inputs: tuple[dict, ...]) -> dict:
        self._check_backend()
        identifiers = []
        for row in inputs:
            if set(row) != {"sample_id", "values"}:
                raise ValueError("预测接口只接受身份与特征")
            if not isinstance(row["sample_id"], str) or not row["sample_id"]:
                raise ValueError("预测身份无效")
            if len(row["values"]) != len(self.task.features) or any(
                type(x) not in {int, float} or not math.isfinite(x) for x in row["values"]
            ):
                raise ValueError("预测特征与任务不兼容")
            identifiers.append(row["sample_id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("预测身份重复")
        manifest, content = self.artifacts.read(artifact)
        request = manifest["request"]
        if (request["task_signature"] != self.task.signature
                or request["backend"] != self._specification
                or request["evidence_kind"] != self.evidence_kind):
            raise ValueError("模型任务、训练配置或依赖版本不兼容")
        if not inputs:
            return {}
        pipeline = self.backend.load(content)
        predictions = self.backend.predict(pipeline, inputs)
        if set(predictions) != set(identifiers) or any(
            not self.task.accepts_target(value) for value in predictions.values()
        ):
            raise ValueError("训练后端返回了不兼容的预测")
        return predictions


@dataclass(frozen=True)
class SklearnRecipe:
    """接线用训练配置，分类用逻辑回归、连续目标用岭回归；未据此选定比赛基线。"""
    standardize: bool = True
    regularization: float = 1.0
    max_iter: int = 1000
    seed: int = 17

    def __post_init__(self):
        if type(self.standardize) is not bool or type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError("处理开关或迭代上限无效")
        if (type(self.regularization) not in {int, float}
                or not math.isfinite(self.regularization) or self.regularization <= 0):
            raise ValueError("正则强度须为有限正数")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("随机种子超出范围")


class SklearnBackend:
    """预备的scikit-learn训练/保存适配器，默认关闭实际执行。

    创建配置及准备请求只读取包版本，不导入训练库。正式启用需显式设置
    execution_enabled并由运行入口提供授权与资源控制；工程入口默认拒绝
    real_model。joblib只用于可信的自产流水线，环境版本由工具层核对。
    """
    evidence_kind = "real_model"

    def __init__(self, recipe: SklearnRecipe | None = None, *, execution_enabled: bool = False):
        self.recipe = recipe or SklearnRecipe()
        self.execution_enabled = execution_enabled

    @property
    def specification(self) -> dict:
        packages = {}
        for name in ("scikit-learn", "numpy", "scipy", "joblib"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        try:
            catboost_version = version("catboost")
        except PackageNotFoundError:
            catboost_version = None
        return {"implementation": "sklearn-pipeline-v5", "catalog_version": CATALOG_VERSION, "recipe": asdict(self.recipe),
                "python": platform.python_version(), "packages": packages,
                "optional_packages": {"catboost": catboost_version}}

    def _require_execution(self):
        if self.execution_enabled is not True:
            raise RuntimeError("真实训练与模型执行尚未启用")

    def _pipeline(self, task: TaskSpec, components=None):
        # 延迟导入使工程样例保持标准库环境；整个Pipeline只在已选fit行上拟合。
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if components is not None:
            components.validate_task(task)
            if components.predictor == "catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor
                factory = CatBoostClassifier if task.task_type == "classification" else CatBoostRegressor
                loss = (("MultiClass" if len(task.classes) > 2 else "Logloss")
                        if task.task_type == "classification" else
                        "MAE" if components.regression_loss == "mae" else "RMSE")
                # 固定迭代使用当前fit清单，不把explore/adopt答案交给作者脚本的早停入口。
                predictor = factory(iterations=components.trees, depth=components.max_depth or CATBOOST_DEFAULT_DEPTH,
                    learning_rate=components.learning_rate, l2_leaf_reg=components.regularization,
                    loss_function=loss, task_type="CPU", thread_count=1, random_seed=self.recipe.seed,
                    use_best_model=False, allow_writing_files=False, verbose=False)
            elif components.predictor == "forest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                factory = RandomForestClassifier if task.task_type == "classification" else RandomForestRegressor
                predictor = factory(n_estimators=components.trees, max_depth=components.max_depth,
                                    random_state=self.recipe.seed, n_jobs=1)
            else:
                predictor = (LogisticRegression(C=1 / components.regularization,
                    max_iter=components.max_iter, random_state=self.recipe.seed)
                    if task.task_type == "classification" else Ridge(alpha=components.regularization))
            processor = "passthrough"
            if components.processor == "standard":
                processor = StandardScaler()
            elif components.processor == "robust":
                from sklearn.preprocessing import RobustScaler
                processor = RobustScaler()
            elif components.processor == "condition_standard":
                from harness.learning.models.condition_processing import ConditionStandardScaler
                processor = ConditionStandardScaler(tuple(task.features.index(f) for f in task.condition_features),
                                                    task.condition_decimals)
            features = "passthrough"
            if components.features == "nonconstant":
                from sklearn.feature_selection import VarianceThreshold
                features = VarianceThreshold(threshold=0.)
            return Pipeline([("processor", processor), ("features", features), ("predictor", predictor)])

        predictor = (LogisticRegression(C=1 / self.recipe.regularization,
                                       max_iter=self.recipe.max_iter, random_state=self.recipe.seed)
                     if task.task_type == "classification"
                     else Ridge(alpha=self.recipe.regularization))
        processor = StandardScaler() if self.recipe.standardize else "passthrough"
        return Pipeline([("processor", processor), ("predictor", predictor)])

    def fit(self, task, batch_id, action, rows, *, initialize=False):
        self._require_execution()
        pipeline = self._pipeline(task, action.pipeline)
        fit_args = {}
        if action.pipeline and action.pipeline.class_weight == "balanced":
            # 统一采用sklearn的逆频率公式，不混用CatBoost不同归一化的auto_class_weights。
            counts = Counter(s.label for s in rows)
            fit_args["predictor__sample_weight"] = [len(rows) / (len(counts) * counts[s.label]) for s in rows]
        pipeline.fit([list(s.values) for s in rows], [s.label for s in rows], **fit_args)
        return pipeline

    def fit_correction(self, task, batch_id, action, rows, base, predicates):
        """只拟合修正分支；基础对象原样嵌入，回归目标为标签减基础预测。"""
        self._require_execution()
        from dataclasses import replace
        corrected_rows = rows
        if action.correction.mode == "residual":
            inputs = tuple({"sample_id": s.sample_id, "values": s.values} for s in rows)
            predicted = self.predict(base, inputs)
            corrected_rows = tuple(replace(s, label=s.label-predicted[s.sample_id]) for s in rows)
            if any(not math.isfinite(s.label) for s in corrected_rows):
                raise ValueError("残差数值超出有限范围")
        patch = self.fit(task, batch_id, replace(action, correction=None), corrected_rows)
        return CorrectedPipeline(base, patch, action.correction.mode, tuple(deepcopy(predicates)))

    def dump(self, pipeline) -> bytes:
        self._require_execution()
        import joblib
        content = BytesIO()
        joblib.dump(pipeline, content, compress=3)
        return content.getvalue()

    def load(self, content: bytes):
        self._require_execution()
        import joblib
        return joblib.load(BytesIO(content))

    def predict(self, pipeline, inputs):
        self._require_execution()
        if isinstance(pipeline, CorrectedPipeline):
            base = self.predict(pipeline.base, inputs)
            selected = tuple(row for row in inputs if inside(row["values"], pipeline.predicates))
            patch = self.predict(pipeline.correction, selected) if selected else {}
            return combine(base, patch, inputs, pipeline.mode, pipeline.predicates)
        predictor = getattr(pipeline, "named_steps", {}).get("predictor")
        # CatBoost.predict默认独立使用全部CPU线程，显式沿用训练侧的单线程约定。
        options = {"thread_count": 1} if type(predictor).__module__.split(".")[0] == "catboost" else {}
        values = pipeline.predict([row["values"] for row in inputs], **options)
        if len(values) != len(inputs):
            raise ValueError("预测器返回数量与输入不一致")
        # numpy标量转为JSON可保存的Python数值；合法目标由工具层校验。
        return {row["sample_id"]: value.item() if hasattr(value, "item") else value
                for row, value in zip(inputs, values)}
