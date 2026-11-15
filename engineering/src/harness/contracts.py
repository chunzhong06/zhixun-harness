"""公共记录与可替换接口；官方字段在平台适配器中转换。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol
import hashlib
import json
import math
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.corrections import CorrectionSpec


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    """声明目标、指标及输入语义，随模型和反馈保存。

    diagnostic_features限定联合区域检查范围；condition_features须预测时
    已知，condition_decimals固定其分组精度。三者均进入任务与恢复身份。
    """
    task_id: str
    features: tuple[str, ...]
    classes: tuple[int, ...] = ()
    metric: str = "accuracy"
    task_type: str = "classification"
    target_unit: str | None = None
    diagnostic_features: tuple[str, ...] = ()
    condition_features: tuple[str, ...] = ()
    condition_decimals: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id or not self.features:
            raise ValueError("任务与输入字段必须明确")
        if any(not isinstance(f, str) or not f for f in self.features) or len(set(self.features)) != len(self.features):
            raise ValueError("输入字段应为互不重复的名称")
        if self.task_type == "classification":
            if self.metric not in {"accuracy", "macro_f1"} or not self.classes or any(type(c) is not int for c in self.classes):
                raise ValueError("分类任务需要整数类别及支持的分类指标")
            if len(set(self.classes)) != len(self.classes):
                raise ValueError("类别不能重复")
        elif self.task_type == "regression":
            if self.metric not in {"mae", "rmse"} or self.classes:
                raise ValueError("数值预测需声明mae或rmse，不填写分类类别")
        else:
            raise ValueError("未知预测任务类型")
        if self.target_unit is not None and (not isinstance(self.target_unit, str) or not self.target_unit.strip()):
            raise ValueError("目标单位应为非空字符串或未声明")
        for names in (self.diagnostic_features, self.condition_features):
            if len(set(names)) != len(names) or any(n not in self.features for n in names):
                raise ValueError("诊断与工况字段须来自任务输入且互不重复")
        if len(self.diagnostic_features) > 16:
            raise ValueError("联合区域诊断最多明确选择16个字段")
        if len(self.condition_features) != len(self.condition_decimals) or any(
                type(n) is not int or not 0 <= n <= 6 for n in self.condition_decimals):
            raise ValueError("工况字段须逐项声明0至6位分组精度")

    @classmethod
    def parse(cls, value: dict) -> TaskSpec:
        return cls(**{**value, "features": tuple(value["features"]),
                      "classes": tuple(value.get("classes", ())),
                      "diagnostic_features": tuple(value.get("diagnostic_features", ())),
                      "condition_features": tuple(value.get("condition_features", ())),
                      "condition_decimals": tuple(value.get("condition_decimals", ()))})

    @property
    def objective(self) -> dict:
        return {"task_type": self.task_type, "metric": self.metric,
                "direction": "maximize" if self.task_type == "classification" else "minimize",
                "target_unit": self.target_unit}

    def accepts_target(self, value) -> bool:
        if self.task_type == "classification":
            return type(value) is int and value in self.classes
        return type(value) in {int, float} and math.isfinite(value)

    def utility(self, score: float) -> float:
        """只在排序及计算改善量时统一方向；对外分数保留原指标和单位。"""
        return score if self.task_type == "classification" else -score

    @property
    def signature(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Sample:
    """平台适配器给出的某次样本修订。

    available_at是该修订及其标签可用的批次序号，非文件重发时刻。
    group仅用于报告和历史覆盖；类别由label给出，独立设备/轨迹由unit_id声明。
    time_start/end表示该输入实际使用的观察区间，不表示标签可见时刻。
    serve行只有输入，答案通过后续更高revision返回。
    """
    sample_id: str
    revision: int
    values: tuple[float, ...]
    label: int | float | None
    role: str
    group: str
    available_at: int
    unit_id: str | None = None
    time_start: float | None = None
    time_end: float | None = None

    @classmethod
    def parse(cls, row: dict) -> Sample:
        return cls(**{**row, "values": tuple(row["values"])})

    def validate(self, task: TaskSpec, ordinal: int) -> None:
        if self.unit_id is not None and (not isinstance(self.unit_id,str) or not self.unit_id):
            raise ValueError("独立单位须为非空身份或未声明")
        times=(self.time_start,self.time_end)
        if any(v is not None for v in times):
            if self.unit_id is None or any(type(v) not in {int,float} or not math.isfinite(v) for v in times) or self.time_end < self.time_start:
                raise ValueError("时间区间需要独立单位和有效起止值")
        if not self.sample_id or self.revision < 1:
            raise ValueError("样本身份或修订号无效")
        if self.role not in {"fit", "explore", "adopt", "serve"}:
            raise ValueError("未知的数据用途")
        if not 0 <= self.available_at <= ordinal:
            raise ValueError("资料包含当前尚不可使用的信息")
        if len(self.values) != len(task.features) or not all(
            type(x) in {int, float} and math.isfinite(x) for x in self.values
        ):
            raise ValueError("输入字段或数值无效")
        if self.role != "serve" and not task.accepts_target(self.label):
            raise ValueError("训练与评价样本需要合法标签")
        if self.role == "serve" and self.label is not None:
            raise ValueError("服务输入的答案应通过后续反馈提供")


@dataclass(frozen=True)
class BatchContext:
    """一次调用的输入快照，可包含累计重发的旧样本；批次身份须稳定。"""
    task_id: str
    batch_id: str
    ordinal: int
    samples: tuple[Sample, ...]

    @classmethod
    def parse(cls, row: dict) -> BatchContext:
        return cls(row["task_id"], row["batch_id"], row["ordinal"],
                   tuple(Sample.parse(s) for s in row["samples"]))


@dataclass(frozen=True)
class GasPairPatch:
    """仅在原预测为 reported 时允许恢复 target，两者均使用任务公开的类别编号。"""
    reported: int
    target: int
    method: str

    def __post_init__(self):
        if (any(type(v) is not int or v not in range(1, 7) for v in (self.reported, self.target))
                or self.reported == self.target or self.method not in ("prior", "raw-knn", "shape-knn", "shape-blend")):
            raise ValueError("气体定向复查的类别或方法不符")


@dataclass(frozen=True)
class ActionPlan:
    """资料配方与可选完整组件组合；pipeline为空时沿用当前版本的已知配方。

    初始化或当前版本由后端定义配置时，空值交给同一后端默认配置。
    普通后端重拟合流水线；epochs由支持检查点续训的工具接收。
    """
    recipe: str
    ratio: float = 0.0
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    pipeline: PipelineRecipe | None = None
    method_id: str | None = None
    correction: CorrectionSpec | None = None
    epochs: int | None = None  # 仅用于从父检查点续训的工具，按额外训练轮数计。
    focus_label: int | None = None  # 由工具开放的类别编号；在固定训练量内提高该类出现次数。
    gas_pair: GasPairPatch | None = None
    fit_manifest: str | None = None  # 后端登记的清单名称；不接受外部路径或样本编号。

    @classmethod
    def parse(cls, row: dict) -> ActionPlan:
        refs = row.get("evidence_refs", ())
        if (not isinstance(refs, (list, tuple)) or len(refs) > 8
                or any(not isinstance(ref, str) or not ref for ref in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError("计划依据应为互不重复的事实编号列表")
        pipeline = PipelineRecipe.parse(row["pipeline"]) if row.get("pipeline") is not None else None
        correction = CorrectionSpec(**row["correction"]) if row.get("correction") is not None else None
        pair = GasPairPatch(**row["gas_pair"]) if row.get("gas_pair") is not None else None
        plan = cls(**{**row, "evidence_refs": tuple(refs), "pipeline": pipeline, "correction": correction, "gas_pair": pair})
        if plan.fit_manifest is not None and (
                not isinstance(plan.fit_manifest, str) or not plan.fit_manifest
                or plan.recipe != "recent" or plan.epochs is None or plan.focus_label is not None
                or pipeline or correction or pair or plan.method_id):
            raise ValueError("冻结清单仅用于独立的续训动作")
        if pair and (plan.recipe != "recent" or plan.epochs is not None or pipeline or correction or plan.method_id):
            raise ValueError("局部复查使用已确认父版本和原训练清单")
        if plan.focus_label is not None and (type(plan.focus_label) is not int or plan.epochs is None):
            raise ValueError("定向续训必须指定类别整数和训练遍数")
        if plan.epochs is not None and (type(plan.epochs) is not int or not 1 <= plan.epochs <= 10000
                                         or pipeline or correction):
            raise ValueError("续训轮数或组件组合与随包工具不兼容")
        if plan.recipe not in {"recent", "uniform", "coverage"}:
            raise ValueError("计划使用了未开放的维护动作")
        allowed = {0.0} if plan.recipe == "recent" else {0.5, 1.0}
        if type(plan.ratio) not in {int, float} or plan.ratio not in allowed or not isinstance(plan.reason, str):
            raise ValueError("计划用量不在共同目录中")
        if plan.method_id is not None and (not isinstance(plan.method_id, str) or not plan.method_id):
            raise ValueError("方法来源应为非空身份")
        if plan.correction and (plan.pipeline is None or plan.method_id is not None):
            raise ValueError("修正动作须明确分支完整配方，不附普通重训方法身份")
        return plan

    @property
    def key(self) -> str:
        suffix = "-p" + self.pipeline.key if self.pipeline else ""
        if self.correction:
            suffix += "-c" + digest(asdict(self.correction))[:16]
        if self.epochs is not None:
            suffix += f"-epochs{self.epochs}"
        if self.focus_label is not None:
            suffix += f"-focus{self.focus_label}"
        if self.gas_pair:
            suffix += f"-{self.gas_pair.method}-{self.gas_pair.reported}-to-{self.gas_pair.target}"
        if self.fit_manifest is not None:
            suffix += f"-fit-{self.fit_manifest}"
        return f"{self.recipe}-{self.ratio:g}" + suffix

    def to_record(self):
        """返回与JSON落盘一致的动作形态，恢复后引用仍保持列表语义。"""
        record = {**asdict(self), "evidence_refs": list(self.evidence_refs)}
        if self.epochs is None:
            record.pop("epochs")  # 普通动作保留既有序列化身份和恢复键。
        if self.focus_label is None:
            record.pop("focus_label")
        if self.gas_pair is None:
            record.pop("gas_pair")
        if self.fit_manifest is None:
            record.pop("fit_manifest")
        return record


@dataclass(frozen=True)
class ModelBundle:
    """完整版本的来源与适配器产物。

    training_ids用于排除已拟合样本，training_versions用于核对修订来源。
    artifact是可JSON保存的工具引用，可指向样例表或完整流水线文件。
    工具负责保存、核对和重新加载实际内容；运行状态保存采用关系。
    pipeline记录实际组件配方，供下一批普通重训沿用；None表示配置由后端定义。
    """
    model_id: str
    task_signature: str
    parent_id: str | None
    training_ids: tuple[str, ...]
    training_versions: tuple[str, ...]
    artifact: dict
    evidence_kind: str
    pipeline: PipelineRecipe | None = None
    correction: dict | None = None

    @classmethod
    def parse(cls, value: dict) -> ModelBundle:
        return cls(**{**value, "pipeline": PipelineRecipe.parse(value["pipeline"]) if value.get("pipeline") else None,
                      "training_ids": tuple(value["training_ids"]),
                      "training_versions": tuple(value["training_versions"])})


@dataclass(frozen=True)
class EngineConfig:
    """公共开关与样例上限；demo_budget按预设操作点数计量，不代表算力价格。"""
    policy: str = "plan_once"
    demo_budget_per_batch: int = 24
    recent_cap: int = 64
    history_cap: int = 64
    fit_sampling: str = "uniform"  # 可切换保留已观察类别，最终选择由真实任务比较。
    archive_cap: int = 3
    max_trials: int = 3
    max_plan_calls: int = 4
    min_improvement: float = 0.0
    seed: int = 17
    component_actions: bool = False
    active_inspection: bool = False
    method_memory: bool = False
    method_guidance: bool = True  # 可单独关闭文献提示，保持同一观察、工具和预算作比较。
    action_guidance: bool = True  # 可单独比较条件化改法排序建议，动作权限和采用规则保持。
    refinement_basis: str = "incumbent"  # best_trial从本批完整有效配方继续试改；仍按完整配置重建。
    local_corrections: bool = False
    candidate_pool_cap: int = 3
    candidate_min_support: int = 4
    action_routing: str = "priority"
    routing_step: float = .25
    routing_bias_limit: float = 2.
    routing_epsilon: float = .1
    routing_targets: dict | None = None
    problem_tracking: bool = False
    problem_depth: str = "adaptive"
    problem_capacity: int = 1
    problem_max_depth: int = 2
    problem_check_budget: int = 2
    problem_min_support: int = 4
    problem_reduction_target: float = .5
    max_checks: int = 2
    adaptive_loop: bool = False
    max_plan_repairs: int = 1
    max_idle_rounds: int = 2
    run_wall_seconds: float | None = None
    batch_wall_seconds: float | None = None
    output_reserve_seconds: float = 0.
    run_service_tokens: int | None = None
    batch_service_tokens: int | None = None
    service_call_reserve_tokens: int = 0
    output_fallback: bool = True
    worker_wall_seconds: float = 30.
    worker_cpu_seconds: float | None = None
    worker_memory_bytes: int | None = None
    worker_poll_seconds: float = .05
    execution_mode: str = "interface_fixture"
    fit_cv_folds: int = 0
    min_explore_for_cv: int = 1
    fit_cv_mode: str = "group"
    fit_cv_gap: float = 0.
    maintenance_trigger: str = "every_batch"
    maintenance_interval: int = 3
    maintenance_probe_interval: int = 5
    drift_min_samples: int = 30
    drift_threshold: float = 50.
    drift_delta: float = .005
    loss_scale: float | None = None
    inspection_order: str = "planner"
    inspection_exploration_every: int = 4
    resource_strategy: str = "equal"
    resource_fractions: tuple[float, ...] = (.25, 1.)
    resource_reduction: int = 2
    resource_audit_count: int = 1
    resource_tie_tolerance: float = 1e-12
    method_scope: str = "exact"
    method_distance_limit: float = 1.
    explore_current_weight: float | None = .5
    routing_reward_window: int = 20
    routing_ucb_exploration: float = 1.
    initial_pipeline: PipelineRecipe | None = None

    def __post_init__(self):
        from harness.runtime.runtime_budget import validate_limits
        # JSON恢复与Python调用使用同一配方对象；None保留旧工程样例的后端默认值。
        if isinstance(self.initial_pipeline, dict):
            object.__setattr__(self, "initial_pipeline", PipelineRecipe.parse(self.initial_pipeline))
        elif self.initial_pipeline is not None and not isinstance(self.initial_pipeline, PipelineRecipe):
            raise ValueError("初始配方须为组件配方或空值")
        if isinstance(self.resource_fractions, list):
            object.__setattr__(self, "resource_fractions", tuple(self.resource_fractions))
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("资料选择与训练的公共种子超出范围")
        if self.explore_current_weight is not None and (type(self.explore_current_weight) not in {int,float}
                or not math.isfinite(self.explore_current_weight) or not 0 <= self.explore_current_weight <= 1):
            raise ValueError("当前探索权重须为0至1；None保留按行汇总参照")
        if self.fit_sampling not in {"uniform", "class_coverage"}:
            raise ValueError("未知拟合抽样规则")
        from harness.methods.planning.maintenance import validate_policy
        validate_limits(self)
        from harness.runtime.execution import ExecutionLimits
        ExecutionLimits(wall_seconds=self.worker_wall_seconds,cpu_seconds=self.worker_cpu_seconds,
                        memory_bytes=self.worker_memory_bytes,poll_seconds=self.worker_poll_seconds)
        validate_policy(self)
        if self.fit_cv_mode not in {"group","time"} or not math.isfinite(self.fit_cv_gap) or self.fit_cv_gap<0:
            raise ValueError("交叉验证时间关系配置无效")
        if self.inspection_order not in {"planner", "decision_coverage"} or type(self.inspection_exploration_every) is not int or self.inspection_exploration_every < 1:
            raise ValueError("检查排序配置无效")
        if self.resource_strategy not in {"equal", "successive_halving", "reliable_halving"}:
            raise ValueError("未知资源分配规则")
        if self.refinement_basis not in {"incumbent", "best_trial", "complementary"}:
            raise ValueError("未知配方接续方式")
        if self.refinement_basis != "incumbent" and not self.component_actions:
            raise ValueError("配方接续需要启用组件动作")
        if type(self.local_corrections) is not bool or (self.local_corrections and not self.component_actions):
            raise ValueError("局部修正需要组件动作布尔开关")
        if type(self.candidate_pool_cap) is not int or not 1 <= self.candidate_pool_cap <= 5 or type(self.candidate_min_support) is not int or self.candidate_min_support < 2:
            raise ValueError("互补起点容量或支持条件无效")
        if self.action_routing not in {"priority", "balanced_opportunity", "epsilon", "reward_ucb"}:
            raise ValueError("未知尝试机会规则")
        if (type(self.routing_reward_window) is not int or not 2 <= self.routing_reward_window <= 100
                or type(self.routing_ucb_exploration) not in {int,float} or not math.isfinite(self.routing_ucb_exploration)
                or self.routing_ucb_exploration <= 0):
            raise ValueError("收益选择需要有限历史窗口和正探索系数")
        if (any(type(v) not in {int,float} or not math.isfinite(v) or v <= 0 for v in (self.routing_step,self.routing_bias_limit))
                or type(self.routing_epsilon) not in {int,float} or not math.isfinite(self.routing_epsilon)
                or not 0 <= self.routing_epsilon <= 1):
            raise ValueError("机会偏置参数无效")
        if self.routing_targets is not None and (not isinstance(self.routing_targets, dict) or any(
                not isinstance(k,str) or type(v) not in {int,float} or not math.isfinite(v) or v <= 0 for k,v in self.routing_targets.items())):
            raise ValueError("机会份额应为正的有限权重")
        if type(self.problem_tracking) is not bool or self.problem_depth not in {"fixed", "adaptive", "rotation"}:
            raise ValueError("问题跟踪或深查规则无效")
        if self.problem_tracking and not (self.component_actions and self.adaptive_loop and self.policy!="plan_once"):
            raise ValueError("问题跟踪需要反馈组件流程")
        if any(type(v) is not int or not 1 <= v <= 8 for v in (self.problem_capacity,self.problem_max_depth,self.problem_check_budget)) or type(self.problem_min_support) is not int or self.problem_min_support < 2:
            raise ValueError("问题容量、深度或支持条件无效")
        if type(self.problem_reduction_target) not in {int,float} or not math.isfinite(self.problem_reduction_target) or not 0 < self.problem_reduction_target <= 1:
            raise ValueError("问题改进目标应在0到1之间")
        levels = self.resource_fractions
        if not levels or len(levels) > 4 or any(type(v) not in {int,float} or not math.isfinite(v) or not 0 < v <= 1 for v in levels) or list(levels) != sorted(set(levels)) or levels[-1] != 1:
            raise ValueError("资源档须严格递增并以1结束")
        if type(self.resource_reduction) is not int or self.resource_reduction < 2:
            raise ValueError("筛选比例无效")
        if type(self.resource_audit_count) is not int or not 1 <= self.resource_audit_count <= 3 or type(self.resource_tie_tolerance) not in {int,float} or not math.isfinite(self.resource_tie_tolerance) or self.resource_tie_tolerance < 0:
            raise ValueError("筛选抽查数量或同分范围无效")
        if self.resource_strategy != "equal" and (not self.adaptive_loop or self.policy=="plan_once" or self.fit_cv_folds):
            raise ValueError("分档投入使用反馈循环，与fit交叉验证分别比较")
        if self.method_scope not in {"exact", "similar_recheck"} or not math.isfinite(self.method_distance_limit) or self.method_distance_limit < 0:
            raise ValueError("方法范围配置无效")
        if type(self.fit_cv_folds) is not int or self.fit_cv_folds not in {0,2,3,4,5} or type(self.min_explore_for_cv) is not int or self.min_explore_for_cv < 1:
            raise ValueError("小样本交叉验证配置无效")
        if type(self.output_fallback) is not bool or self.execution_mode not in {"interface_fixture", "real_model"}:
            raise ValueError("输出接续或执行来源配置无效")
        if self.policy not in {"plan_once", "plan_feedback", "plan_feedback_reuse"}:
            raise ValueError("未知运行模式")
        if any(type(v) is not bool for v in (self.component_actions, self.active_inspection, self.method_memory, self.method_guidance, self.action_guidance, self.adaptive_loop)):
            raise ValueError("能力开关须为布尔值")
        if type(self.max_checks) is not int or not 1 <= self.max_checks <= 4:
            raise ValueError("检查数量无效")
        if (self.active_inspection or self.method_memory) and not self.component_actions:
            raise ValueError("主动检查和方法复用需要启用组件动作")
        if self.adaptive_loop and not self.component_actions:
            raise ValueError("反馈工作选择需要启用组件动作")
        if type(self.max_plan_repairs) is not int or not 0 <= self.max_plan_repairs <= 3:
            raise ValueError("计划纠正次数无效")
        if type(self.max_idle_rounds) is not int or not 1 <= self.max_idle_rounds <= 4:
            raise ValueError("无新工作轮次上限无效")
        if self.demo_budget_per_batch < 4:
            raise ValueError("样例额度至少容纳初始化及必要输出")
        if min(self.recent_cap, self.history_cap, self.max_trials, self.max_plan_calls) < 1:
            raise ValueError("容量及尝试上限须为正数")
        if self.archive_cap < 0 or not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("归档容量或采用门槛无效")


class Planner(Protocol):
    identity: str

    def propose(self, context: dict) -> dict:
        """读取当时允许的事实并返回当前阶段的工作计划。

        原安排返回actions/stop；decision阶段返回decisions.py定义的单种工作。
        当前采用集及其答案由控制器保留，服务适配器负责网络超时和鉴权。
        """
        ...


class ModelTools(Protocol):
    identity: str
    evidence_kind: str

    def build(self, task: TaskSpec, batch_id: str, action: ActionPlan,
                rows: tuple[Sample, ...], *, initialize: bool = False, base: dict | None = None,
                parent: dict | None = None) -> dict:
        """仅用rows构造可JSON保存的产物，initialize区分首次初始化。

        抛出异常表示本次任务失败；控制器记录已用额度并保留其他版本。
        返回引用前应保存完整产物；本地TrainingModelTools按请求复用已发布文件。
        控制器中已开始但未确认的调用仍需核对结果，不能静默重新训练。
        base仅用于开放修正能力的工具，包含冻结基础版本及预测时可见的区域条件。
          parent仅传给声明续训轮数能力的工具，指向实际在用的父产物。
        """
        ...

    def predict(self, artifact: dict, inputs: tuple[dict, ...]) -> dict[str, int | float]:
        """输入只有身份与特征，禁止把评价标签交给预测器。"""
        ...
