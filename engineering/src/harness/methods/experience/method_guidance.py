"""把已审阅的方法条件映射到本轮可执行选择；原始观察仍由planning_facts拥有。"""
from copy import deepcopy

from harness.contracts import digest


GUIDANCE_VERSION = 3
# 文献只支持其原方法。transfer说明本项目采用的部分，逐次运行的事实另行引用。
METHODS = {
    "tabular_boosting": {
        "title": "比较预提取特征上的提升树改法",
        "source": {"title": "TabReD", "url": "https://github.com/yandex-research/tabred/blob/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/paper/bin/catboost_.py",
                   "location": "论文§5.1—5.2与作者CatBoost训练入口", "note": "N084"},
        "transfer": "采用成熟提升树作为线性/森林之外的可执行候选；使用固定迭代，未移植作者验证集早停或论文排名。",
        "when": "数值特征与当前探索表现已提供，catboost动作已开放，实际训练环境需具备其依赖。",
        "suggestion": "当前配方留有错误时，可保持资料、特征与训练权重相同，只换catboost，比较非线性特征组合是否有用。读取现配方，显式pipeline会替换全部字段；随后再比较深度或迭代量。",
        "tradeoff": "加入一个训练候选及可选依赖；CPU单线程、树深上限和总预算共同控制费用。该论文支持候选价值，具体任务收益由共同评价给出。",
    },
    "class_balance": {
        "title": "让少数类别在训练中得到足够权重",
        "source": {"title": "scikit-learn类别权重", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
                   "location": "class_weight balanced公式；相关覆盖问题见N055/N083", "note": "N084"},
        "transfer": "对实际fit清单使用n/(类别数×该类条数)的样本权重，各预测器共用公式；没有复制额外采样算法。",
        "when": "分类拟合池的类别数量不均，探索结果存在类别漏判，目录开放balanced。",
        "suggestion": "先核对input_change与data_selection中的类别覆盖，再看error_profile的类别样本数、召回和漏判。保持相同资料及预测器，比较class_weight=none与balanced。完全缺失的类别需先补入合法历史资料，权重不产生新样本。",
        "tradeoff": "不增加训练行数；可能提高少数类召回并降低其他类准确率，也会放大罕见误标影响。按原任务主指标和逐类得失决定采用，不改变评分口径。",
    },
    "regression_objective": {
        "title": "比较连续目标的误差取舍",
        "source": {"title": "CatBoost训练目标", "url": "https://catboost.ai/docs/en/references/training-parameters/common#loss_function",
                   "location": "loss_function：RMSE与MAE", "note": "N084"},
        "transfer": "在相同提升树、资料与预算下切换回归损失；保留官方目标和原单位评分。",
        "when": "连续目标已有探索误差，catboost与mae损失已开放。",
        "suggestion": "error_profile描述预测偏高/偏低及最大误差占比。default在catboost中为RMSE，强调较大误差；mae侧重绝对误差。保持其他配置相同比较两者，按objective.metric和各组表现选择；损失切换不保证消除偏差。",
        "tradeoff": "大误差可能来自关键退化阶段，不能直接当作异常丢弃。寿命任务的偏高风险和阶段分组须来自正式业务定义；不自行截断寿命标签或改成非对称评分。",
    },
    "component_comparison": {
        "title": "根据组件试改结果安排下一步",
        "source": {"title": "MLE-STAR", "url": "https://arxiv.org/abs/2506.15692v3",
                   "location": "§3.2、附录A.4—A.8", "note": "N048"},
        "transfer": "借鉴局部试改和结果反馈；这里使用已登记组件，未移植开放代码搜索。",
        "when": "已有当前探索评价，且开放完整组件配方。",
        "suggestion": "从明确的问题选择处理器、特征或预测器，先保持其余配方和训练资料相同作比较；根据已完成结果决定继续改该处、比较组合或结束。组合改动要说明各处为什么需要配合。",
        "tradeoff": "每次比较都计入训练和评价费用；有限预算优先安排能改变后续选择的试改。",
    },
    "scaling": {
        "title": "让缩放方式对应实际输入分布",
        "source": {"title": "scikit-learn预处理指南", "url": "https://scikit-learn.org/stable/modules/preprocessing.html",
                   "location": "标准化、含极端值的缩放与Pipeline", "note": "N081"},
        "transfer": "采用维护者对标准化和稳健缩放的适用条件；检查阈值是本项目描述性规则。",
        "when": "合法拟合资料至少两条，且开放多个缩放处理器。",
        "suggestion": "feature_scale检查各特征量纲跨度；tail_profile检查单个特征全距相对中部跨度。前者支持比较标准化，后者支持比较robust与standard。跨度本身不区分少量极端值和正常宽分布；读取检查的样本数、恒定与零中部跨度标记后再提出假设。线性预测器优先关注尺度影响。",
        "tradeoff": "稳健缩放保留极端观测；处理器只在拟合资料上学习，并和下游预测器一起重建、评价。",
    },
    "error_localization": {
        "title": "先看错误集中在哪里",
        "source": {"title": "TensorFlow Model Analysis", "url": "https://www.tensorflow.org/tfx/model_analysis/model_validations",
                   "location": "分组评价与基线比较", "note": "N009"},
        "transfer": "借鉴分组表现分析；按输入中位数检查错误的工具由本项目实现。",
        "when": "当前探索评价可用且有足够资料形成两侧，error_slices工具已开放。",
        "suggestion": "先读最差分组及各组样本数；需要进一步区分输入范围时用error_slices。让两种检查结果对应不同改法，沿任务指标及方向比较候选，并留意原有组的表现。",
        "tradeoff": "检查复用已取得的预测；小组结果受样本构成影响，输入区间是统计分组，物理原因需要另有依据。",
    },
    "joint_error_regions": {
        "title": "找出联合条件下的错误并用新资料复查",
        "source": {"title": "SliceLine / Incremental SliceLine", "url": "https://mboehm7.github.io/resources/btw2025b.pdf",
                   "location": "联合谓词、支持数与模型变化后的误差更新；N085", "note": "N085"},
        "transfer": "借鉴联合条件搜索；本项目采用有界二分搜索、冻结谓词和新单位复查，没有移植完整增量搜索算法。",
        "when": "有合法fit/explore及当前探索预测；独立复查还需要单位身份和后续新资料。",
        "suggestion": "error_regions提供单条件与双条件区域，并复查此前冻结区域。读取覆盖、单位和复查状态再决定改动；模型已更换的区域表现与同模型问题复现分别解释。",
        "tradeoff": "搜索限制特征数与深度；探索所得区域需要后续复查，筛选造成的乐观偏差不作为实测收益。",
    },
    "history_mix": {
        "title": "比较近期适应与保留历史覆盖",
        "source": {"title": "Avalanche存储策略", "url": "https://github.com/ContinualAI/avalanche/blob/v0.6.0/avalanche/training/storage_policy.py",
                   "location": "ReservoirSamplingBuffer、BalancedExemplarsBuffer", "note": "N081"},
        "transfer": "借鉴随机和分组记忆；本项目从可见历史中确定性抽样，不实现其在线蓄水池更新。",
        "when": "近期和历史拟合池均有资料，当前允许历史回放配方。",
        "suggestion": "保持组件相同，比较recent与uniform；报告分组确有覆盖意义时再比较coverage。ratio是历史条数/近期条数。结合当前类别、分组覆盖和已完成尝试选择，不因存在历史就默认加入更多。",
        "tradeoff": "增加历史可能保留旧条件，也可能稀释近期变化；历史量受用量和总投入约束。group含义由任务声明，不能自行当作设备或故障类别。",
    },
    "method_recheck": {
        "title": "让旧方法先接受当前条件检验",
        "source": {"title": "SELeCT", "url": "https://arxiv.org/abs/2408.09324",
                   "location": "§III—IV；N029、N056固定实现核查", "note": "N029"},
        "transfer": "原工作复查旧模型状态；这里把召回后核查的分工用于已保存组件配方。",
        "when": "当前方法目录返回了可复核候选。",
        "suggestion": "先看候选范围、历史共同对照与反向结果。eligible方法可沿原method_id使用；其余候选在decision阶段通过recheck与同资料普通重训配对比较，再决定是否继续采用。",
        "tradeoff": "复核也消耗额度；相近范围提供候选线索，实际采用仍由本轮共同评价决定。",
    },
}


def guidance_identity():
    """来源或条件变化进入运行身份；修改筛选语义时同步提升版本。"""
    return digest([GUIDANCE_VERSION, METHODS])


def build_method_guidance(context):
    """仅从已开放摘要和能力目录选提示，无文件检索、额外计算或模型调用。

    卡片列出当前关联事实和可用操作，保留来源与本项目改写范围。提示不替
    计划器选择赢家，也不产生新的fact；事实编号仍只来自当前观察目录。
    """
    facts = context["facts"]
    items = {item["kind"]: item for item in facts.get("evidence", [])}
    maintenance = context.get("maintenance", {})
    components = maintenance.get("components", {})
    checks = maintenance.get("available_checks", {})
    recipes = context["allowed_recipes"]
    cards = []

    def add(key, kinds, actions):
        cards.append({"id": key, **deepcopy(METHODS[key]), "available_actions": actions,
                      "observation_refs": [items[k]["id"] for k in kinds if k in items]})

    performance = items.get("performance", {}).get("value", {})
    profile = performance.get("error_profile") or {}
    if "catboost" in components.get("predictor", []) and performance.get("status") == "available":
        add("tabular_boosting", ("performance", "input_change", "attempts", "resources"),
            {"predictor": ["catboost"], "parameters": deepcopy(components.get("parameters", {}))})
    inputs = items.get("input_change", {}).get("value", {})
    recent_counts = [c["count"] for c in inputs.get("recent", {}).get("targets", {}).get("counts", [])]
    if (profile.get("task_type") == "classification" and len(set(recent_counts)) > 1
            and any(c["missed"] > 0 for c in profile.get("classes", []))
            and "balanced" in components.get("class_weight", [])):
        add("class_balance", ("performance", "input_change", "attempts"),
            {"class_weight": deepcopy(components["class_weight"])})
    if (profile.get("task_type") == "regression" and "catboost" in components.get("predictor", [])
            and "mae" in components.get("regression_loss", [])):
        add("regression_objective", ("performance", "attempts", "resources"),
            {"predictor": ["catboost"], "regression_loss": deepcopy(components["regression_loss"])})
    if components and performance.get("status") == "available":
        add("component_comparison", ("performance", "attempts", "resources"),
            {key: deepcopy(components[key]) for key in ("processor", "features", "predictor")})
    if facts["recent_count"] >= 2 and len(components.get("processor", [])) > 1:
        add("scaling", ("input_change", "investigations", "attempts"),
            {"processor": deepcopy(components["processor"]),
             "checks": [name for name in ("feature_scale", "tail_profile") if name in checks]})
    if performance.get("status") == "available" and performance.get("count", 0) >= 4 and "error_slices" in checks:
        add("error_localization", ("performance", "investigations"), {"checks": ["error_slices"]})
        if "error_regions" in checks:
            add("joint_error_regions", ("performance", "investigations"), {"checks": ["error_regions"]})
    replay = {key: deepcopy(value) for key, value in recipes.items() if key in {"uniform", "coverage"} and value}
    if facts["recent_count"] and facts["history_count"] and replay:
        add("history_mix", ("input_change", "performance_change", "attempts", "resources"),
            {"recipes": {**({"recent": deepcopy(recipes["recent"])} if recipes.get("recent") else {}), **replay}})
    methods = maintenance.get("methods", [])
    if methods:
        add("method_recheck", ("methods", "attempts", "resources"),
            {"method_ids": [method["id"] for method in methods], "recheck_stage": "decision"})
    return {"version": GUIDANCE_VERSION, "catalog_digest": guidance_identity(),
            "basis": "已审阅文献与成熟实现的条件化建议；观察与项目效果由facts及后续评价提供",
            "cards": cards}


def validate_method_guidance(context):
    """带提示的请求必须对应本地目录和当前事实；独立旧格式样例仍可省略。"""
    if "method_guidance" in context and context["method_guidance"] != build_method_guidance(context):
        raise ValueError("方法提示与当前事实、开放能力或来源版本不一致")
