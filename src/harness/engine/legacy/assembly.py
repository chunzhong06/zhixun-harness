"""统一装配工程替身或真实后端；真实模型和网络需要各自的显式许可。"""
from pathlib import Path

from harness.core.contracts import EngineConfig
from harness.engine.legacy.engine import Harness
from harness.adapters.executors.fixture_backend import FixtureTrainingBackend,SimulatedHttpService
from harness.learning.models.training_tools import TrainingModelTools,SklearnBackend,SklearnRecipe
from harness.engine.legacy.process_tools import ProcessTrainingTools
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.engine.legacy.execution import LocalJobRunner,ExecutionLimits
from harness.adapters.providers.pytrio_planner import PyTrioPlanner
from harness.adapters.providers.service_transport import HttpChatTransport
from harness.adapters.storage.local import LocalStore


def supervision_required(config, workers=False):
    """真实后端及有总时限/资源限额的运行统一使用可停止的工作进程。"""
    return (workers or config.execution_mode=="real_model" or config.run_wall_seconds is not None
            or config.batch_wall_seconds is not None or config.worker_cpu_seconds is not None
            or config.worker_memory_bytes is not None or config.worker_wall_seconds!=30.)


def assemble(task,config:EngineConfig,folder:Path,service_config,*,fixture=None,workers=False,
             allow_real_execution=False,allow_network=False,api_key=None,limits=None,
             planner_kind="service",backend=None):
    """本函数准备对象，不训练；实际run仍按配置许可及后端能力执行。

    密钥由调用者传入，不查找账户文件。真实训练需要执行许可，服务计划器
    另需网络许可和凭据；固定规则不调用服务。工程模式使用预写输出。
    """
    if planner_kind not in {"service","static","periodic","history","components","ordered_fixed","ordered_rotating","ordered_evidence"}:
        raise ValueError("未知计划器装配方式")
    if planner_kind.startswith("ordered_") and not config.component_actions:
        raise ValueError("改法顺序计划器需要开放组件动作")
    real=config.execution_mode=="real_model"
    if real and not allow_real_execution:
        raise ValueError("真实装配需要执行许可")
    if real and planner_kind=="service" and not (allow_network and api_key):
        raise ValueError("真实装配需要执行许可、服务网络许可及明确凭据")
    if not real and ((fixture is None and backend is None) or allow_network or api_key is not None):
        raise ValueError("工程模式需要预写样例，且不使用实际网络或凭据")
    folder=Path(folder)
    # 抽样与训练的随机性共用显式种子，避免只改了资料抽样而训练仍沿用默认种子。
    backend=backend or (SklearnBackend(SklearnRecipe(seed=config.seed),execution_enabled=True)
                       if real else FixtureTrainingBackend(fixture))
    if isinstance(backend,SklearnBackend) and backend.recipe.seed != config.seed:
        raise ValueError("注入后端的训练种子与公共配置不一致")
    if backend.evidence_kind != config.execution_mode:
        raise ValueError("训练后端与配置的执行来源不一致")
    if isinstance(backend,SklearnBackend) and any(value is None for value in backend.specification["packages"].values()):
        raise ValueError("真实训练依赖尚未齐备，请按任务环境安装并核对版本后执行")
    artifacts=ModelArtifacts(folder/"models")
    configured=ExecutionLimits(wall_seconds=config.worker_wall_seconds,cpu_seconds=config.worker_cpu_seconds,
                               memory_bytes=config.worker_memory_bytes,poll_seconds=config.worker_poll_seconds)
    if limits is not None and limits != configured:
        raise ValueError("执行器限制与公共配置不一致，请先合并为同一配置")
    if supervision_required(config,workers):
        runner=LocalJobRunner(folder/"jobs",artifacts,configured)
        tools=ProcessTrainingTools(task,backend,artifacts,runner)
    else:
        tools=TrainingModelTools(task,backend,artifacts)
    if planner_kind=="service":
        transport=(HttpChatTransport(service_config,api_key,allow_network=True) if real else
                   HttpChatTransport(service_config,"fixture-only-key",wire=SimulatedHttpService()))
        planner=PyTrioPlanner(service_config,transport,folder/"对话接口记录.json",allow_network=real)
    else:
        from harness.research.methods.planning.baselines import FixedMaintenancePlanner, FixedComponentPlanner, FixedHistoryPlanner
        from harness.research.methods.search.action_selection import ActionOrderPlanner
        planner=(ActionOrderPlanner(planner_kind.removeprefix("ordered_")) if planner_kind.startswith("ordered_")
                 else FixedComponentPlanner() if planner_kind=="components" else FixedHistoryPlanner() if planner_kind=="history"
                 else FixedMaintenancePlanner(planner_kind=="periodic"))
    return Harness(task,config,tools,planner,LocalStore(folder),allow_real_execution=allow_real_execution)
