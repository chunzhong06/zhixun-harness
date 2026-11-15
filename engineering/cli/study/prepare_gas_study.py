"""冻结气体任务的共同起点及比较清单；不装配训练器，不调用模型服务。"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.contracts import EngineConfig, digest
from harness.data.package_input import read_package
from harness.data.development import export_development
from harness.evaluation.comparison import prepare_comparison
from harness.runtime.storage import atomic_json


def command_path(path):
    """命令从项目根目录执行；项目内输入可随整个工作区搬移，外部挂载保留其位置。"""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def prepare_study(manifest, specification, output):
    """只读现有开发包，输出首批筛选入口、完整配方和待执行比较。

    首批筛选移除adopt/serve及其答案，只保留fit/explore；选择初始配方
    时不使用后续批次。维护比较继续使用原包，运行目录与预备目录分开。
    输出须为空；部分写入失败的目录由调用者核查后清理，不覆盖旧证据。
    """
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    package = read_package(manifest)
    answers_path = manifest.parent.parent / "evaluation/answers.json"
    answer_spec = json.loads(answers_path.read_text(encoding="utf-8"))
    if package.task.task_id != "uci-gas-drift" or len(package.batches) != 10:
        raise ValueError("本入口需要已登记的十批气体本地开发协议")
    if specification.get("schema_version") != 1 or not specification.get("profiles"):
        raise ValueError("缺少起点候选配置")
    first = package.batches[0]
    # 终局批只接受无答案服务输入；第九批可同时携带第八批合法返回的explore。
    for ordinal, batch in enumerate(package.batches, 1):
        if batch.ordinal != ordinal or batch.batch_id != f"gas-source-{ordinal}":
            raise ValueError("批次身份不符合已保存的气体协议")
        if ordinal >= 9 and any(s.role in {"fit", "adopt"} for s in batch.samples):
            raise ValueError("终局批不得包含用于训练或采用的资料")
    configs = {}
    for name, pipe in specification["profiles"].items():
        if not name.isascii() or not name.replace("_", "").isalnum():
            raise ValueError("配方名称只能使用ASCII字母、数字和下划线")
        configs[name] = EngineConfig(**{**specification["base_config"], "initial_pipeline": pipe})
        configs[name].initial_pipeline.validate_task(package.task)
    if output.exists() and any(output.iterdir()):
        raise ValueError("准备目录非空，请保留原记录并选择新的目录")
    output.mkdir(parents=True, exist_ok=True)
    screen_batch = replace(first, samples=tuple(s for s in first.samples if s.role in {"fit", "explore"}))
    export_development(package.task, (screen_batch,), {"protocol": "gas-initial-screen-v1",
        "official": False, "roles": "仅首批fit/explore；无采用、服务及未来答案", "answers": []}, output / "initial-screen")
    screen_package=read_package(output/"initial-screen/runtime/manifest.json")
    definition={"schema_version":1,"baseline_id":"static","initial_batch":first.batch_id,
        "maintenance_batches":[b.batch_id for b in package.batches[1:8]],
        "holdout_batches":[b.batch_id for b in package.batches[8:]],
        "recovery":{"drop":.10,"tolerance":.05,"consecutive":2}}
    atomic_json(output/"study-definition.json",definition)
    entries, common = [], None
    for name, cfg in configs.items():
        config_path = output / f"{name}.json"
        atomic_json(config_path, asdict(cfg))
        comparison = prepare_comparison(package.task, package.batches, cfg,
            factory_identity="shared-assembly-v3:local-sklearn", local_only=True, suite="maintenance",
            answer_spec=answer_spec,study_definition=definition)
        initial = comparison["initialization"]
        shared = (initial["training_digest"], initial["seed"])
        if common is not None and common != shared:
            raise ValueError("起点模型未使用同一拟合资料与种子")
        if not initial["condition"]["ready"]:
            raise ValueError("首批资料不满足候选训练条件")
        common = shared
        atomic_json(output / name / "comparison-plan.json", {"status": "prepared_only", "executed": False,
            "model_training": False, "plan": comparison})
        # 起点筛选只拟合一次并评分，使用静态计划兼容的配置，不进入反馈/检查循环。
        screen_config = replace(cfg, policy="plan_once", active_inspection=False, method_memory=False,
            adaptive_loop=False, problem_tracking=False, resource_strategy="equal", fit_cv_folds=0)
        screen_config_path = output / "initial-screen" / f"{name}.json"
        atomic_json(screen_config_path, asdict(screen_config))
        # 这里只记录命令参数；用户授权前不会执行其中的真实运行开关。
        entries.append({"profile": name, "pipeline": asdict(cfg.initial_pipeline),
            "config_file":f"{name}.json","config_digest":digest(asdict(cfg)),
            "screen_config_file":f"initial-screen/{name}.json","screen_config_digest":digest(asdict(screen_config)),
            "run_directory":f"executed/initial/{name}",
            "initialization": initial,
            "screen_argv": ["python", "-B", "engineering/cli/workflow/run_workflow.py", "--planner", "static",
                "--manifest", command_path(output / "initial-screen/runtime/manifest.json"),
                "--config", command_path(screen_config_path), "--output", command_path(output / "executed/initial" / name),
                "--enable-real-execution"],
            "comparison_prepare_argv": ["python", "-B", "engineering/cli/study/compare_workflow.py", "--local-only",
                "--suite", "maintenance", "--study-definition", command_path(output/"study-definition.json"),
                "--prepare-only", "--manifest", command_path(manifest),
                "--config", command_path(config_path), "--answers", command_path(answers_path),
                "--output", command_path(output / "selected-plan" / name)]})
    report = {"schema_version": 2, "executed": False, "model_training": False, "model_called": False,
        "command_working_directory": "project_root",
        "manifest": command_path(manifest), "input_digest": digest([asdict(b) for b in package.batches]),
        "final_answer_digest": digest(answer_spec),
        "screen_input_digest":digest([asdict(b) for b in screen_package.batches]),
        "study_definition":definition,
        "specification": specification, "initial_profiles": entries,
        "selection": {"selected": None, "metric": package.task.metric, "scope": "首批explore",
            "tie_rule": "按配置文件中候选顺序选择；该顺序在观察结果前固定",
            "freeze_before": "启动第二批之前冻结起点配方，所有维护比较使用同一份配置",
            "evidence": "首批静态运行state中初始化模型的explore评分及完整费用；选中后记录该运行摘要",
            "final_scope": "源批9和10独立评价，不用于选择配方、规则或阈值"}}
    atomic_json(output / "study.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/prepared/gas/gas-development/runtime/manifest.json")
    parser.add_argument("--spec", type=Path, default=ROOT / "engineering/configs/experiments/gas-study.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_study(args.manifest, json.loads(args.spec.read_text(encoding="utf-8")), args.output)
    print("起点筛选入口与共同维护配置已准备；训练和服务未执行。")


if __name__ == "__main__":
    main()
