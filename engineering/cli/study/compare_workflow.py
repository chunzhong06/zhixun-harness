"""共用数据、装配、预算和评价运行对照；可仅导出待执行的配置清单。"""
import argparse
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from engineering.cli.workflow.run_example import load_example
from harness.runtime.assembly import assemble
from harness.evaluation.comparison import prepare_comparison, run_comparison
from harness.contracts import EngineConfig,digest
from harness.data.package_input import read_package
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.runtime.storage import atomic_json


def main():
    root=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example",type=Path,default=root/"engineering/tests/fixtures/tasks/classification.json")
    parser.add_argument("--manifest",type=Path,help="按公共映射读取同一批次序列")
    parser.add_argument("--config",type=Path,help="各组共同EngineConfig")
    parser.add_argument("--model-config",type=Path)
    parser.add_argument("--evaluation",type=Path,help="统一阶段与恢复定义")
    parser.add_argument("--reference",type=Path,help="评价侧固定历史参照清单")
    parser.add_argument("--answers",type=Path,help="评价侧最终答案；不传给运行装配或计划器")
    parser.add_argument("--suite",choices=["standard","maintenance","mechanisms"],default="standard")
    parser.add_argument("--study-definition",type=Path,help="评价侧阶段与共同恢复定义")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--policy",choices=["plan_once","plan_feedback","plan_feedback_reuse"])
    parser.add_argument("--budget",type=int)
    parser.add_argument("--worker-timeout",type=float)
    parser.add_argument("--extended",action="store_true")
    parser.add_argument("--local-only",action="store_true",help="只比较本地规则与静态/重训参照，不读取凭据或创建模型服务")
    parser.add_argument("--compare-ordering",action="store_true",help="增加同目录固定、轮换和观察驱动的顺序对照")
    parser.add_argument("--compare-refinement",action="store_true",help="增加同预算的服务配方与本批有效配方接续对照")
    parser.add_argument("--compare-stage2",action="store_true",help="增加局部修正、互补起点、可靠分档、机会选择和问题深查的同预算开关对照")
    parser.add_argument("--loss-scale",type=float)
    parser.add_argument("--prepare-only",action="store_true",help="只保存输入与比较配置，不装配或执行模型")
    parser.add_argument("--enable-real-execution",action="store_true")
    parser.add_argument("--enable-service-network",action="store_true")
    args=parser.parse_args()
    if args.suite != "standard":
        args.local_only=True
    if args.local_only and args.enable_service_network:
        raise ValueError("本地比较不启用服务网络")
    settings=json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for flag,key in [(args.policy,"policy"),(args.budget,"demo_budget_per_batch"),
                     (args.worker_timeout,"worker_wall_seconds"),(args.loss_scale,"loss_scale")]:
        if flag is not None:settings[key]=flag
    settings.setdefault("policy","plan_feedback_reuse")
    for key in ("component_actions","active_inspection","method_memory","adaptive_loop"):
        settings.setdefault(key,True)
    config=EngineConfig(**settings)
    task,batches,fixture=load_example(args.example)
    if args.manifest:
        package=read_package(args.manifest)
        if config.execution_mode=="interface_fixture" and (package.origin!="interface_fixture" or package.task!=task):
            raise ValueError("工程比较需要相同任务的工程资料")
        task,batches=package.task,package.batches
    real=config.execution_mode=="real_model"
    if real and not args.manifest:
        raise ValueError("真实运行预备配置需要显式资料映射清单")
    service=(None if args.local_only else PyTrioConfig(**json.loads(args.model_config.read_text(encoding="utf-8")))
             if args.model_config else PyTrioConfig(model="fixture-qwen"))
    evaluation=json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else {}
    reference=json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    answers=json.loads(args.answers.read_text(encoding="utf-8")) if args.answers else None
    study_definition=json.loads(args.study_definition.read_text(encoding="utf-8")) if args.study_definition else None
    identity="shared-assembly-v3:"+digest([fixture if not real else "sklearn-pipeline",asdict(service) if service else None])
    if args.prepare_only:
        plan=prepare_comparison(task,batches,config,factory_identity=identity,extended=args.extended,ordering=args.compare_ordering,refinement=args.compare_refinement,stage2=args.compare_stage2,local_only=args.local_only,evaluation=evaluation,reference_spec=reference,answer_spec=answers,suite=args.suite,study_definition=study_definition)
        atomic_json(args.output/"comparison-plan.json",{"status":"prepared_only","executed":False,"model_training":False,"plan":plan})
        print("共同对照配置已准备；未装配、调用或训练模型。")
        return
    if real and not args.enable_real_execution:
        raise ValueError("真实对照需要显式执行许可")
    if real and not args.local_only and not (args.enable_service_network and args.model_config):
        raise ValueError("真实对照需要显式执行/网络许可及服务配置")
    # 预备模式提前返回，只有明确运行真实对照时才读取这一个凭据变量。
    key=os.environ.get("PYTRIO_API_KEY") if real and not args.local_only else None
    def factory(case,cfg,target,kind):
        return assemble(task,cfg,target,service,fixture=fixture if not real else None,workers=True,
            allow_real_execution=args.enable_real_execution,allow_network=real and kind=="service" and args.enable_service_network,
            api_key=key if kind=="service" else None,planner_kind=kind)
    report=run_comparison(task,batches,args.output,config=config,engine_factory=factory,factory_identity=identity,
                          extended=args.extended,ordering=args.compare_ordering,refinement=args.compare_refinement,stage2=args.compare_stage2,local_only=args.local_only,evaluation=evaluation,allow_real_execution=args.enable_real_execution,reference_spec=reference,answer_spec=answers,suite=args.suite,study_definition=study_definition)
    complete=sum(row["status"]=="completed" for row in report["cases"])
    print(f"共同对照完成：{complete}/{len(report['cases'])}组；执行来源={config.execution_mode}。")
    if complete!=len(report["cases"]):raise SystemExit(1)


if __name__=="__main__":main()
