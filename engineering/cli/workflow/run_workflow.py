"""一键逐批运行公共流程；默认仅使用预写预测与模拟服务。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.runtime.assembly import assemble,supervision_required
from harness.contracts import EngineConfig
from harness.runtime.handoff import LocalPackageEndpoint,PartitionedEndpoint,run_stream
from harness.data.package_input import read_package
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.runtime.storage import atomic_json
from harness.evaluation.longitudinal import evaluate_history
from engineering.cli.workflow.run_example import load_example


def main():
    if sys.argv[1:2] == ["native-record"]:
        from engineering.cli.workflow.native_recording import main as native_record
        return native_record(sys.argv[2:])
    if sys.argv[1:2] == ["native-research"]:
        from engineering.cli.workflow.native_research import main as native_research
        return native_research(sys.argv[2:])
    if sys.argv[1:2] == ["native-continue"]:
        from engineering.cli.workflow.native_continuation import main as native_continue
        return native_continue(sys.argv[2:])
    if sys.argv[1:2] == ["official-components"]:
        from engineering.cli.workflow.official_components import main as official_components
        return official_components(sys.argv[2:])
    if sys.argv[1:2] == ["london-code-research"]:
        from engineering.cli.workflow.mlestar_author import main as code_research
        return code_research(sys.argv[2:])
    if sys.argv[1:2] == ["london-complete"]:
        from engineering.cli.workflow.london_complete import main as london_complete
        return london_complete(sys.argv[2:])
    if sys.argv[1:2] == ["engine-field-maintenance"]:
        from engineering.cli.workflow.engine_field_maintenance import main as engine_field
        return engine_field(sys.argv[2:])
    if sys.argv[1:2] == ["published-program-predict"]:
        from engineering.cli.workflow.predict_published_program import main as predict_published
        return predict_published(sys.argv[2:])
    if sys.argv[1:2] == ["london-publication-refit"]:
        from engineering.cli.workflow.london_publication_refit import main as publication_refit
        return publication_refit(sys.argv[2:])
    if sys.argv[1:2] == ["london-midstream-replay"]:
        from engineering.cli.workflow.london_midstream_replay import main as replay_midstream
        return replay_midstream(sys.argv[2:])
    if sys.argv[1:2] == ["london-midstream"]:
        from engineering.cli.workflow.london_midstream import main as midstream
        return midstream(sys.argv[2:])
    if sys.argv[1:2] == ["london-aggregation"]:
        from engineering.cli.workflow.london_aggregation import main as aggregation
        return aggregation(sys.argv[2:])
    if sys.argv[1:2] == ["london-source-pool"]:
        from engineering.cli.workflow.london_source_pool import main as source_pool
        return source_pool(sys.argv[2:])
    if sys.argv[1:2] == ["london-code-driver"]:
        from engineering.cli.workflow.london_code_driver import main as code_driver
        return code_driver(sys.argv[2:])
    if sys.argv[1:2] == ["london-field-driver"]:
        from engineering.cli.workflow.london_field_driver import main as london_driver
        return london_driver(sys.argv[2:])
    if sys.argv[1:2] == ["station-field-driver"]:
        from engineering.cli.workflow.station_field_driver import main as station_driver
        return station_driver(sys.argv[2:])
    if sys.argv[1:2] == ["air-field-driver"]:
        from engineering.cli.workflow.air_field_driver import main as field_driver
        return field_driver(sys.argv[2:])
    if sys.argv[1:2] == ["field-classify"]:
        from engineering.cli.workflow.field_classification import main as field_classify
        return field_classify(sys.argv[2:])
    if sys.argv[1:2] == ["gas-predict"]:
        from engineering.cli.workflow.gas_observer_predict import main as gas_predict
        return gas_predict(sys.argv[2:])
    if sys.argv[1:2] == ["gas-public-feedback"]:
        from engineering.cli.workflow.gas_public_feedback import main as public_feedback
        return public_feedback(sys.argv[2:])
    if sys.argv[1:2] == ["life-checkpoint"]:
        from engineering.cli.workflow.serve_life_checkpoint import main as life_checkpoint
        return life_checkpoint(sys.argv[2:])
    if sys.argv[1:2] == ["gas-feedback"]:
        from engineering.cli.workflow.gas_feedback import main as gas_feedback
        return gas_feedback(sys.argv[2:])
    if sys.argv[1:2] == ["life-reference"]:
        from engineering.cli.workflow.engine_reference import life_main
        return life_main(sys.argv[2:])
    if sys.argv[1:2] == ["engine-reference"]:
        from engineering.cli.workflow.engine_reference import main as engine_reference
        return engine_reference(sys.argv[2:])
    if sys.argv[1:2] == ["engine-predict"]:
        from engineering.cli.workflow.engine_input_predict import main as engine_predict
        return engine_predict(sys.argv[2:])
    if sys.argv[1:2] == ["bearing-load-predict"]:
        from engineering.cli.workflow.bearing_load_predict import main as bearing_load_predict
        return bearing_load_predict(sys.argv[2:])
    if sys.argv[1:2] == ["bearing-phase-predict"]:
        from engineering.cli.workflow.bearing_phase_predict import main as bearing_phase_predict
        return bearing_phase_predict(sys.argv[2:])
    if sys.argv[1:2] == ["public-run"]:
        from engineering.cli.workflow.public_maintenance import main as public_run
        return public_run(sys.argv[2:])
    if sys.argv[1:2] == ["public-calibration"]:
        from engineering.cli.workflow.public_calibration import main as calibrate
        return calibrate(sys.argv[2:])
    if sys.argv[1:2] == ["official-gas-proposal"]:
        from engineering.cli.workflow.gas_first_proposal import main as first_proposal
        return first_proposal(sys.argv[2:])
    if sys.argv[1:2] == ["official-compare"]:
        from engineering.cli.workflow.official_compare import main as compare
        return compare(sys.argv[2:])
    if sys.argv[1:2] == ["official-stream"]:
        from engineering.cli.workflow.official_stream import main as official
        return official(sys.argv[2:])
    if sys.argv[1:2] in (["official-bearing"], ["official-engine"]):
        from harness.data.official_bearing import read_official_bearing
        from harness.data.official_engine import read_official_engine
        from harness.testing.rul_continuation import run_rehearsal
        command = sys.argv[1]
        reader, archive = ((read_official_bearing, "bearing/cpu-b-rul-20261110-v1.zip")
                           if command == "official-bearing" else (read_official_engine, "engine/cpu-c-guide-20261110-v1.zip"))
        parser = argparse.ArgumentParser(description="用官方设备观测与预写预测演练寿命维护")
        parser.add_argument("--package",type=Path,default=ROOT/"data/official"/archive)
        parser.add_argument("--output",type=Path,required=True)
        args = parser.parse_args(sys.argv[2:])
        run_rehearsal(reader(args.package),args.output)
        print(args.output/f"{command}-trace.json")
        return
    if sys.argv[1:2] == ["official-gas-pilot"]:
        from harness.evaluation.gas_pilot import run_arm
        parser = argparse.ArgumentParser(description="官方气体包本地CPU预测、续训与留出评价")
        parser.add_argument("--arm", required=True, choices=["static", "fixed", "adaptive", "focused", "focused-reuse",
                            "pair-prior", "pair-raw-knn", "pair-shape-knn", "pair-shape-blend",
                            "sequence-cascade", "sequence-protected", "sequence-first-window", "query-replay",
                            "query-first", "query-current"])
        parser.add_argument("--enable-real-execution", action="store_true", required=True)
        parser.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
        parser.add_argument("--protocol", type=Path, default=ROOT/"engineering/configs/experiments/official-gas-pilot.json")
        parser.add_argument("--order", choices=["reverse","rotate"], help="仅用于事先固定的整组顺序对照")
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        run_arm(args.package, args.protocol, args.output, args.arm, args.order)
        return
    if sys.argv[1:2] == ["official-gas"]:
        from harness.data.official_gas import read_official_gas
        from harness.testing.gas_continuation import run_rehearsal
        parser = argparse.ArgumentParser(description="用官方气体包和预写预测复查采用及续接，不执行模型")
        parser.add_argument("--package", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--policy", choices=["plan_once", "plan_feedback", "plan_feedback_reuse"], default="plan_feedback")
        args = parser.parse_args(sys.argv[2:])
        run_rehearsal(read_official_gas(args.package), args.output, args.policy)
        print(f"官方气体接口演练结果已写入 {args.output / 'official-gas-trace.json'}")
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=ROOT/"engineering/tests/fixtures/packages/package_layout/manifest.json")
    parser.add_argument("--example",type=Path,default=ROOT/"engineering/tests/fixtures/tasks/classification.json")
    parser.add_argument("--model-config",type=Path,default=ROOT/"engineering/tests/fixtures/services/pytrio_config.json")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--evaluation",type=Path,help="显式阶段与恢复定义JSON")
    parser.add_argument("--reference",type=Path,help="评价侧固定历史参照清单")
    parser.add_argument("--answers",type=Path,help="独立最终答案，运行提交后只用于评价")
    parser.add_argument("--split-config",type=Path,help="可选独立组/时间划分规则，含policy及metadata")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--policy",choices=["plan_once","plan_feedback","plan_feedback_reuse"],default="plan_feedback")
    parser.add_argument("--planner",choices=["service","static","periodic","history","components","ordered_fixed","ordered_rotating","ordered_evidence"],default="service")
    parser.add_argument("--workers",action="store_true")
    parser.add_argument("--enable-real-execution",action="store_true")
    parser.add_argument("--enable-service-network",action="store_true")
    args=parser.parse_args()
    if args.answers and args.split_config:
        raise ValueError("动态划分运行请导出实际批次后用独立评价入口绑定答案")
    local = args.planner != "service"
    if local and args.enable_service_network:
        raise ValueError("本地计划器不启用服务网络")
    settings=json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    settings.setdefault("policy",args.policy)
    for key in ("component_actions","active_inspection","method_memory","adaptive_loop"):
        settings.setdefault(key,True)
    config=EngineConfig(**settings)
    package=read_package(args.manifest,batch_index=0)
    reference=None
    if args.reference:
        from harness.evaluation.reference_evaluation import validate_reference
        reference=json.loads(args.reference.read_text(encoding="utf-8"))
        all_batches=read_package(args.manifest).batches
        validate_reference(package.task,reference,{"task":asdict(package.task),"models":{},
            "samples":{s.sample_id:asdict(s) for b in all_batches for s in b.samples}})
    real=config.execution_mode=="real_model"
    if not real and package.origin!="interface_fixture":
        raise ValueError("真实资料须通过明确的真实执行配置接入；工程入口只接受样例")
    fixture=None
    if not real:
        task,_,fixture=load_example(args.example)
        if task!=package.task:
            raise ValueError("样例与资料目标不一致")
    # 仅在两个实际执行开关已明确给出时读取指定变量，不读取其他账户环境。
    key=os.environ.get("PYTRIO_API_KEY") if not local and real and args.enable_real_execution and args.enable_service_network else None
    service=None if local else PyTrioConfig(**json.loads(args.model_config.read_text(encoding="utf-8")))
    engine=assemble(package.task,config,args.output,service,fixture=fixture,workers=args.workers,
                    allow_real_execution=args.enable_real_execution,allow_network=args.enable_service_network,api_key=key,
                    planner_kind=args.planner)
    endpoint=LocalPackageEndpoint(args.manifest,args.output/"accepted")
    if args.split_config:
        from harness.data.splitting import SplitPolicy
        specification=json.loads(args.split_config.read_text(encoding="utf-8"))
        endpoint=PartitionedEndpoint(endpoint,args.output/"partitions",SplitPolicy(**specification["policy"]),
                                     metadata=specification.get("metadata"))
    state,handoff=run_stream(engine,endpoint,args.output/"handoff")
    if args.answers:
        from harness.evaluation.final_answers import prepare_answers,score_final
        # 运行已完成后才读取独立答案；与计划、训练和恢复状态没有回写路径。
        full_package=read_package(args.manifest)
        prepared=prepare_answers(package.task,full_package.batches,json.loads(args.answers.read_text(encoding="utf-8")))
        final=score_final(package.task,state,prepared)
        atomic_json(args.output/"final-evaluation.json",final)
        definition=json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else {}
        atomic_json(args.output/"final-longitudinal.json",evaluate_history(state,**definition,final_report=final))
    reference_report=None
    if reference is not None:
        from harness.evaluation.reference_evaluation import evaluate_references,reference_tools
        destination=args.output/'reference-evaluation'
        reference_report=evaluate_references(package.task,state,reference,reference_tools(engine,destination),destination,
                                             allow_real_execution=args.enable_real_execution)
    if args.evaluation or reference is not None:
        evaluation=json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else {}
        atomic_json(args.output/"longitudinal.json",evaluate_history(state,**evaluation,reference_report=reference_report))
        atomic_json(args.output/"evaluation-definition.json",evaluation)
    atomic_json(args.output/"run-receipt.json",{"completed_batches":len(state["completed"]),
        "accepted_batches":len(handoff["accepted"]),"evidence_kind":state["evidence_kind"],
        "model_experiment":state["model_experiment"],"input_manifest_digest":endpoint.identity,
        "execution":config.execution_mode,"planner":args.planner,"workers":supervision_required(config,args.workers)})
    print(f"逐批流程完成：{len(state['completed'])}批均已接收；执行来源={state['evidence_kind']}。")


if __name__=="__main__":
    main()
