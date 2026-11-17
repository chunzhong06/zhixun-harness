"""固定共同输入、预算和评价，使用可替换装配比较完整运行及能力开关。"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path

from harness.core.contracts import EngineConfig, digest
from harness.evaluation.costs import operation_costs
from harness.adapters.providers.pytrio_planner import PyTrioConfig
from harness.adapters.storage.local import LocalStore, atomic_json
from harness.core.contracts import ActionPlan


def comparison_cases(base: EngineConfig, *, extended=False, ordering=False, refinement=False, stage2=False, local_only=False, suite="standard"):
    """完整运行与两个单因素关闭项共用其他参数，固定规则也使用共同采用器。"""
    if not (base.component_actions and base.active_inspection and base.method_memory):
        raise ValueError("完整比较配置需要开启组件、主动检查与方法经验")
    fixed = replace(base, policy="plan_once", active_inspection=False, method_memory=False, adaptive_loop=False,
                    maintenance_trigger="every_batch",resource_strategy="equal",problem_tracking=False)
    if suite not in {"standard", "maintenance", "mechanisms"}:
        raise ValueError("未知比较集合")
    if suite != "standard":
        if extended or ordering or refinement or stage2:
            raise ValueError("精简比较集合不叠加旧的批量扩展开关")
        fixed = replace(fixed, fit_cv_folds=0)
        shared = replace(base, policy="plan_feedback", adaptive_loop=True, active_inspection=True,
            method_memory=True, component_actions=True, local_corrections=True, refinement_basis="complementary",
            problem_tracking=True, fit_cv_folds=0, resource_strategy="equal", maintenance_trigger="every_batch")
        mature = replace(shared, problem_depth="fixed", action_routing="reward_ucb")
        problem = replace(shared, problem_depth="adaptive", action_routing="balanced_opportunity")
        cases = [
            ("static", "保持初始模型，仅服务", replace(fixed, maintenance_trigger="never"), False),
            ("periodic", "每批近期重训", fixed, True),
            ("history", "近期与历史的固定维护", fixed, "history"),
            ("mature", "同工具成熟收益选择", mature, "ordered_evidence"),
            ("problem", "问题驱动与机会补偿", problem, "ordered_evidence"),
        ]
        if suite == "mechanisms":
            # 与mature/problem共同形成两因素比较；其余配置完全共用shared。
            cases += [("adaptive_reward", "按问题检查与成熟收益选择",
                       replace(mature, problem_depth="adaptive"), "ordered_evidence"),
                      ("fixed_opportunity", "固定检查与机会补偿",
                       replace(mature, action_routing="balanced_opportunity"), "ordered_evidence")]
        return cases
    cases = [
        ("static", "保持初始模型", fixed, False),
        ("periodic", "每批近期重训", fixed, True),
        ("full", "完整系统", base, None),
        ("without_inspection", "关闭主动检查", replace(base, active_inspection=False), None),
        ("without_memory", "关闭方法经验", replace(base, method_memory=False), None),
    ]
    if local_only:
        cases = cases[:2] + [("local_rules", "本地观察驱动规则", base, "ordered_evidence")]
    if extended:
        grid = replace(base,policy="plan_feedback",adaptive_loop=True,active_inspection=False,method_memory=False,
                       maintenance_trigger="every_batch",resource_strategy="equal",fit_cv_folds=0)
        cases += [
            ("fixed_components","固定组件等额搜索",grid,"components"),
            ("triggered_components","误差变化触发组件搜索",replace(grid,maintenance_trigger="page_hinkley"),"components"),
            ("halving_components","固定组件分档搜索",replace(grid,resource_strategy="successive_halving"),"components"),
        ]
    if ordering:
        shared = replace(base, policy="plan_feedback", adaptive_loop=True, active_inspection=False,
                         method_memory=False, maintenance_trigger="every_batch", resource_strategy="equal", fit_cv_folds=0)
        cases += [(f"order_{mode}", label, shared, f"ordered_{mode}") for mode, label in (
            ("fixed", "同目录固定顺序"), ("rotating", "同目录跨批轮换"), ("evidence", "同目录观察驱动与探索"))]
    if refinement:
        # 动作类型、排序规则、入口与预算一致；只比较后续组合的配方起点。
        shared = replace(base, policy="plan_feedback", adaptive_loop=True, active_inspection=False,
                         method_memory=False, maintenance_trigger="every_batch", resource_strategy="equal", fit_cv_folds=0)
        cases += [(f"refine_{basis}", label, replace(shared, refinement_basis=basis), "ordered_evidence")
                  for basis, label in (("incumbent", "从在用模型配方试改"), ("best_trial", "从本批有效配方接续"))]
    if stage2:
        # 用确定规则隔离各项机制，保留调用与训练预算；不因增加能力而放宽资源。
        shared = replace(base, policy="plan_feedback", adaptive_loop=True, active_inspection=True,
                         local_corrections=True, refinement_basis="complementary",
                         resource_strategy="reliable_halving", action_routing="balanced_opportunity",
                         problem_tracking=True, problem_depth="adaptive",fit_cv_folds=0)
        variants = [
            ("stage2", "组合能力与本地规则", shared),
            ("without_correction", "相同流程关闭局部修正", replace(shared, local_corrections=False)),
            ("single_best", "只从总体较好配方继续", replace(shared, refinement_basis="best_trial")),
            ("fixed_halving", "采用固定分档筛选", replace(shared, resource_strategy="successive_halving")),
            ("priority_only", "采用原优先级排序", replace(shared, action_routing="priority")),
            ("reward_ucb", "同目录的近期收益UCB参照", replace(shared, action_routing="reward_ucb")),
            ("direct_full", "小目录直接完整执行参照", replace(shared, resource_strategy="equal")),
            ("without_problems", "关闭问题跟踪与深查", replace(shared, problem_tracking=False)),
            ("fixed_depth", "固定问题深入顺序", replace(shared, problem_depth="fixed")),
        ]
        cases += [("mechanism_"+key, label, cfg, "ordered_evidence") for key,label,cfg in variants]
    return cases


def expected_feedback(batches):
    """只使用先预测、后来合法发回的答案；同一样本取最后一次已释放修订。"""
    served, answers = set(), {}
    for batch in batches:
        for sample in batch.samples:
            if sample.sample_id in served and sample.label is not None and sample.available_at <= batch.ordinal:
                previous = answers.get(sample.sample_id)
                if previous is None or sample.revision > previous["revision"]:
                    answers[sample.sample_id] = {"revision": sample.revision, "label": sample.label}
        served.update(s.sample_id for s in batch.samples if s.role == "serve" and s.available_at == batch.ordinal)
    return served, answers


def summarize_case(task, state, batches, *, error=None):
    """评价共同的后续服务输出，探索分数不进入跨方案成绩。

    缺预测、缺反馈或运行中断时保留状态和消耗，整段分数留空，不通过
    删除困难样本或失败运行制造较高均值。未释放答案的输出单列等待数量。
    """
    state = state or {"operations": {}, "completed": {}, "predictions": {}, "feedback": {}, "decisions": {}}
    served, expected = expected_feedback(batches)
    actual = {}
    for item in state["feedback"].values():
        if item["source"] == "service_feedback":
            prior = actual.get(item["sample_id"])
            if prior is None or item["label_revision"] > prior["label_revision"]:
                actual[item["sample_id"]] = item
    aligned = set(actual) == set(expected) and all(
        actual[sid]["label_revision"] == value["revision"] and actual[sid]["label"] == value["label"]
        for sid, value in expected.items())
    complete = (error is None and len(state["completed"]) == len(batches)
                and set(state["predictions"]) == served)
    score = None
    if complete and aligned and expected:
        differences = [abs(actual[sid]["prediction"] - value["label"]) for sid, value in expected.items()]
        if task.metric == "macro_f1":
            from harness.evaluation.metrics import macro_f1
            score = macro_f1((v["label"], actual[sid]["prediction"]) for sid, v in expected.items())
        elif task.metric == "accuracy":
            score = sum(actual[sid]["prediction"] == value["label"] for sid, value in expected.items()) / len(expected)
        elif task.metric == "mae":
            score = math.fsum(differences) / len(differences)
        else:
            score = math.sqrt(math.fsum(value * value for value in differences) / len(differences))
    costs = operation_costs(list(state["operations"].values()))
    return {"status": "completed" if complete else "incomplete", "error": error,
            "completed_batches": len(state["completed"]), "expected_batches": len(batches),
            "predicted_samples": len(state["predictions"]), "expected_samples": len(served),
            "labeled_samples": len(actual), "expected_labeled_samples": len(expected),
            "pending_label_samples": len(served - set(expected)), "feedback_aligned": aligned,
            "evaluation_versions": [f"{sid}@{actual[sid]['label_revision']}" for sid in sorted(actual)],
            "feedback_score": score, "objective": task.objective, "cost": costs,
            "adoptions": sum(d["selected_model"] != d["current_model"] for d in state.get("decisions",{}).values()),
            "build_operations": sum(op["name"].startswith("build:") for op in state["operations"].values()),
            "model_experiment": state.get("model_experiment",False)}


def prepare_comparison(task,batches,base,*,factory_identity,extended=False,ordering=False,refinement=False,stage2=False,local_only=False,evaluation=None,reference_spec=None,answer_spec=None,suite="standard",study_definition=None):
    """冻结共同条件，不构造模型、计划器或依赖训练框架。"""
    cases=comparison_cases(base,extended=extended,ordering=ordering,refinement=refinement,stage2=stage2,local_only=local_only,suite=suite)
    if study_definition is not None:
        from harness.evaluation.study import validate_definition
        validate_definition(task, batches, study_definition)
        if answer_spec is None:
            raise ValueError("分阶段研究评价需要独立最终答案")
    from harness.adapters.datasets.selection import ingest, fitting_pools, select_rows
    from harness.adapters.datasets.data_conditions import training_condition
    if not batches:
        raise ValueError("比较需要非空批次")
    if base.initial_pipeline is not None:
        base.initial_pipeline.validate_task(task)
    # 复用控制器的实际首批选择规则；不构造预测器，也不读取未来资料选起点。
    first = batches[0]
    initial_action = ActionPlan("recent", pipeline=base.initial_pipeline)
    initial_state, _ = ingest(task, first, {})
    recent, history = fitting_pools(initial_state, first.ordinal)
    initial_rows = select_rows(task.task_id, first.batch_id, initial_action, recent, history, base, task=task)
    initialization = {"pipeline": asdict(base.initial_pipeline) if base.initial_pipeline else None,
        "pipeline_source": "explicit" if base.initial_pipeline else "backend_default",
        "seed": base.seed, "batch_id": first.batch_id,
        "training_versions": [f"{s.sample_id}@{s.revision}" for s in initial_rows],
        "training_digest": digest([asdict(s) for s in initial_rows]),
        "condition": training_condition(task, initial_rows, initial_action)}
    if answer_spec is not None:
        from harness.evaluation.final_answers import prepare_answers
        prepare_answers(task,batches,answer_spec)
    reference_id=None
    if reference_spec is not None:
        from harness.evaluation.reference_evaluation import validate_reference
        reference_id=validate_reference(task,reference_spec,{"task":asdict(task),"models":{},
            "samples":{s.sample_id:asdict(s) for b in batches for s in b.samples}})
    if (extended or stage2 or suite!="standard") and task.task_type!="classification" and base.loss_scale is None:
        raise ValueError("回归触发对照须先声明公共loss_scale")
    # 源码按职责分包后，记录整个harness树，以相对路径区分各包的同名文件。
    source_root=Path(__file__).resolve().parents[1]
    source_files={p.relative_to(source_root).as_posix():sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source_root.rglob("*.py"))}
    manifest={"format":2,"suite":suite,"study_definition":study_definition,"evidence_kind":base.execution_mode,"task":asdict(task),
        "input_digest":digest([asdict(b) for b in batches]),"factory_identity":factory_identity,
        "source_digest":digest(source_files),"source_files":source_files,"base_config":asdict(base),
        "initialization":initialization,
        "evaluation":"same_released_service_labels_latest_revision","evaluation_definition":evaluation or {},
        "budget_kind":"shared_configured_run_batch_worker_and_service_limits","reference_digest":reference_id,
        "final_answer_digest":digest(answer_spec) if answer_spec is not None else None,
        "cases":[{"id":key,"label":label,"config":asdict(cfg),"fixed_retrain":mode} for key,label,cfg,mode in cases]}
    return json.loads(json.dumps(manifest,ensure_ascii=False,allow_nan=False))


def run_comparison(task,batches,folder,*,config,engine_factory,factory_identity,
                   extended=False,ordering=False,refinement=False,stage2=False,local_only=False,evaluation=None,allow_real_execution=False,reference_spec=None,answer_spec=None,suite="standard",study_definition=None):
    """factory接收(case_id, config, output_folder, planner_kind)，只装配而不执行。

    调用者明确提供后端与服务；真实模式先核对许可，再调用装配。所有组在
    开始前检查任务、配置和工具身份，随后使用同一评价器。失败项保留状态、
    已消耗投入和答案覆盖；已完成运行恢复原记录。
    """
    if config.execution_mode=="real_model" and not allow_real_execution:
        raise ValueError("共同对照的真实执行未获明确启用")
    folder=Path(folder)
    manifest=prepare_comparison(task,batches,config,factory_identity=factory_identity,
                                extended=extended,ordering=ordering,refinement=refinement,stage2=stage2,local_only=local_only,evaluation=evaluation,reference_spec=reference_spec,answer_spec=answer_spec,suite=suite,study_definition=study_definition)
    from harness.evaluation.final_answers import prepare_answers, score_final
    prepared_answers=prepare_answers(task,batches,answer_spec) if answer_spec is not None else None
    cases=comparison_cases(config,extended=extended,ordering=ordering,refinement=refinement,stage2=stage2,local_only=local_only,suite=suite)
    registry=LocalStore(folder)
    summaries=[]
    with registry.exclusive():
        existing=registry.load()
        engines=[]
        for key,label,cfg,mode in cases:
            kind=(mode if isinstance(mode,str) else "service" if mode is None else "periodic" if mode else "static")
            engine=engine_factory(key,cfg,folder/key,kind)
            if engine.task!=task or asdict(engine.config)!=asdict(cfg) or engine.tools.evidence_kind!=config.execution_mode:
                raise ValueError("对照装配改变了共同任务、配置或执行来源")
            engines.append(engine)
        identities={e.tools.identity for e in engines}
        if len(identities)!=1:
            raise ValueError("各组训练工具的配置或版本不一致")
        manifest["resolved_tools_identity"]=next(iter(identities))
        manifest["resolved_planners"]={case[0]:engine.planner.identity for case,engine in zip(cases,engines)}
        if existing not in (None,manifest):
            raise ValueError("比较目录的输入、源码、配置或装配已改变，请使用独立目录")
        registry.save(manifest)
        from harness.evaluation.longitudinal import evaluate_history
        for (key,label,cfg,mode),engine in zip(cases,engines):
            error=None
            try:
                state=engine.run(batches)
            except Exception as exc:
                error=type(exc).__name__
                state=engine.store.load()
            summary={"id":key,"label":label,**summarize_case(task,state,batches,error=error)}
            if state:
                reference=None
                if reference_spec is not None:
                    from harness.evaluation.reference_evaluation import evaluate_references,reference_tools
                    destination=folder/key/'reference-evaluation'
                    reference=evaluate_references(task,state,reference_spec,reference_tools(engine,destination),destination,
                                                  allow_real_execution=allow_real_execution)
                history=evaluate_history(state,**(evaluation or {}),reference_report=reference)
                if prepared_answers is not None:
                    final=score_final(task,state,prepared_answers)
                    summary["final_evaluation"]={k:v for k,v in final.items() if k!="records"}
                    summary["final_longitudinal"]=evaluate_history(state,**(evaluation or {}),final_report=final)
                    atomic_json(folder/key/"final-evaluation.json",final)
                summary["longitudinal"]=history
                summary["evaluation_cost"]=reference["cost"] if reference else None
                atomic_json(folder/key/"longitudinal.json",history)
                atomic_json(folder/key/"evaluation-definition.json",evaluation or {})
            summaries.append(summary)
            report={"manifest_digest":digest(manifest),"model_experiment":config.execution_mode=="real_model",
                    "winner":None,"cases":summaries}
            atomic_json(folder/"comparison.json",report)
        source=("预写模型与模拟计划的工程比较" if config.execution_mode=="interface_fixture" else "按实际模型与服务记录汇总的比较")
        lines=["# 共同条件比较记录","",source+"。", "",
               "共同输入、动作目录、预算和评价定义见state.json；各组实际花费及未完成尝试分别保留。","",
               "| 配置 | 完成批次 | 已返回答案 | 反馈指标 | 终局指标 | 工程点数 | 工作者CPU秒 | 失败操作 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in summaries:
            score="未形成完整可比结果" if row["feedback_score"] is None else f"{row['feedback_score']:.4g}"
            cost=row["cost"]
            final_score=row.get("final_evaluation",{}).get("score")
            final_text="未配置或覆盖未全" if final_score is None else f"{final_score:.4g}"
            lines.append(f"| {row['label']} | {row['completed_batches']}/{row['expected_batches']} | "
                f"{row['labeled_samples']}/{row['expected_labeled_samples']} | {score} | {final_text} | {cost['demo_units']} | "
                f"{cost['worker_cpu_seconds']} | {cost['failed_operations']} |")
        lines += ["","逐阶段/分组表现、恢复定义和全部用量见comparison.json各组longitudinal。",""]
        (folder/"比较说明.md").write_text("\n".join(lines),encoding="utf-8")
        if study_definition is not None:
            from harness.evaluation.study import evaluate_study,render_study
            final_reports = {row["id"]:json.loads((folder/row["id"]/"final-evaluation.json").read_text(encoding="utf-8"))
                             for row in summaries if (folder/row["id"]/"final-evaluation.json").exists()}
            states = {row["id"]:LocalStore(folder/row["id"]).load() for row in summaries}
            study = evaluate_study(task,batches,study_definition,final_reports,states)
            atomic_json(folder/"study-evaluation.json",study)
            (folder/"维护结果.md").write_text(render_study(study),encoding="utf-8")
    return report


def run_fixture_comparison(task,batches,fixture,folder:Path,*,config=None,limits=None,extended=False,evaluation=None,reference_spec=None):
    """兼容工程入口，通过统一装配使用预写模型和模拟HTTP服务。"""
    from harness.engine.legacy.assembly import assemble
    base=config or EngineConfig(policy="plan_feedback_reuse",component_actions=True,
                                active_inspection=True,method_memory=True,adaptive_loop=True)
    if base.execution_mode!="interface_fixture":
        raise ValueError("工程包装入口仅使用interface_fixture")
    if limits is not None:
        base=replace(base,worker_wall_seconds=limits.wall_seconds,worker_cpu_seconds=limits.cpu_seconds,
                     worker_memory_bytes=limits.memory_bytes,worker_poll_seconds=limits.poll_seconds)
    service=PyTrioConfig(model="fixture-qwen")
    def factory(key,cfg,target,kind):
        return assemble(task,cfg,target,service,fixture=fixture,workers=True,planner_kind=kind)
    return run_comparison(task,batches,folder,config=base,engine_factory=factory,
        factory_identity="fixture-assembly-v2:"+digest(fixture),extended=extended,evaluation=evaluation,reference_spec=reference_spec)
