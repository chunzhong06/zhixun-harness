"""统一Harness的批次控制、资料选择、提名、采用与版本接续。"""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from time import perf_counter, time
from typing import Callable

from harness.core.contracts import (ActionPlan, BatchContext, EngineConfig, ModelBundle,
                        ModelTools, Planner, Sample, TaskSpec, digest)
from harness.adapters.datasets.selection import fitting_pools, ingest, select_rows, selection_report
from harness.evaluation.metrics import prediction_score, grouped_scores, compare_evaluations, confirmation_scope, error_profile
from harness.engine.legacy.execution import JobUnconfirmed
from harness.research.methods.planning.planning_facts import FACTS_VERSION, HISTORY_LIMIT, build_facts, validate_action_context
from harness.research.methods.experience.method_guidance import build_method_guidance, guidance_identity
from harness.research.methods.search.action_selection import ACTION_SELECTION_VERSION, build_action_options, proposal_origin
from harness.core.recipe import component_catalog
from harness.learning.models.corrections import inside, selection_key
from harness.research.methods.planning.correction_planning import correction_catalog, resolve_base
from harness.research.methods.search.search_policy import routing_history, trial_routing
from harness.research.methods.diagnosis.problem_tracking import refresh_problems, problem_router
from harness.research.methods.diagnosis.investigation import CheckPlan, check_catalog, execute_check, select_checks
from harness.research.methods.experience.method_memory import METHOD_MEMORY_VERSION, record_methods, retrieve_methods, scope_for
from harness.research.methods.diagnosis.error_regions import frozen_regions
from harness.adapters.storage.local import LocalStore
from harness.engine.legacy.runtime_budget import RuntimeBudget


class UnconfirmedOperation(RuntimeError):
    """外部步骤已开始但结果未确认，接入正式工具时需其幂等/恢复协议。"""


class Harness:
    """持有一个任务的模型、资料和执行状态；三种计划方式共用采用逻辑。

    run在独占运行目录内顺序推进批次。after_event用于工程故障注入，
    在事件已保存后调用；它抛出的异常向外传播，不冒充适配器失败。
    """
    def __init__(self, task: TaskSpec, config: EngineConfig, tools: ModelTools,
                 planner: Planner, store: LocalStore,
                 after_event: Callable[[dict], None] | None = None, *, clock=time,
                 allow_real_execution=False):
        if (tools.evidence_kind == "real_model" or config.execution_mode == "real_model") and not allow_real_execution:
            raise ValueError("当前入口只运行接口样例；真实模型执行须另行接入")
        if tools.evidence_kind != config.execution_mode:
            raise ValueError("工具来源与执行模式不一致")
        if config.initial_pipeline is not None:
            config.initial_pipeline.validate_task(task)
        if config.action_routing=="reward_ucb" and task.task_type=="regression" and config.loss_scale is None:
            raise ValueError("回归收益选择需要事前声明loss_scale")
        if config.execution_mode=="real_model" and not getattr(tools,"capabilities",{}).get("supervised_execution"):
            raise ValueError("真实后端须通过受控工作进程执行")
        self.task, self.config = task, config
        self.tools, self.planner, self.store = tools, planner, store
        self.after_event = after_event
        self.state: dict = {}
        self.clock = clock

    def _save(self):
        self.store.save(self.state)

    def _event(self, key: str, kind: str, **details):
        if any(e["key"] == key for e in self.state["events"]):
            return
        event = {"key": key, "kind": kind, "evidence_kind": self.config.execution_mode, **details}
        self.state["events"].append(event)
        self._save()
        if self.after_event:
            self.after_event(event)

    def _operation(self, batch: BatchContext, name: str, units: int,
                   reserve: int, function: Callable[[], object], recovery=None, requested_action=None, context_key=None,
                   effective_action=None):
        """先登记操作和额度，再调用适配器，最后持久化确认结果。

        已确认结果按稳定步骤ID复用；失败或额度不足返回None，原因保存在记录。
        started状态通过登记工具的恢复接口核对，无法确认时保留该状态。
        本层登记操作，执行限时与外部结果核对由适配器负责。
        """
        key = f"{batch.batch_id}:{name}"
        is_build = name.startswith("build:") or (name.startswith("cv:") and name.endswith(":build"))
        if is_build and hasattr(self.tools, "set_progress_callback"):
            self.tools.set_progress_callback(lambda update: self._event(
                f"{key}:progress:{update['index']}", "execution_progress", batch_id=batch.batch_id,
                operation=name, progress=update))
        old = self.state["operations"].get(key)
        if old:
            if old["status"] == "started":
                receipt = recovery() if recovery is not None else None
                if receipt is None:
                    raise UnconfirmedOperation(f"步骤 {key} 的外部结果未确认")
                old.update(execution=deepcopy(receipt), elapsed_seconds=receipt.get("wall_seconds"))
                if receipt["status"] == "completed":
                    old.update(status="completed", value=receipt["artifact"], recovered=True)
                    self._event(key, "operation_completed", batch_id=batch.batch_id, operation=name)
                elif receipt["status"] == "failed":
                    old.update(status="failed", error=receipt.get("failure", "worker_failed"), recovered=True)
                    self._event(key, "operation_failed", batch_id=batch.batch_id, operation=name, error=old["error"])
                else:
                    raise UnconfirmedOperation(f"步骤 {key} 仍未确认")
            return old.get("value") if old["status"] == "completed" else None
        used = self.state["ledger"].setdefault(batch.batch_id, {"demo_units": 0})
        essential = name.startswith("serve:") or name == "build:initialize"
        service = name.startswith(("plan:", "inspect_plan:")) and hasattr(self.planner, "audit")
        if service and hasattr(self.planner, "uses_service"):
            service = self.planner.uses_service(batch.batch_id)
        runtime = RuntimeBudget(self.state, self.config, self.clock)
        blocked, view = runtime.admit(batch.batch_id, essential=essential, service=service)
        if blocked:
            self._event(f"{key}:runtime_budget", "runtime_budget_stop", batch_id=batch.batch_id,
                        operation=name, reason=blocked, budget=view)
            return None
        # 维护步骤额外留出一次原版输出，避免采用通过后才挤占接续额度。
        reserve += int(self.config.output_fallback and not essential)
        if used["demo_units"] + units + reserve > self.config.demo_budget_per_batch:
            self._event(f"{key}:budget", "budget_stop", batch_id=batch.batch_id, operation=name)
            return None
        used["demo_units"] += units
        operation = {"status": "started", "demo_units": units, "elapsed_seconds": None,
                     "batch_id": batch.batch_id, "name": name}
        if service:
            operation["service_operation"] = True
        if service and self.config.service_call_reserve_tokens:
            operation["service_reservation_tokens"] = self.config.service_call_reserve_tokens
        if view["remaining_seconds"] is not None:
            available = view["remaining_seconds"] - (0 if essential else self.config.output_reserve_seconds)
            operation["runtime_timeout_seconds"] = available
            owner = self.planner if service else self.tools
            if hasattr(owner, "set_runtime_timeout"):
                owner.set_runtime_timeout(available)
        if requested_action is not None:
            operation["requested_action"] = deepcopy(requested_action)
        if effective_action is not None:
            operation["effective_action"] = deepcopy(effective_action)
        if context_key is not None:
            operation["context_key"] = context_key
        self.state["operations"][key] = operation
        self._save()
        if not service and hasattr(self.tools,"last_execution"):
            self.tools.last_execution = None
        started = perf_counter()
        audit = getattr(self.planner, "audit", [])
        audit_start = len(audit)
        def attach_receipts():
            if not service and getattr(self.tools, "last_execution", None) is not None:
                operation["execution"] = deepcopy(self.tools.last_execution)
            if name.startswith(("plan:", "inspect_plan:")):
                operation["service_calls"] = [{k: entry[k] for k in ("request_id", "usage", "status", "evidence_kind")}
                    for entry in audit[audit_start:]]
        try:
            result = function()
            # 在登记完成前验证JSON可保存性，使恢复能读取相同产物。
            digest(result)
        except JobUnconfirmed as exc:
            # 工作者可能仍在执行，保持started以便按原身份继续核查。
            raise UnconfirmedOperation(f"步骤 {key} 的工作进程尚未确认") from exc
        except Exception as exc:
            attach_receipts()
            operation.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                             elapsed_seconds=perf_counter() - started)
            if hasattr(exc, "repairable"):
                operation.update(plan_repairable=exc.repairable, error_code=exc.code)
            self._event(key, "operation_failed", batch_id=batch.batch_id,
                        operation=name, error=operation["error"])
            return None
        attach_receipts()
        operation.update(status="completed", value=result, elapsed_seconds=perf_counter() - started)
        self._event(key, "operation_completed", batch_id=batch.batch_id, operation=name)
        return result

    def _bundle(self, model_id: str) -> ModelBundle:
        bundle = ModelBundle.parse(self.state["models"][model_id])
        if bundle.task_signature != self.task.signature:
            raise ValueError("版本与本任务不兼容")
        return bundle

    def _predictions(self, bundle: ModelBundle, rows: tuple[Sample, ...], *, service=False) -> dict:
        inputs = tuple({"sample_id": s.sample_id, "values": list(s.values)} for s in rows)
        # 适配器取得自己的副本，不能通过修改参数覆盖已保存版本。
        predict = getattr(self.tools, "predict_service", self.tools.predict) if service else self.tools.predict
        predictions = predict(deepcopy(bundle.artifact), inputs)
        if set(predictions) != {s.sample_id for s in rows}:
            raise ValueError("预测缺少样本、含额外样本或身份错配")
        if any(not self.task.accepts_target(p) for p in predictions.values()):
            raise ValueError("预测值与任务目标不兼容")
        return predictions

    def _score(self, batch: BatchContext, bundle: ModelBundle,
               rows: tuple[Sample, ...], purpose: str, reserve: int) -> dict | None:
        if not rows:
            return None
        if set(bundle.training_ids).intersection(s.sample_id for s in rows):
            raise ValueError("共同评价包含该版本已拟合的样本")

        def evaluate(predictions=None):
            predictions = self._predictions(bundle, rows) if predictions is None else predictions
            score = prediction_score(self.task, rows, predictions)
            result = {"score": score, "predictions": predictions,
                    "groups": grouped_scores(self.task, rows, predictions),
                    "error_profile": error_profile(self.task, rows, predictions),
                    "sample_ids": [s.sample_id for s in rows], "model_id": bundle.model_id,
                    "sample_versions": [f"{s.sample_id}@{s.revision}" for s in rows],
                    "purpose": purpose, "objective": self.task.objective,
                    "evidence_kind": self.config.execution_mode}
            if purpose == "explore":
                from harness.evaluation.metrics import exploration_windows
                result.update(exploration_windows(self.task,rows,predictions,set(self.state["predictions"]),
                                                  self.config.explore_current_weight))
                windows=result["observation_windows"]
                diagnostic="current" if "current" in windows else "returned"
                result.update(pooled_error_profile=result["error_profile"],diagnostic_window=diagnostic,
                              error_profile=windows[diagnostic]["error_profile"])
            return result

        def recover():
            receipt=self.tools.recover_predict(bundle.artifact,tuple({"sample_id":s.sample_id,"values":list(s.values)} for s in rows))
            if receipt and receipt["status"] == "completed":
                return {**receipt,"artifact":evaluate(receipt["artifact"])}
            return receipt
        return self._operation(batch, f"{purpose}:{bundle.model_id}", 1, reserve, evaluate,
            recovery=recover if hasattr(self.tools,"recover_predict") else None)

    def _serve(self,batch,bundle,rows,reserve=0):
        inputs=tuple({"sample_id":s.sample_id,"values":list(s.values)} for s in rows)
        recover = (getattr(self.tools, "recover_service", None) if hasattr(self.tools, "predict_service")
                   else getattr(self.tools, "recover_predict", None))
        return self._operation(batch,f"serve:{bundle.model_id}",1,reserve,
            lambda:self._predictions(bundle,rows,service=True),
            recovery=(lambda:recover(bundle.artifact,inputs)) if recover else None)

    def _effective_action(self, action: ActionPlan, parent_id: str | None):
        """将普通重训解析为父版本配方；显式完整配方用于主动替换组件。"""
        if action.pipeline is None and parent_id is not None:
            return replace(action, pipeline=self._bundle(parent_id).pipeline)
        return action

    def _select_rows(self, batch, action, recent, history):
        """共用选择与回执入口；组件修改不改变配方抽样，恢复可核对原清单。"""
        rows = select_rows(self.task.task_id, batch.batch_id, action, recent, history,
                           self.config, task=self.task)
        if action.fit_manifest is not None:
            rows = self.tools.select_fit_rows(action.fit_manifest, rows)
        base = resolve_base(self, batch, action)
        if base:
            used = set(base["bundle"]["training_ids"])
            rows = tuple(s for s in rows if s.sample_id not in used and inside(s.values, base["predicates"]))
        report = selection_report(self.task, action, recent, history, rows, self.config)
        if action.fit_manifest is not None:
            report["fit_manifest"] = action.fit_manifest
        if base:
            report["correction_base"] = base["bundle"]["model_id"]
        key = selection_key(batch.batch_id, action)
        reports = self.state.setdefault("data_selections", {})
        if key in reports and reports[key] != report:
            raise ValueError("同一资料配方的抽样结果在恢复时发生变化")
        reports[key] = report
        self._event(key, "data_selected", batch_id=batch.batch_id, report=report)
        return rows

    def _training_manifest(self, action, rows, parent_id):
        effective = self._effective_action(action, parent_id)
        parts = [[(s.sample_id, s.revision) for s in rows], asdict(effective.pipeline) if effective.pipeline else None]
        if effective.epochs is not None:
            parts.extend([effective.epochs, self._bundle(parent_id).artifact])
        if effective.focus_label is not None:
            parts.append(effective.focus_label)
        if effective.gas_pair:
            parts.extend([asdict(effective.gas_pair), self._bundle(parent_id).artifact])
        if effective.correction:
            parts += [asdict(effective.correction), self._bundle(effective.correction.base_model_id).artifact]
        return digest(parts)

    def _build(self, batch: BatchContext, action: ActionPlan,
               rows: tuple[Sample, ...], parent_id: str | None, initialize=False, *, variant=None):
        # 先解析实际配方再核对可训练条件；采用过的森林等组件不会被空计划重置。
        effective = self._effective_action(action, parent_id)
        base = resolve_base(self, batch, effective)
        options = {"base": base} if base else {}
        if effective.epochs is not None or effective.gas_pair:
            options["parent"] = self._bundle(parent_id).artifact
        action_key = "initialize" if initialize else action.key
        if variant:
            action_key += "@" + variant
        from harness.adapters.datasets.data_conditions import training_condition
        condition=training_condition(self.task,rows,effective)
        if base and len(rows) < 2:
            condition.update(ready=False, reason="insufficient_fresh_correction_rows")
        self.state.setdefault("fit_conditions", {})[f"{batch.batch_id}:build:{action_key}"] = {
            "fit_condition": condition, "effective_action": effective.to_record(),
            "requested_action": action.to_record()}
        if not condition["ready"]:
            self._event(f"{batch.batch_id}:fit_unavailable:{action.key}:{variant}","fit_unavailable",
                        batch_id=batch.batch_id,action=action.to_record(),condition=condition)
            return None
        model_id = f"{batch.batch_id}/{action_key}"
        if model_id in self.state["models"]:
            return self._bundle(model_id)
        # 只有能保证同输入产生同权重的后端提供此键。探索与确认始终在当前批次重新评分。
        fit_key = None
        if not initialize and getattr(self.tools, "reuse_identical_fit", False):
            fit_key = self.tools.fit_identity(self.task, effective, rows, options["parent"])
            previous = self.state.get("completed_fits", {}).get(fit_key)
            if previous:
                bundle = self._bundle(previous)
                self.tools.verify_artifact(bundle.artifact)
                self._event(f"{batch.batch_id}:cached_fit:{action_key}", "fit_reused",
                            batch_id=batch.batch_id, model_id=previous, fit_key=fit_key)
                return bundle
        artifact = self._operation(
            batch, f"build:{action_key}", 2 if initialize else 3,
            1 if initialize else 4,
            lambda: self.tools.build(self.task, batch.batch_id, effective, rows, initialize=initialize, **options),
            recovery=(lambda: self.tools.recover_build(self.task, batch.batch_id, effective, rows,
                initialize=initialize, **options)) if hasattr(self.tools, "recover_build") else None,
            requested_action=action.to_record(), effective_action=effective.to_record(),
        )
        if artifact is None:
            return None
        training_ids = {s.sample_id for s in rows}
        versions = {f"{s.sample_id}@{s.revision}" for s in rows}
        if base:
            training_ids.update(base["bundle"]["training_ids"])
            versions.update(base["bundle"]["training_versions"])
        # 组合产物保存分支配方；普通全量重训继承基础配方，避免把残差模型误当作业务模型。
        bundle = ModelBundle(model_id, self.task.signature, parent_id,
                             tuple(sorted(training_ids)) if base else tuple(s.sample_id for s in rows),
                             tuple(sorted(versions)) if base else tuple(f"{s.sample_id}@{s.revision}" for s in rows),
                             artifact, self.tools.evidence_kind,
                             ModelBundle.parse(base["bundle"]).pipeline if base else effective.pipeline,
                             {**asdict(effective.correction), "base_artifact": base["bundle"]["artifact"],
                              "branch_pipeline": asdict(effective.pipeline), "depth": 1} if base else None)
        self.state["models"][model_id] = asdict(bundle)
        if fit_key is not None:
            self.state.setdefault("completed_fits", {})[fit_key] = model_id
        self._event(f"{batch.batch_id}:bundle:{action_key}", "bundle_saved",
                    batch_id=batch.batch_id, model_id=model_id,
                    recipe=action.to_record(), effective_recipe=effective.to_record(),
                    pipeline_source="explicit" if action.pipeline else "incumbent" if effective.pipeline else "backend_default",
                    training_ids=list(bundle.training_ids),
                    training_versions=list(bundle.training_versions))
        return bundle

    def _feedback(self, batch: BatchContext):
        for sample in batch.samples:
            pred = self.state["predictions"].get(sample.sample_id)
            if pred is None or sample.label is None:
                continue
            key = f"service:{sample.sample_id}:{sample.revision}"
            if key not in self.state["feedback"]:
                prior = self.state.get("gas_prior_predictions", {}).get(sample.sample_id, ())
                if prior or "gas_repair" in pred:
                    from harness.evaluation.gas_service import matches_observation
                    matches = [p for p in (*prior, pred) if matches_observation(p, sample)]
                    if len(matches) != 1:
                        # 答案可能带了修订后的测量，或同一身份曾多次发出。仅给唯一对应的预测记账。
                        entry = {"source": "unmatched_gas_feedback", "sample_id": sample.sample_id,
                                 "label_revision": sample.revision, "label": sample.label, "received_in": batch.batch_id,
                                 "reason": "ambiguous_issue" if matches else "input_mismatch"}
                        self.state["feedback"][key] = entry
                        self._event(key, "unmatched_gas_feedback", **entry)
                        continue
                    pred = matches[0]
                result = {"source": "service_feedback", "sample_id": sample.sample_id,
                          "label_revision": sample.revision, "received_in": batch.batch_id,
                          "model_id": pred["model_id"], "prediction": pred["prediction"],
                          "label": sample.label, "objective": self.task.objective,
                          "evidence_kind": self.config.execution_mode}
                if self.task.task_type == "classification":
                    result["correct"] = pred["prediction"] == sample.label
                    if self.task.task_id == "official-gas-A":
                        from harness.evaluation.gas_service import assess_return
                        result.update(prediction_batch=pred["batch_id"], prediction_revision=pred.get("sample_revision"),
                                      prediction_group=pred.get("group"),
                                      gas_repair_result=assess_return(pred, sample))
                else:
                    result["absolute_error"] = abs(pred["prediction"] - sample.label)
                self.state["feedback"][key] = result
                self._event(key, "feedback_received", **result)
        from harness.research.methods.diagnosis.problem_tracking import update_problem_feedback
        update_problem_feedback(self)

    def _execute_check(self, batch, plan, recent, explore, baseline):
        """从运行侧状态取得冻结区域与模型拟合身份，复用已有探索预测。"""
        region_context = None
        if plan.tool == "error_regions":
            model = self.state["models"][baseline["model_id"]]
            training_rows = tuple(Sample.parse(self.state["samples"][sid]) for sid in model["training_ids"])
            region_context = {"ordinal": batch.ordinal, "history": frozen_regions(self.state, batch.batch_id),
                              "training_rows": training_rows}
        return execute_check(plan, self.task, recent, explore, baseline, region_context=region_context)

    def _context(self, batch, recent, history, baseline, trials, scores):
        facts, evidence = build_facts(self.task, batch, recent, history, baseline, trials, self.state, self.config,
                                      signed_groups=getattr(self.tools, "capabilities", {}).get("group_error_direction", False),
                                      check_devices=getattr(self.tools, "capabilities", {}).get("device_bias", False))
        checks = self.state["inspections"].get(batch.batch_id, [])
        region_history = [{key: region[key] for key in ("id", "predicates", "model_id", "ordinal", "discovery")}
                          for region in frozen_regions(self.state, batch.batch_id)] if self.config.active_inspection else []
        methods = (retrieve_methods(self.state["methods"], scope_for(self.task, recent, checks),
                   recent=recent, mode=self.config.method_scope, distance_limit=self.config.method_distance_limit,
                   current_model=self.state["models"].get(self.state["current"]), task_type=self.task.task_type)
                   if self.config.method_memory else [])
        if self.config.component_actions:
            extra = [("investigations", {"results": checks, "frozen_regions": region_history}),
                     ("methods", {"candidates": methods})]
            if self.config.problem_tracking:
                extra.append(("problems", refresh_problems(self,batch,baseline,trials)))
            for kind, value in extra:
                key = "fact-" + digest([self.task.signature, batch.batch_id, kind, value])
                item = {"id": key, "kind": kind, "value": value, "evidence_kind": self.state["evidence_kind"]}
                facts["evidence"].append(item)
                evidence[key] = {"item": item, "scope": {"task_signature": self.task.signature, "batch_id": batch.batch_id},
                                 "sources": {"kind": kind, "ids": [m["id"] for m in methods] if kind == "methods" else
                                             sorted({p["evaluation_key"] for p in value["items"]}) if kind == "problems" else
                                             [c["check_key"] for c in checks]}}
        self.state["planning_evidence"].update(evidence)
        # 计划快照与运行中的trials列表分离，后续尝试不能回写先前看见的事实。
        context = {
            "task_id": self.task.task_id, "batch_id": batch.batch_id,
            "task": asdict(self.task),
            "objective": self.task.objective,
            "policy": self.config.policy,
            "facts": facts,
            "baseline_score": baseline["score"] if baseline else None,
            "best_exploration_score": self.task.utility(max(scores.values())) if scores else None,
            "trials": trials,
            "allowed_recipes": {"recent": [0], "uniform": [0.5, 1], "coverage": [0.5, 1]},
            "remaining_demo_units": self.config.demo_budget_per_batch - self.state["ledger"][batch.batch_id]["demo_units"],
            "past_decisions": [v["decision"] for v in list(self.state["completed"].values())[-HISTORY_LIMIT:]],
        }
        capabilities = getattr(self.tools, "capabilities", {})
        allowed_epochs = capabilities.get("continuation_epochs")
        if hasattr(self.tools, "prepare_fit_manifests") and baseline:
            # 动态选样需绑定实际在用权重；确认样本始终留在采用器。
            parent = self.state["models"][baseline["model_id"]]["artifact"]
            self.tools.prepare_fit_manifests(batch, baseline, parent)
        if hasattr(self.tools, "fit_manifest_context"):
            context["fit_manifests"] = self.tools.fit_manifest_context(batch, baseline)
        if allowed_epochs is not None:
            context["continuation_epochs"] = list(allowed_epochs)
            if capabilities.get("fixed_public_training"):
                # 随包工具只接受已冻结的训练清单；历史重采样不是一个可执行选项。
                context["allowed_recipes"] = {"recent": [0]}
        if capabilities.get("gas_pair_methods") or capabilities.get("focus_labels"):
            from harness.research.methods.planning.gas_continuation import gas_planning_context
            context.update(gas_planning_context(capabilities, batch, baseline, self.state))
        if self.config.component_actions:
            components = component_catalog()
            if not self.task.condition_features:
                components["processor"].remove("condition_standard")
            context["maintenance"] = {"work_order_id": f"{self.task.task_id}/{batch.batch_id}",
                "stage": "improvement", "components": components,
                "batch_ordinal": batch.ordinal, "trial_limit": self.config.max_trials,
                "runtime_budget": RuntimeBudget(self.state, self.config, self.clock).snapshot(batch.batch_id),
                "execution_capabilities": deepcopy(getattr(self.tools, "capabilities", {})),
                "entry": deepcopy(self.state.get("maintenance_entries", {}).get(batch.batch_id)),
                "resource_strategy": self.config.resource_strategy,
                "refinement_basis": self.config.refinement_basis,
                "candidate_pool_cap": self.config.candidate_pool_cap, "candidate_min_support": self.config.candidate_min_support,
                "action_routing": self.config.action_routing, "routing_step": self.config.routing_step,
                "routing_bias_limit": self.config.routing_bias_limit, "routing_epsilon": self.config.routing_epsilon,
                "routing_targets": deepcopy(self.config.routing_targets), "routing_history": routing_history(self.state, batch.batch_id, trials),
                "routing_reward_window":self.config.routing_reward_window,"routing_ucb_exploration":self.config.routing_ucb_exploration,
                "routing_loss_scale":self.config.loss_scale,
                "inspection_order": self.config.inspection_order,
                "available_checks": check_catalog() if self.config.active_inspection else {},
                "max_checks": self.config.max_checks,
                "check_candidate_limit":4 if self.config.inspection_order=="decision_coverage" else self.config.max_checks,
                "methods": methods}
            context["maintenance"]["corrections"] = correction_catalog(self, batch, recent, history, trials)
            if self.config.problem_tracking:
                context["maintenance"]["problem_router"] = problem_router(context,self.state,self.config,batch.batch_id)
            from harness.research.methods.diagnosis.inspection_followup import pending_followups
            context["maintenance"]["inspection_followups"] = pending_followups(self.state,batch.batch_id,facts)
            context["maintenance"]["error_region_history"] = region_history
            from harness.adapters.datasets.data_conditions import training_condition
            conditions = {}
            for candidate in build_action_options(context).get("candidates", []):
                action = ActionPlan.parse(candidate["action"])
                rows = select_rows(self.task.task_id,batch.batch_id,action,recent,history,self.config,task=self.task)
                base = resolve_base(self,batch,action)
                if base:
                    used = set(base["bundle"]["training_ids"])
                    rows = tuple(s for s in rows if s.sample_id not in used and inside(s.values,base["predicates"]))
                condition = training_condition(self.task,rows,self._effective_action(action,self.state["current"]))
                if base and len(rows)<2:
                    condition.update(ready=False,reason="insufficient_fresh_correction_rows")
                conditions[action.key] = condition
            # 只读选择不写构建记录；真正执行时按同一清单再核对，费用由实际操作登记。
            context["maintenance"]["action_conditions"] = conditions
            if self.config.action_guidance:
                context["maintenance"]["action_options"] = build_action_options(context)
        if self.config.method_guidance:
            context["method_guidance"] = build_method_guidance(context)
        return deepcopy(context)

    def _inspect(self, batch, recent, history, explore, baseline, scores):
        """冻结检查计划、执行只读工具并回传新证据；恢复复用已确认操作。"""
        if not self.config.active_inspection or batch.batch_id in self.state["inspection_done"]:
            return
        context_key = f"{batch.batch_id}:inspection_context:0"
        saved = next((e for e in self.state["events"] if e["key"] == context_key), None)
        if saved:
            context = deepcopy(saved["context"])
        else:
            context = self._context(batch, recent, history, baseline, [], scores)
            context["maintenance"]["stage"] = "inspection"
        self._event(context_key, "inspection_context", context=context)
        reply = self._operation(batch, "inspect_plan:0", 1, 4, lambda: self.planner.propose(deepcopy(context)))
        plans = []
        if reply is not None:
            try:
                if (set(reply) != {"checks", "stop"} or type(reply["stop"]) is not bool
                        or not isinstance(reply["checks"], list) or len(reply["checks"]) > (4 if self.config.inspection_order=="decision_coverage" else self.config.max_checks)
                        or (reply["stop"] and reply["checks"])):
                    raise ValueError("检查计划格式无效")
                plans = [CheckPlan.parse(c) for c in reply["checks"]]
                if len({p.tool for p in plans}) != len(plans):
                    raise ValueError("同一次检查计划重复使用相同工具")
                for plan in plans:
                    if plan.problem_id is not None:
                        raise ValueError("前置检查尚无问题路由；具体问题检查在反馈循环中执行")
                    if plan.tool not in context["maintenance"]["available_checks"]:
                        raise ValueError("该检查未向当前任务开放")
                    validate_action_context(plan.if_true, context)
                    validate_action_context(plan.if_false, context)
            except (ValueError, TypeError) as exc:
                plans = []
                self._event(f"{batch.batch_id}:inspection_invalid", "inspection_invalid", reason=str(exc))
        results = []
        ordering_key = f"{batch.batch_id}:inspection_order"
        saved_selection = next((e for e in self.state["events"] if e["key"]==ordering_key),None)
        if saved_selection:
            plans = [CheckPlan.parse(raw) for raw in saved_selection["selected_plans"]]
        else:
            plans, ordering = select_checks(plans,self.task,recent,explore,baseline["model_id"],self.state,self.config)
            self._event(ordering_key, "inspection_order", criteria=ordering,
                        selected=[plan.tool for plan in plans],selected_plans=[asdict(p) for p in plans])
        for index, plan in enumerate(plans):
            name = f"inspect:{index}:{plan.tool}"
            result = self._operation(batch, name, check_catalog()[plan.tool]["cost_demo_units"], 4,
                                     lambda: self._execute_check(batch, plan, recent, explore, baseline))
            if result is not None:
                result = {**deepcopy(result), "check_key": f"{batch.batch_id}:{name}"}
                results.append(result)
                self._event(f"{batch.batch_id}:inspection_result:{index}", "inspection_result", result=result)
        self.state["inspections"][batch.batch_id] = results
        self.state["inspection_done"].append(batch.batch_id)
        self._event(f"{batch.batch_id}:inspection_done", "inspection_finished", count=len(results))

    def _confirm_candidate(self, batch, current, adopt, scores, cv_nominee):
        """先冻结提名，再在共同资料上确认；恢复时使用已保存的决定。"""
        # 同分优先当前版本，其余按稳定身份排序，不依赖候选产生顺序。
        nominee = current.model_id
        if cv_nominee:
            nominee = cv_nominee.model_id
        elif scores:
            nominee = min(scores, key=lambda mid: (-scores[mid], mid != current.model_id, mid))
        self._event(f"{batch.batch_id}:nominated", "candidate_frozen", batch_id=batch.batch_id,
                    nominee=nominee, current=current.model_id)
        selected_id = current.model_id
        decision = {"current_model": current.model_id, "nominee": nominee,
                    "selected_model": current.model_id, "reason": "探索比较后保持当前版本",
                  "improvement": None, "objective": self.task.objective, "evidence_kind": self.config.execution_mode}
        if batch.batch_id in self.state["decisions"]:
            # 发布前中断时沿用已确认决定，避免已使用采用资料使它被改判为保持。
            decision = self.state["decisions"][batch.batch_id]
            selected_id = decision["selected_model"]
        elif nominee != current.model_id:
            if not adopt or any(s.sample_id in self.state["used_adoption_ids"] for s in adopt):
                decision["reason"] = "缺少新的共同采用资料，保持当前版本"
            else:
                base = self._score(batch, current, adopt, "adopt", 2)
                candidate = self._score(batch, self._bundle(nominee), adopt, "adopt", 1) if base else None
                if base:
                    self.state["used_adoption_ids"] = sorted(set(self.state["used_adoption_ids"]) | {s.sample_id for s in adopt})
                if base and candidate:
                    delta = self.task.utility(candidate["score"]) - self.task.utility(base["score"])
                    decision.update(improvement=delta, baseline_score=base["score"], candidate_score=candidate["score"])
                    decision["comparison"] = compare_evaluations(self.task, base, candidate)
                    if decision["comparison"]["status"] == "matched":
                        explored = self.state["operations"].get(f"{batch.batch_id}:explore:{nominee}", {}).get("value")
                        scope = confirmation_scope(self.task.task_id, explored, candidate)
                        if scope is not None:
                            decision["confirmation_scope"] = scope
                    if delta > self.config.min_improvement + 1e-12:
                        selected_id = nominee
                        decision["reason"] = "共同采用比较通过，采用候选版本"
                    else:
                        decision["reason"] = "共同采用比较未超过门槛，保持当前版本"
                else:
                    decision["reason"] = "采用比较未完成，保持当前版本"
        decision["selected_model"] = selected_id
        self.state["decisions"][batch.batch_id] = decision
        self._event(f"{batch.batch_id}:decision", "adoption_decision", batch_id=batch.batch_id, **decision)
        return decision

    def _serve_selected(self, batch, current, serve, decision):
        """输出失败时可回退；返回最终生效决定及该版本实际发出的预测。"""
        selected_id = decision["selected_model"]
        outputs = self._serve(batch,self._bundle(selected_id),serve,
                             1 if self.config.output_fallback and selected_id != current.model_id else 0) if serve else {}
        if outputs is None and self.config.output_fallback and selected_id != current.model_id:
            rejected_id = selected_id
            outputs = self._serve(batch,current,serve)
            if outputs is not None:
                selected_id = current.model_id
                decision = {**decision, "approved_model": rejected_id, "selected_model": selected_id,
                            "reason": "候选未完成服务输出，由当前可用版本接续", "output_fallback": True}
                self.state["decisions"][batch.batch_id] = decision
                self._event(f"{batch.batch_id}:fallback", "output_fallback", batch_id=batch.batch_id, **decision)
        if outputs is None:
            raise RuntimeError("服务预测未完成；本批尚未提交")
        return decision, outputs

    def _run_batch(self, batch: BatchContext) -> dict:
        fingerprint = digest(asdict(batch))
        completed = self.state["completed"].get(batch.batch_id)
        if completed:
            if completed["input_digest"] != fingerprint:
                raise ValueError("已完成批次的内容发生改变，请使用新的修订批次")
            return completed
        if batch.ordinal <= self.state["last_ordinal"]:
            raise ValueError("批次顺序倒退")
        pending = self.state["pending"]
        if pending and pending != {"batch_id": batch.batch_id, "input_digest": fingerprint}:
            raise ValueError("上一批尚未完成，或待处理批次内容已变化")
        if not pending:
            rows, counts = ingest(self.task, batch, self.state["samples"])
            self.state["samples"] = rows
            self.state["receipts"][batch.batch_id] = counts
            self.state["pending"] = {"batch_id": batch.batch_id, "input_digest": fingerprint}
            RuntimeBudget(self.state, self.config, self.clock).enter(batch.batch_id)
            self._event(f"{batch.batch_id}:received", "batch_received",
                        batch_id=batch.batch_id, **counts)
        self._feedback(batch)
        recent, history = fitting_pools(self.state["samples"], batch.ordinal)
        if getattr(self.tools, "capabilities", {}).get("fixed_public_training"):
            # 优化轮不会带来新的训练标签。重复使用原训练池，保留样本的原到达时刻。
            recent, history = (*history, *recent), ()
        explore = tuple(s for s in batch.samples if s.role == "explore" and s.available_at == batch.ordinal)
        adopt = tuple(s for s in batch.samples if s.role == "adopt" and s.available_at == batch.ordinal)
        serve = tuple(s for s in batch.samples if s.role == "serve" and s.available_at == batch.ordinal)
        if self.state["current"] is None:
            initial_action = ActionPlan("recent", reason="按共同初始配置建立起点",
                                        pipeline=self.config.initial_pipeline)
            initial_rows = self._select_rows(batch, initial_action, recent, history)
            initial = self._build(batch, initial_action, initial_rows, None, True)
            if initial is None:
                raise RuntimeError("没有可用初始版本，无法完成本批预测")
            self.state["current"] = initial.model_id
            self._event(f"{batch.batch_id}:initialized", "initialized", model_id=initial.model_id)
        current = self._bundle(self.state["current"])
        from harness.research.methods.planning.maintenance import maintenance_entry
        entry = maintenance_entry(self.state,self.task,batch,self.config)
        self._event(f"{batch.batch_id}:maintenance_entry", "maintenance_entry", result=entry)
        use_cv = entry["enter"] and self.config.fit_cv_folds > 0 and len(explore) < self.config.min_explore_for_cv
        baseline_result = self._score(batch, current, explore, "explore", 3) if entry["enter"] and not use_cv else None
        scores = {current.model_id: self.task.utility(baseline_result["score"])} if baseline_result else {}
        if baseline_result:
            key = f"diagnostic:{batch.batch_id}"
            self.state["feedback"][key] = {"source": "new_batch_diagnostic", **baseline_result}
            self._save()
            if not (self.config.adaptive_loop and self.config.policy != "plan_once"):
                self._inspect(batch, recent, history, explore, baseline_result, scores)
        trials = []
        cv_nominee = None
        if use_cv:
            from harness.research.methods.search.cross_validation import nominate_with_cv
            cv_rows = self._select_rows(batch, ActionPlan("recent"), recent, history)
            cv_nominee, cv_records = nominate_with_cv(self,batch,cv_rows,current)
            trials.append({"action":"group_cross_validation","model_id":cv_nominee.model_id if cv_nominee else None,
                           "status":"completed" if cv_nominee else "unavailable","cv_records":cv_records})
        if self.config.policy == "plan_feedback_reuse" and baseline_result:
            for mid in reversed(self.state["archive"]):
                bundle = self._bundle(mid)
                result = self._score(batch, bundle, explore, "explore", 3)
                trials.append({"action": "reuse", "model_id": mid,
                               "status": "completed" if result else "unavailable"})
                if result:
                    scores[mid] = self.task.utility(result["score"])
        tried = set()
        training_manifests = set()
        plan_limit = 1 if self.config.policy == "plan_once" else self.config.max_plan_calls
        if self.config.adaptive_loop and self.config.policy != "plan_once":
            from harness.research.methods.planning.decision_loop import run_decision_loop
            trials = run_decision_loop(self, batch, recent, history, explore, baseline_result, current, scores, trials)
        elif baseline_result:
            for index in range(plan_limit):
                context_key = f"{batch.batch_id}:plan_context:{index}"
                saved = next((e for e in self.state["events"] if e["key"] == context_key), None)
                # 恢复重用当时的快照，避免后来完成的操作与费用进入旧计划。
                context = deepcopy(saved["context"]) if saved else self._context(
                    batch, recent, history, baseline_result, trials, scores)
                self._event(context_key, "plan_context", context=context)
                response = self._operation(batch, f"plan:{index}", 1, 3,
                                           lambda: self.planner.propose(deepcopy(context)))
                if response is None:
                    break
                try:
                    if not isinstance(response, dict) or set(response) != {"actions", "stop"}:
                        raise ValueError("计划应只包含actions和stop")
                    if type(response["stop"]) is not bool or not isinstance(response["actions"], list):
                        raise ValueError("计划字段类型错误")
                    actions = [ActionPlan.parse(a) for a in response["actions"]]
                    for action in actions:
                        validate_action_context(action, context)
                    if response["stop"] and actions:
                        raise ValueError("停止计划包含执行动作")
                except (ValueError, TypeError) as exc:
                    self._event(f"{batch.batch_id}:plan_invalid:{index}", "plan_invalid", reason=str(exc))
                    break
                if response["stop"] or not actions:
                    break
                self._event(f"{batch.batch_id}:plan_accepted:{index}", "plan_accepted",
                            batch_id=batch.batch_id, context_key=context_key,
                            actions=[action.to_record() for action in actions])
                for action in actions:
                    if action.key in tried or len(tried) >= self.config.max_trials:
                        continue
                    tried.add(action.key)
                    selected = self._select_rows(batch, action, recent, history)
                    manifest = self._training_manifest(action, selected, current.model_id)
                    if not selected or manifest in training_manifests:
                        trials.append({"action": action.key, "status": "skipped_duplicate_or_empty",
                                       "model_id": None, "training_ids": [s.sample_id for s in selected]})
                        continue
                    training_manifests.add(manifest)
                    model = self._build(batch, action, selected, current.model_id)
                    result = self._score(batch, model, explore, "explore", 3) if model else None
                    trials.append({"action": action.key, "status": "completed" if result else "unavailable",
                                   "model_id": model.model_id if model else None,
                                   "exploration_score": result["score"] if result else None,
                                   "training_ids": [s.sample_id for s in selected]})
                    if self.config.component_actions:
                        trials[-1].update(action_plan=action.to_record(), parent_id=current.model_id,
                            proposal_origin=proposal_origin(context, action),
                            **trial_routing(context, action),
                            training_versions=[f"{s.sample_id}@{s.revision}" for s in selected],
                            evaluation_versions=[f"{s.sample_id}@{s.revision}" for s in explore])
                    if result:
                        scores[model.model_id] = self.task.utility(result["score"])
                if len(tried) >= self.config.max_trials:
                    break
        if self.config.problem_tracking:
            report = refresh_problems(self,batch,baseline_result,trials)
            self._event(f"{batch.batch_id}:problems", "problem_progress", report=report)
        decision = self._confirm_candidate(batch, current, adopt, scores, cv_nominee)
        decision, outputs = self._serve_selected(batch, current, serve, decision)
        selected_id = decision["selected_model"]
        if self.config.method_memory:
            observations = record_methods(self.state, self.task, batch.batch_id, recent,
                self.state["inspections"].get(batch.batch_id, []), trials, decision)
            for item in observations:
                self._event(f"{batch.batch_id}:method:{item['method_id']}", "method_observed", **item)
        # 回退及恢复均从最终交付版本的已完成服务操作取回执，避免串入失败候选的过程。
        served_op = self.state["operations"].get(f"{batch.batch_id}:serve:{selected_id}", {})
        repairs = (served_op.get("execution") or {}).get("gas_repairs", {})
        for sid, prediction in outputs.items():
            source=next(s for s in serve if s.sample_id==sid)
            previous = self.state["predictions"].get(sid)
            if self.task.task_id == "official-gas-A" and previous and previous["batch_id"] != batch.batch_id:
                self.state.setdefault("gas_prior_predictions", {}).setdefault(sid, []).append(previous)
            self.state["predictions"][sid] = {"batch_id": batch.batch_id, "model_id": selected_id,
                                              "prediction": prediction, "evidence_kind": self.config.execution_mode,
                                              "group":source.group,"input_values":list(source.values),
                                              "sample_revision":source.revision,"ordinal":batch.ordinal,"unit_id":source.unit_id}
            if sid in repairs:
                self.state["predictions"][sid]["gas_repair"] = repairs[sid]
        from harness.research.methods.diagnosis.problem_tracking import record_problem_adoption
        record_problem_adoption(self,batch,decision,selected_id)
        if selected_id != current.model_id:
            archive = [m for m in self.state["archive"] if m not in {current.model_id, selected_id}]
            archive.append(current.model_id)
            self.state["archive"] = archive[-self.config.archive_cap:] if self.config.archive_cap else []
        self.state["current"] = selected_id
        result = {"input_digest": fingerprint, "received": self.state["receipts"][batch.batch_id],
                  "trials": trials, "decision": decision, "model_id": selected_id, "predictions": outputs}
        # 预测输出、当前版本和批次进度在同一状态快照中提交。
        self.state["completed"][batch.batch_id] = result
        self.state["feedback"][f"adoption:{batch.batch_id}"] = {"source": "internal_adoption", **decision}
        self.state.update(pending=None, last_ordinal=batch.ordinal)
        self._event(f"{batch.batch_id}:committed", "batch_committed", batch_id=batch.batch_id, model_id=selected_id)
        return result

    @property
    def run_identity(self):
        """执行与记录消费者共用的恢复身份；比较语义变更需启用新运行目录。"""
        return digest([asdict(self.task), asdict(self.config), self.tools.identity, self.planner.identity,
                       FACTS_VERSION, check_catalog(), guidance_identity(), METHOD_MEMORY_VERSION, ACTION_SELECTION_VERSION])

    def run(self, batches: tuple[BatchContext, ...]) -> dict:
        """推进顺序批次并导出记录；相同已完成批次只返回保存结果。

        任务、方案、适配器或已完成输入改变时拒绝混用运行目录。
        普通候选失败保留记录后继续；无有效初始版本、服务输出失败和
        外部步骤状态未确认时抛出异常，当前未提交批次留待相应恢复处理。
        """
        config = asdict(self.config)
        fingerprint = self.run_identity
        with self.store.exclusive():
            self.state = self.store.load() or {
                "schema_version": 4, "evidence_kind": self.config.execution_mode,
                "model_experiment": self.config.execution_mode == "real_model",
                "fingerprint": fingerprint, "config": config, "task": asdict(self.task),
                "samples": {}, "receipts": {}, "models": {}, "current": None, "archive": [],
                "operations": {}, "events": [], "ledger": {}, "predictions": {}, "feedback": {},
                "completed": {}, "decisions": {}, "last_ordinal": -1, "pending": None, "used_adoption_ids": [],
                "planning_evidence": {},
                "methods": {}, "inspections": {}, "inspection_done": [],
            }
            if self.state["fingerprint"] != fingerprint:
                raise ValueError("运行目录属于不同任务、方案或适配器版本")
            for batch in batches:
                self._run_batch(batch)
            self.store.export(self.state)
            return self.state
