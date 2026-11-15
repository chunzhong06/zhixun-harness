"""显式接口替身：返回事先写好的计划及预测，完全不拟合模型。"""
from __future__ import annotations

from harness.contracts import ActionPlan, Sample, TaskSpec, digest
from harness.models.pipeline_recipe import PipelineRecipe
from harness.integrations.pytrio_planner import ChatRequest, ChatResponse
from harness.runtime.storage import atomic_json
from dataclasses import asdict
from copy import deepcopy
from io import BytesIO
import json


class FixtureModelTools:
    evidence_kind = "interface_fixture"

    def __init__(self, fixture: dict):
        self.fixture = fixture
        self.identity = "fixture-model-tools-v1:" + digest(fixture)
        self.build_calls = []
        self.predict_calls = []

    def build(self, task: TaskSpec, batch_id: str, action: ActionPlan,
              rows: tuple[Sample, ...], *, initialize=False) -> dict:
        self.build_calls.append({"batch_id": batch_id, "action": action.key,
                                 "sample_ids": [s.sample_id for s in rows], "initialize": initialize})
        if any(s.role != "fit" for s in rows):
            raise ValueError("训练替身收到了非拟合资料")
        if not initialize and action.recipe in self.fixture.get("fail_build", {}).get(batch_id, []):
            raise RuntimeError("预设的样例训练接口失败")
        name = "base" if initialize else self.fixture["build_outputs"][batch_id][action.recipe]
        if action.pipeline and not initialize:
            name = self.fixture.get("component_build_outputs", {}).get(batch_id, {}).get(action.pipeline.key, name)
        return {"fixture_model": name, "evidence_kind": self.evidence_kind,
                "description": "预写预测表，无参数拟合",
                "pipeline": asdict(action.pipeline) if action.pipeline else None}

    def predict(self, artifact: dict, inputs: tuple[dict, ...]) -> dict[str, int | float]:
        if any(set(row) != {"sample_id", "values"} for row in inputs):
            raise ValueError("预测接口只能获得身份与特征")
        self.predict_calls.append({"artifact": artifact["fixture_model"], "inputs": inputs})
        table = self.fixture["predictions"][artifact["fixture_model"]]
        return {row["sample_id"]: table[row["sample_id"]] for row in inputs}


class FixturePlanner:
    identity = "fixture-planner-v5"

    def __init__(self):
        self.contexts = []

    def propose(self, context: dict) -> dict:
        self.contexts.append(context)
        if "maintenance" in context:
            if context["maintenance"]["stage"] == "decision":
                return self._decision_plan(context)
            return self._maintenance_plan(context)
        recipes = [
            {"recipe": "recent", "ratio": 0, "reason": "样例计划：先检查近期训练"},
            {"recipe": "uniform", "ratio": 0.5, "reason": "样例计划：比较有限普通回放"},
            {"recipe": "coverage", "ratio": 1, "reason": "样例计划：比较分组历史覆盖"},
        ]
        evidence = {item["kind"]: item for item in context["facts"].get("evidence", [])}
        if evidence:
            current = evidence["performance"]
            pools = evidence["input_change"]
            attempts = evidence["attempts"]
            # 固定选择顺序保留；引用实际计算的样例事实验证理由与来源的衔接。
            recipes[0].update(reason=f"当前探索{current['value']['metric']}为{current['value']['score']}；样例计划先比较近期重训",
                              evidence_refs=[current["id"], attempts["id"]])
            recipes[1].update(reason=f"当前可用历史资料{context['facts']['history_count']}条；样例计划加入近期数量一半的历史作比较",
                              evidence_refs=[pools["id"], attempts["id"]])
            recipes[2].update(reason=f"历史资料覆盖{pools['value']['history']['groups']['total_values']}个已声明组；样例计划比较覆盖回放",
                              evidence_refs=[pools["id"], attempts["id"]])
        if context["policy"] == "plan_once":
            return {"actions": recipes, "stop": False}
        # 理想值停止只用于工程分支；实际策略在正式任务中另行比较。
        ideal = 1 if context["objective"]["direction"] == "maximize" else 0
        if context["best_exploration_score"] == ideal:
            return {"actions": [], "stop": True}
        tried = {t["action"].split("-")[0] for t in context["trials"] if t["action"] != "reuse"}
        for recipe in (recipes[0], recipes[2], recipes[1]):
            if recipe["recipe"] not in tried:
                return {"actions": [recipe], "stop": False}
        return {"actions": [], "stop": True}

    def _decision_plan(self, context):
        """固定工程轨迹：试一次、按可用工具补查、按检查选择组件，再结束。"""
        facts = {item["kind"]: item for item in context["facts"]["evidence"]}
        def decision(kind, reason, **fields):
            refs=[facts["attempts"]["id"]]
            if context["maintenance"].get("inspection_followups"):
                refs.append(facts["investigations"]["id"])
            return {"operation": kind, "reason": reason, "evidence_refs": refs, **fields}
        tried = [t for t in context["trials"] if t["action"] != "reuse"]
        ideal = 1 if context["objective"]["direction"] == "maximize" else 0
        if context["best_exploration_score"] == ideal:
            return decision("finish", "工程样例达到预设理想值，结束本批尝试")
        if not tried:
            action = ActionPlan("recent", reason="工程样例：先观察近期重训的结果",
                                evidence_refs=(facts["performance"]["id"],))
            return decision("experiment", action.reason, actions=[action.to_record()])
        if context["maintenance"]["available_checks"] and not facts["investigations"]["value"]["results"]:
            check_context = deepcopy(context)
            check_context["maintenance"]["stage"] = "inspection"
            checks = self._maintenance_plan(check_context)["checks"][:1]
            return decision("inspect", "工程样例：已有试改结果，补查是否应调整数据处理", checks=checks,
                            model_id=context["maintenance"]["decision"]["inspectable_models"][0])
        plan = self._maintenance_plan(context)
        if plan["stop"]:
            return decision("finish", "工程样例可用的不同改法已经比较完毕")
        return decision("experiment", "工程样例：依据当前检查和历史安排下一项组件改法", actions=plan["actions"])

    def _maintenance_plan(self, context):
        """预写工程行为：检查结果改变组件配方；不模拟大模型的实际推理质量。"""
        evidence = {item["kind"]: item for item in context["facts"]["evidence"]}
        ref = evidence["input_change"]["id"]
        def action(pipeline=None, refs=(ref,)):
            return asdict(ActionPlan("recent", reason="工程样例：用共同资料检查组件改动", evidence_refs=refs,
                                     pipeline=pipeline))
        if context["maintenance"]["stage"] == "inspection":
            choices = [
                {"tool": "feature_scale", "question": "特征跨度是否提示先检查缩放处理", "threshold": 10.,
                 "if_true": action(PipelineRecipe(processor="standard")),
                 "if_false": action(PipelineRecipe(processor="none"))},
                {"tool": "error_slices", "question": "误差是否集中于某个输入范围，值得比较另一模型族", "threshold": .2,
                 "if_true": action(PipelineRecipe(predictor="forest", trees=32)),
                 "if_false": action(PipelineRecipe(regularization=2.))},
            ]
            choices = [c for c in choices if c["tool"] in context["maintenance"]["available_checks"]]
            return {"checks": choices[:context["maintenance"]["max_checks"]], "stop": False}
        choices = [action()]
        for method in context["maintenance"]["methods"]:
            if method["eligible"]:
                entry = deepcopy(method["action"])
                entry.update(method_id=method["id"], evidence_refs=[evidence["methods"]["id"]],
                             reason="工程样例：对当前范围已有支持的配方重新进行共同评价")
                choices.append(entry)
        for result in evidence["investigations"]["value"]["results"]:
            if result["signal"] is None:
                continue
            selected = deepcopy(result["alternatives"]["true" if result["signal"] else "false"])
            selected.update(evidence_refs=[evidence["investigations"]["id"]],
                            reason="工程样例：根据已完成检查选择组件试改")
            choices.append(selected)
        if len(choices) == 1:
            choices.append(action(PipelineRecipe(regularization=2.)))
        unique = {}
        for choice in choices:
            unique.setdefault(ActionPlan.parse(choice).key, choice)
        tried = {t["action"] for t in context["trials"]}
        remaining = [c for key, c in unique.items() if key not in tried]
        if context["policy"] != "plan_once":
            remaining = remaining[:1]
        return {"actions": remaining[:3], "stop": not remaining}


class FixtureTrainingBackend:
    """把预写预测表封装为完整可保存对象，供真实训练适配链的离线检查。

    fit仅选择已有样例，处理器为恒等占位，不估计参数；load后预测只读
    文件中的表，不依赖上次进程内的对象。证据类型保持interface_fixture。
    """
    evidence_kind = "interface_fixture"

    def __init__(self, fixture: dict):
        self.tools = FixtureModelTools(fixture)
        self.load_calls = 0

    @property
    def specification(self):
        return {"implementation": "fixture-pipeline-v1", "fixture_digest": digest(self.tools.fixture)}

    def fit(self, task, batch_id, action, rows, *, initialize=False):
        artifact = self.tools.build(task, batch_id, action, rows, initialize=initialize)
        return {"processor": {"kind": "identity_fixture", "features": list(task.features)},
                "predictor": {"kind": "lookup_fixture", "artifact": artifact,
                              "table": self.tools.fixture["predictions"][artifact["fixture_model"]]}}

    def dump(self, pipeline):
        return json.dumps(pipeline, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def fit_correction(self, task, batch_id, action, rows, base, predicates):
        """保存预写分支和实际残差目标，核查组合与权限；不学习任何参数。"""
        from dataclasses import replace
        predicted = self.predict(base, tuple({"sample_id": s.sample_id, "values": s.values} for s in rows))
        targets = [s.label-predicted[s.sample_id] if action.correction.mode == "residual" else s.label for s in rows]
        patch = self.fit(task, batch_id, replace(action, correction=None), rows)
        return {"composition": {"base": base, "patch": patch, "mode": action.correction.mode,
                                "predicates": deepcopy(predicates), "fit_targets": targets}}

    def load(self, content):
        self.load_calls += 1
        return json.loads(content)

    def predict(self, pipeline, inputs):
        if "composition" in pipeline:
            from harness.models.corrections import combine, inside
            c = pipeline["composition"]
            base = self.predict(c["base"], inputs)
            subset = tuple(r for r in inputs if inside(r["values"], c["predicates"]))
            patch = self.predict(c["patch"], subset)
            return combine(base, patch, inputs, c["mode"], c["predicates"])
        table = pipeline["predictor"]["table"]
        return {row["sample_id"]: table[row["sample_id"]] for row in inputs}


class OfflineChatTransport:
    """按对话响应形状包装原计划替身，验证适配层；完全不建立网络连接。"""
    identity = "offline-chat-fixture-v4"
    network_enabled = False

    def __init__(self, planner=None):
        self.planner = planner if planner is not None else FixturePlanner()
        self.requests = []

    def send(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        context = json.loads(request.body["messages"][1]["content"])
        plan = self.planner.propose(context)
        return ChatResponse(200, {"id": f"offline-{len(self.requests)}",
                                  "model": request.body["model"],
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}}],
                                  "usage": None})


class SimulatedHttpService:
    """接收真正的HTTP请求对象，在内存中返回预写计划；不建立socket。"""
    simulated = True
    identity = "simulated-pytrio-http-v1"

    def __init__(self, planner=None):
        self.transport = OfflineChatTransport(planner)
        if planner is not None:
            self.identity += ":" + planner.identity
        self.requests = []

    def open(self, request, *, timeout):
        if request.get_method() != "POST" or request.get_header("Authorization") != "Bearer fixture-only-key":
            raise ValueError("模拟HTTP请求不符合预期")
        body = json.loads(request.data.decode("utf-8"))
        self.requests.append({"method": request.get_method(), "url": request.full_url,
                              "timeout": timeout, "body": body, "authentication_present": True})
        reply = self.transport.send(ChatRequest(request.full_url, body, timeout))
        # 固定用量只用于测试费用记录；不是分词结果或实际服务收费。
        reply.body["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        response = BytesIO(json.dumps(reply.body, ensure_ascii=False).encode("utf-8"))
        response.status = reply.status
        return response


class SimulatedSwanLab:
    """实现本项目使用的SDK方法形状，保留调用证据；不导入真实SDK。"""
    simulated = True

    def __init__(self, receipt_file=None):
        self.receipt_file = receipt_file
        self.runs = (json.loads(receipt_file.read_text(encoding="utf-8"))["runs"]
                     if receipt_file and receipt_file.exists() else [])

    def _save(self):
        if self.receipt_file:
            atomic_json(self.receipt_file, {"simulated_sdk": True, "cloud_uploaded": False, "runs": self.runs})

    @staticmethod
    def Text(data, caption=None):
        return {"text": data, "caption": caption}

    def init(self, **kwargs):
        if kwargs["mode"] != "offline":
            raise ValueError("记录演练只允许offline")
        saved = {"config": deepcopy(kwargs), "logs": [], "finished": False}
        self.runs.append(saved)
        self._save()
        return _SimulatedRun(self, saved)


class _SimulatedRun:
    def __init__(self, sdk, saved):
        self.sdk, self.saved = sdk, saved

    def log(self, data, step=None):
        self.saved["logs"].append({"step": step, "data": deepcopy(data)})
        self.sdk._save()

    def finish(self, state="success", error=None):
        self.saved.update(finished=True, state=state)
        self.sdk._save()
