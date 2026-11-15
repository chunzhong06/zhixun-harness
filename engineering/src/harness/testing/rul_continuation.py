"""用两项官方寿命任务的真实输入和预写预测演练版本接续。"""
from collections import defaultdict
from dataclasses import replace
import json

from harness.contracts import BatchContext, EngineConfig, digest
from harness.evaluation.metrics import error_profile, prediction_score
from harness.methods.planning.rul_continuation import RulContinuationPlanner, overestimated_group
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.engine import Harness
from harness.runtime.storage import LocalStore, atomic_json


def representative_units(rows):
    """发动机演练在每个子集中按设备号取两台。真实评价使用全部公开验证设备。"""
    by_subset = defaultdict(set)
    for row in rows:
        by_subset[row.unit_id.split(":", 1)[0]].add(row.unit_id)
    return {unit for units in by_subset.values() for unit in sorted(units)[:2]}


def inputs(package):
    is_engine = package.task.target_unit == "cycles"
    train_units = representative_units(package.train) if is_engine else {r.unit_id for r in package.train}
    units = representative_units(package.validation) if is_engine else {r.unit_id for r in package.validation}
    by_unit = defaultdict(list)
    for row in package.train:
        if row.unit_id in train_units:
            by_unit[row.unit_id].append(row)
    # 构建样例只取各训练设备首、中、末三条，验证设备的答案仅由演练编制和评价使用。
    fit = []
    for members in by_unit.values():
        ordered = sorted(members,key=lambda r:r.time_start)
        fit.extend(ordered[i] for i in sorted({0,len(ordered)//2,len(ordered)-1}))
    fit = tuple(fit)
    stages = ("initial","window-1","window-2","window-3")
    predictions = {s:{} for s in stages}
    observed = [r for r in package.validation if r.unit_id in units and r.time_start <= 36]
    if len(observed) != 36 * len(units):
        raise ValueError("各演练设备需要连续的前36个观测")
    group = observed[0].group if is_engine else "Bearing2_4"
    if group not in {r.group for r in observed}:
        raise ValueError("指定的举例范围缺少观测")
    held = []
    batches = []
    for ordinal in range(3):
        samples = list(held) if ordinal else list(fit)
        held = []
        for row in observed:
            offset = row.time_start - 1 - 12*ordinal
            if not 0 <= offset < 12:
                continue
            role = "explore" if offset < 4 else "adopt" if offset < 8 or ordinal == 2 else "serve"
            now = replace(row,role=role,available_at=ordinal)
            if role=="serve":
                samples.append(replace(now,label=None))
                held.append(replace(now,role="explore",revision=2,available_at=ordinal+1))
            else:
                samples.append(now)
            # 这是验收用的固定答案表。误差在编制样例时注入，计划器、预测接口不接触答案表中的真值。
            errors = (24, 6 if role=="explore" else 36, 4, 2 if role=="explore" else 9)
            for stage, err in zip(stages,errors):
                predictions[stage][row.sample_id] = row.label + (err if row.group == group else 0)
        batches.append(BatchContext(package.task.task_id,stages[ordinal+1],ordinal,tuple(samples)))
    return fit,tuple(batches),predictions


def device_stream(package, fit, assignment):
    """用已冻结的整台划分演练十次交接。每个角色在每个子集取一台，服务答案留在评价侧。"""
    representatives = {}
    for uid, role in sorted(assignment.items()):
        subset = uid.split(":", 1)[0] if package.task.target_unit == "cycles" else "bearing"
        representatives.setdefault((subset, role), uid)
    units = set(representatives.values())
    stages = ["initial"] + [f"round-{i}" for i in range(1, 11)]
    tables = {name: {} for name in stages}
    windows = [[] for _ in stages[1:]]
    for row in package.validation:
        if row.unit_id not in units or row.time_end > 40:
            continue
        ordinal = (row.time_end - 1) // 4
        role = assignment[row.unit_id]
        windows[ordinal].append(replace(row, role=role, available_at=ordinal,
                                        label=None if role == "serve" else row.label))
        # 故意给首个候选设置“探索改善、确认变差”，第二个候选两处改善。
        # 后续仍有试验机会，系统应依据确认结果保留已采用版本。
        for i, stage in enumerate(stages):
            error = 24 if i == 0 else (5 if role == "explore" else 32) if i == 1 else 4 if i == 2 else 8
            tables[stage][row.sample_id] = row.label + error
    if any(len(window) != 4 * len(units) for window in windows):
        raise ValueError("冻结的每台演练设备均需提供前40次连续观测")
    batches = tuple(BatchContext(package.task.task_id, stages[i+1], i,
                                 (tuple(fit) if i == 0 else ()) + tuple(window))
                    for i, window in enumerate(windows))
    return batches, tables


class RulReplayTools:
    evidence_kind = "interface_fixture"
    capabilities = {"fixed_public_training":True,"continuation_epochs":[1],"group_error_direction":True}

    def __init__(self, package, fit, tables, folder, *, check_devices=False):
        self.package, self.tables = package,tables
        if check_devices:
            self.capabilities = {**self.capabilities, "device_bias": True}
        self.fit = {r.sample_id:r for r in fit}
        self.values = {r.sample_id:r.values for r in package.validation}
        self.artifacts = ModelArtifacts(folder)
        identity = "bearing-replay-v1" if package.task.target_unit == "minutes" else "engine-replay-v1"
        self.identity = digest([identity,package.train_sha,package.validation_sha,
                                package.checkpoint_sha,sorted(self.fit),tables,self.capabilities])

    def build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        if task!=self.package.task or len(rows)!=len(self.fit) or {r.sample_id:r for r in rows}!=self.fit:
            raise ValueError("构建只允许使用已声明的官方训练快照")
        request = {"stage":"initial" if initialize else batch_id,"source":self.package.checkpoint_sha,
                   "evidence_kind":self.evidence_kind,"fit_ids":sorted(self.fit)}
        if initialize:
            return self.artifacts.create(request,lambda:self.package.checkpoint)
        if action.epochs!=1 or not parent:
            raise ValueError("样例续训必须绑定一个已存在的父版本")
        manifest,_ = self.artifacts.read(parent)
        if manifest["request"]["source"]!=request["source"]:
            raise ValueError("父版本的初始检查点不匹配")
        request.update(parent=parent,parent_stage=manifest["request"]["stage"],epochs=action.epochs)
        return self.artifacts.create(request,lambda:json.dumps(request,sort_keys=True).encode())

    def recover_build(self, *args, **kwargs):
        return {"status":"completed","artifact":self.build(*args,**kwargs)}

    def predict(self, artifact, inputs):
        manifest,_ = self.artifacts.read(artifact)
        table = self.tables[manifest["request"]["stage"]]
        result = {}
        for row in inputs:
            sid = row["sample_id"]
            if set(row)!={"sample_id","values"} or tuple(row["values"])!=self.values[sid]:
                raise ValueError("预测接口只接收原始快照身份和测量特征")
            result[sid] = table[sid]
        return result


def trace(package, batches, state, tools):
    rounds = []
    for batch in batches:
        bid = batch.batch_id
        if bid not in state["completed"]:
            continue
        done = state["completed"][bid]
        decision = done["decision"]
        current, selected = decision["current_model"],decision["selected_model"]
        saved = next(e for e in state["events"] if e["key"]==f"{bid}:plan_context:0")
        issue = overestimated_group(saved["context"])
        builds = []
        for attempt in done["trials"]:
            mid = attempt.get("model_id")
            if mid is None or attempt["action"]=="reuse":
                continue
            bundle = state["models"][mid]
            manifest,_ = tools.artifacts.read(bundle["artifact"])
            request = manifest["request"]
            if request["parent"] != state["models"][current]["artifact"]:
                raise ValueError("候选未从本轮在用产物出发")
            op = state["operations"][f"{bid}:build:{attempt['action']}"]
            builds.append({"candidate":mid,"parent":current,"parent_stage":request["parent_stage"],
                           "epochs":request["epochs"],"reason":op["requested_action"]["reason"],
                           "current_explore_rmse":state["operations"][f"{bid}:explore:{current}"]["value"]["score"],
                           "candidate_explore_rmse":state["operations"][f"{bid}:explore:{mid}"]["value"]["score"]})
        scores = []
        for mid in dict.fromkeys((current,decision["nominee"])):
            op = state["operations"].get(f"{bid}:adopt:{mid}",{})
            if not op.get("value"):
                continue
            receipt = op["value"]
            by_unit = defaultdict(list)
            for row in batch.samples:
                if row.role=="adopt":
                    by_unit[row.unit_id].append(row)
            units = {}
            for unit,rows in by_unit.items():
                pred = {r.sample_id:receipt["predictions"][r.sample_id] for r in rows}
                profile = error_profile(package.task,rows,pred)
                units[unit] = {"rows":len(rows),"rmse":prediction_score(package.task,rows,pred),
                               "mean_signed_error":profile["mean_signed_error"],"over":profile["overprediction_count"]}
            scores.append({"model":mid,"rmse":receipt["score"],"units":units})
        returns = []
        for row in batch.samples:
            if row.revision!=2:
                continue
            f = state["feedback"][f"service:{row.sample_id}:2"]
            issued = state["predictions"][row.sample_id]
            identity = {"bearing":row.unit_id} if package.task.target_unit == "minutes" else {
                "engine":row.unit_id, "condition":row.group, "cycle":row.time_start}
            returns.append({"id":row.sample_id,**identity,"from":issued["batch_id"],
                            "model":f["model_id"],"prediction":f["prediction"],"answer":f["label"],
                            "received_in":f["received_in"],"signed_error":f["prediction"]-f["label"]})
        rounds.append({"batch":bid,"incumbent":current,"issue":issue[1] if issue else None,"builds":builds,
                       "confirmation":scores,"selected":selected,"reason":decision["reason"],
                       "service_model":done["model_id"] if done["predictions"] else None,
                       "service_rows":len(done["predictions"]),"feedback_returned":returns})
    return {"evidence_kind":"interface_fixture","model_experiment":False,"unit":package.task.target_unit,
            "package_id":package.package_id,"checkpoint_sha256":package.checkpoint_sha,
            "official":{"training_rows":len(package.train),"validation_rows":len(package.validation),
                        "training_sha256":package.train_sha,"validation_sha256":package.validation_sha},
            "fit_fixture_rows":len(tools.fit),"validation_window":[1,36],"rounds":rounds}


def run_rehearsal(package, output, *, limit=None):
    fit,batches,tables = inputs(package)
    tools = RulReplayTools(package,fit,tables,output/"models")
    config = EngineConfig(policy="plan_feedback",recent_cap=len(fit),max_trials=1)
    engine = Harness(package.task,config,tools,RulContinuationPlanner(),LocalStore(output))
    state = engine.run(batches if limit is None else batches[:limit])
    report = trace(package,batches,state,tools)
    task_name = "bearing" if package.task.target_unit == "minutes" else "engine"
    atomic_json(output/f"official-{task_name}-trace.json",report)
    return report
