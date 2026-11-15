"""在共同的官方样本分片上由两路独立的 Luna 请求驱动真实气体维护。

prepare/advance 处理已到达的回复，watch 自动续跑；只有汇齐四组的
实际服务输出后才打开留出答案、执行末版的跨采集批评价。
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.workflow.public_maintenance import gas_service_windows
from harness.contracts import digest
from harness.data.gas_input import feature_schema, read_gas_batches, REGISTERED_SHA256
from harness.data.official_gas import LABELS, read_official_gas
from harness.evaluation.gas_pilot import partition
from harness.evaluation.metrics import macro_f1
from harness.methods.planning.gas_continuation import FirstWindowPairPlanner
from harness.models.gas_views import append_pair, build_pair_memory, pair_model
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import LocalStore, atomic_json

PACKAGE = ROOT / "data/official/gas/cpu-a-guide-20261110-v1.zip"
SPLIT = ROOT / "engineering/configs/experiments/official-gas-pilot.json"
DEFAULT = ROOT / "results/experiments/luna-comparison/gas"
DRIVEN = ("project", "mature")
ARMS = ("original", "fixed", *DRIVEN)
QUANTILES = (10, 25, 50, 75, 90)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def persist(path, record):
    """请求和已发出的预测只允许幂等重放。"""
    if path.exists():
        if read(path) != record:
            raise ValueError(f"既有记录与当前输入不符：{path}")
    else:
        atomic_json(path, record)


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def predict(model, rows):
    x = np.asarray([r.values for r in rows])
    return np.concatenate([model.predict(x[i:i+128]).argmax(axis=1)+1 for i in range(0, len(x), 128)])


def scores(rows, predictions):
    truth = [r.label for r in rows]
    return {"macro_f1": macro_f1(zip(truth, predictions, strict=True)),
            "accuracy": float(np.mean(np.asarray(truth) == predictions))}


def rules(model):
    patch = model.meta.get("gas_pair_patch")
    if not patch:
        return []
    return patch.get("rules", [{"reported": patch["reported"], "target": patch["target"]}])


def prepare(folder):
    if file_sha(PACKAGE) != PACKAGE_SHA:
        raise ValueError("官方原包身份不符")
    package = read_official_gas(PACKAGE)
    batches, held = partition(package, read(SPLIT))
    batches = gas_service_windows(batches, held)
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / "models" / (sha256(package.checkpoint).hexdigest()+".npz")
    base.parent.mkdir(exist_ok=True)
    if not base.exists():
        with TemporaryDirectory(dir=folder) as tmp:
            staged = Path(tmp)/"base.npz"
            staged.write_bytes(package.checkpoint)
            os.replace(staged, base)
    if file_sha(base) != base.stem:
        raise ValueError("初始权重摘要不符")
    x = np.asarray([r.values for r in package.train])
    y = np.asarray([r.label-1 for r in package.train])
    persist(folder/"protocol.json", {"protocol":"luna-gas-ten-rounds-v1","official_package_sha256":PACKAGE_SHA,
        "split_sha256":file_sha(SPLIT), "original":base.stem, "rounds":len(batches),"train_rows":len(x),
        "driver":"Luna, one independent response per driven arm per round", "budget":{
            "max_candidates_per_round":2,"epochs_per_fit":1,"max_rows_per_candidate":len(x),
            "max_training_row_visits_per_round":2*len(x)},
        "project":"filter or protected shape-blend pair; explore macro F1 includes parent, strict confirmation gate",
        "mature":"Self-Healing ML adapted from H_LLM.fit_model; Luna produces a subset query, "
                 "exploration accuracy ranks candidates in proposal order, winner deploys directly. "
                 "One call, restricted expressions, deterministic fitted winner reused instead of refitting.",
        "mature_source":"https://github.com/pauliusrauba/Self_Healing_ML/tree/a48a8323e4f23c2e5fc1a0c06dc5605b34cc2825",
        "fixed":"FirstWindowPairPlanner ranks up to two repeated mistake directions on first exploration only. Each "
                "round attempt the first not yet adopted, select with project gate.",
        "service":"same-round deployment after permitted archived exploration/confirmation; service labels withheld",
        "stress":"frozen final weights on previously examined public UCI 224 batches 3-8 and 9-10"})
    return package, batches, held, pair_model(Model), x, y, base


def role(batch, name):
    return tuple(row for row in batch.samples if row.role == name)


def make_request(arm, number, model, parent, batch, x, history):
    """请求不持有确认集和服务标签。特征漂移、尾部错误来自当轮探索与原训练集。"""
    rows = role(batch, "explore")
    pred = predict(model, rows)
    y = np.asarray([r.label for r in rows])
    xe = np.asarray([r.values for r in rows])
    errors = Counter((r.label, int(p)) for r, p in zip(rows, pred, strict=True) if r.label != p)
    confusions = [{"target":t,"reported":r,"count":c} for (t,r),c in
                  sorted(errors.items(),key=lambda i:(-i[1],i[0]))]
    quart = np.quantile(x,[.25,.5,.75],axis=0)
    shift = np.divide(np.median(xe,axis=0)-quart[1],quart[2]-quart[0],
                      out=np.zeros(x.shape[1]),where=quart[2]>quart[0])
    features = [0]
    for i in sorted(range(x.shape[1]), key=lambda k:(-abs(shift[k]),k)):
        if all(i//8 != other//8 for other in features):
            features.append(i)
        if len(features) >= 6:
            break
    profiles = []
    schema = feature_schema()
    for i in features:
        low, high = xe[:,i] <= quart[0,i], xe[:,i] >= quart[2,i]
        profiles.append({"feature":i, "name":schema[i]["name"],"median_shift_train_iqr":round(float(shift[i]),3),
            "explore_min_max": [float(xe[:,i].min()),float(xe[:,i].max())],
            "low25_rows_errors":[int(low.sum()),int(np.sum(low&(pred!=y)))],
            "high75_rows_errors":[int(high.sum()),int(np.sum(high&(pred!=y)))]})
    past = [{"round":h["round"], "actions":[{k:c[k] for k in ("action","status","explore")} for c in h["candidates"]],
             "adopted":h["changed"],"nominee_index":h["nominee_index"],"confirmation":h["confirmation"]} for h in history]
    patch = arm != "mature"
    req = {"task":"gas","arm":arm,"round":number,"objective":"maximize macro F1 on subsequent unseen gas measurements",
        "parent_sha256":parent,"class_order_1_based":list(LABELS),"explore_rows":len(rows),
        "explore_score":scores(rows,pred),"explore_confusions":confusions,"active_patch_rules":rules(model),
        "train_rows":len(x),"sensors":profiles,"history":past,"budget":{
            "one_planner_call":True,"max_candidates":2,"max_epochs_per_candidate":1,"max_total_training_rows":2*len(x)},
        "tools":{"row_filter":{"kind":["exclude","include"],"feature":"integer 0..127 (zero based)",
                    "op":["le","ge"],"quantile":list(QUANTILES),
                    "alternative_to_quantile":"threshold: finite numeric value in raw sensor units",
                    "effect":"fit 1 epoch on the rows matching (include) or outside (exclude) the condition. Quantile is from original fit data."},
                 "keep":"return empty actions []"},
        "selection":"project ranks incumbent and proposals on exploration macro F1 (tie incumbent), then requires strictly better independent confirmation macro F1"
                    if patch else "Self-Healing ML: diagnose shift, propose up to two subset queries. Fitted candidate with best exploration accuracy is deployed; ties follow proposal order. Empty actions retain parent.",
        "response_schema":{"task":"gas","arm":arm,"round":number,"observation_key":"copy","reason":"brief hypothesis",
                           "actions":"list of 0..2 distinct candidates, each its own object"}}
    if patch:
        req["tools"]["pair_patch"]={"kind":"patch","reported":"1-based wrong model prediction",
            "target":"1-based true class of that observed error","effect":"train protected shape-blend correction lookup; max 2 active directions; cannot duplicate a direction"}
    req["observation_key"] = digest(req)
    return req


def exchange(folder, side, arm, number):
    return folder/side/"gas"/arm/f"{number:02d}.json"


def initial_requests(folder, env):
    _, batches, _, model_type, x, _, base = env
    for arm in DRIVEN:
        request = make_request(arm,1,model_type.load(base),base.stem,batches[0],x,[])
        persist(exchange(folder,"requests",arm,1),request)


def store_model(folder, model):
    """先在同一文件系统写完整 NPZ，按内容寻址；运行中断后可以重用。"""
    with TemporaryDirectory(dir=folder) as tmp:
        staged = Path(tmp)/"model.npz"
        model.save(staged)
        name = file_sha(staged)
        dest = folder/"models"/(name+".npz")
        if dest.exists():
            if file_sha(dest) != name:
                raise ValueError("既有候选权重已损坏")
        else:
            os.replace(staged,dest)
    return name


def candidate(folder, parent, proposed, req, env, seen, rows):
    package, _, _, model_type, x, y, _ = env
    start = time.process_time()
    cost = {"fit_rows":0,"reference_rows":0,"steps":0,"epochs":0}
    rec = {"action":proposed,"status":"invalid","explore":None,"cost":cost}
    model = model_type.load(parent)
    try:
        if not isinstance(proposed,dict):
            raise ValueError("动作须为对象")
        kind = proposed.get("kind")
        if kind == "patch":
            if req["arm"] == "mature" or set(proposed) != {"kind","reported","target"}:
                raise ValueError("此组未开放该修正或字段有误")
            pair = (proposed["reported"], proposed["target"])
            if any(type(i) is not int for i in pair) or not all(1 <= i <= 6 for i in pair) or pair[0] == pair[1]:
                raise ValueError("需给出不同的有效类别")
            if req["arm"] != "fixed" and not any((e["reported"],e["target"]) == pair for e in req["explore_confusions"]):
                raise ValueError("探索反馈未出现该方向")
            if len(rules(model)) >= 2 or any((r["reported"],r["target"])==pair for r in rules(model)):
                raise ValueError("重复或超出两个有效修正规则")
            signature = ("patch",*pair)
            if signature in seen:
                raise ValueError("同轮重复动作")
            seen.add(signature)
            spec = {"method":"shape-blend","reported":pair[0],"target":pair[1]}
            if "gas_pair_patch" in model.meta:
                model.meta["gas_pair_patch"] = append_pair(model.meta["gas_pair_patch"],spec,"protected")
            else:
                cost["reference_rows"] = len(x)
                model.meta["gas_pair_patch"] = build_pair_memory(spec,x,y,model.mean,model.scale)
        elif kind in ("include","exclude"):
            keys = set(proposed)
            if keys not in ({"kind","feature","op","quantile"},{"kind","feature","op","threshold"}):
                raise ValueError("筛选字段不符合请求")
            i, op = proposed["feature"], proposed["op"]
            if type(i) is not int or not 0 <= i < x.shape[1] or op not in ("le","ge"):
                raise ValueError("特征号或比较操作无效")
            if "quantile" in proposed:
                q = proposed["quantile"]
                if type(q) is not int or q not in QUANTILES:
                    raise ValueError("训练分位数未登记")
                threshold = float(np.quantile(x[:,i],q/100))
            else:
                threshold = proposed["threshold"]
                if type(threshold) not in (float,int) or not math.isfinite(threshold):
                    raise ValueError("需要有限阈值")
            mask = x[:,i] <= threshold if op == "le" else x[:,i] >= threshold
            keep = mask if kind == "include" else ~mask
            n = int(keep.sum())
            if n == 0 or n == len(x):
                raise ValueError("筛选没有得到非空且改变的训练子集")
            signature = ("fit",digest([r.sample_id for r,k in zip(package.train,keep,strict=True) if k]))
            if signature in seen:
                raise ValueError("同轮两个条件选中相同训练行")
            seen.add(signature)
            rec.update(fit_ids_digest=signature[1],effective_threshold=threshold,
                class_rows=np.bincount(y[keep],minlength=6).tolist())
            cost.update(fit_rows=n,epochs=1)
            steps = model.meta["steps"]
            model.fit(x[keep],y[keep],1)
            cost["steps"] = model.meta["steps"]-steps
        else:
            raise ValueError("未登记动作")
        rec["model"] = store_model(folder,model)
        rec["explore"] = scores(rows,predict(model,rows))
        rec["status"] = "ok"
    except (ValueError,TypeError,FloatingPointError) as exc:
        rec["error"] = str(exc)
        if cost["fit_rows"] or cost["reference_rows"]:
            rec["status"] = "failed"
    cost["cpu_seconds"] = time.process_time()-start
    return rec


def read_response(path, req):
    reply = read(path)
    if not isinstance(reply,dict) or reply.get("observation_key") != req["observation_key"]:
        raise ValueError(f"回复的观察身份不匹配：{path}")
    if any(k in reply and reply[k] != req[k] for k in ("arm","round","task")):
        raise ValueError(f"回复的组、任务或轮次错误：{path}")
    reason, actions = reply.get("reason"), reply.get("actions")
    if not isinstance(reason,str) or not reason.strip() or not isinstance(actions,list) or len(actions)>2:
        raise ValueError(f"回复缺少理由或候选超过两项：{path}")
    return actions,reason


def run_round(folder, arm, number, env, history):
    """每组只恢复自己的父权重和已发布历史；一轮内所有候选从相同父版出发。"""
    _, batches, _, model_type, _, _, base = env
    receipt = folder/"rounds"/arm/f"{number:02d}.json"
    parent_sha = history[-1]["served"] if history else base.stem
    parent = folder/"models"/(parent_sha+".npz")
    if receipt.exists():
        record = read(receipt)
        if record["parent"] != parent_sha or record["round"] != number or record["arm"] != arm:
            raise ValueError("恢复链与原决策不一致")
        if arm in DRIVEN and file_sha(exchange(folder,"responses",arm,number)) != record["response_sha256"]:
            raise ValueError("已消费的驱动回复被改写")
        return record
    if file_sha(parent) != parent_sha:
        raise ValueError("在用权重发生变动")
    model = model_type.load(parent)
    batch = batches[number-1]
    explore, confirm, service = [role(batch,r) for r in ("explore","adopt","serve")]
    req = make_request(arm,number,model,parent_sha,batch,env[4],history)
    fixed_ranking, response_sha = [],None
    if arm in DRIVEN:
        persist(exchange(folder,"requests",arm,number),req)
        response = exchange(folder,"responses",arm,number)
        if not response.is_file():
            return None
        actions, reason = read_response(response,req)
        response_sha = file_sha(response)
    elif arm == "fixed":
        fixed_ranking = history[0]["fixed_ranking"] if history else FirstWindowPairPlanner._rank(
            {"past_decisions":[],"gas_confusions":req["explore_confusions"]})
        active = {(r["reported"],r["target"]) for r in rules(model)}
        todo = [(r,t) for r,t in fixed_ranking if (r,t) not in active]
        actions = [{"kind":"patch","reported":todo[0][0],"target":todo[0][1]}] if todo else []
        reason = "首轮冻结的错误方向依次试改"
    else:
        actions, reason = [],"沿用官方起点"
    started = time.process_time()
    candidates, seen = [],set()
    for action in actions:
        candidates.append(candidate(folder,parent,action,req,env,seen,explore))
    eligible = [i for i, c in enumerate(candidates) if c["status"]=="ok"]
    nominee, served, confirmation = None, parent_sha, None
    if eligible:
        metric = "accuracy" if arm == "mature" else "macro_f1"
        # 作者 np.argmax 保留首个同分提案，项目实现同分先选父版、其余按稳定身份。
        if arm == "mature":
            nominee = max(eligible,key=lambda i:candidates[i]["explore"][metric])
            served = candidates[nominee]["model"]
        else:
            best = min(eligible,key=lambda i:(-candidates[i]["explore"][metric],candidates[i]["model"]))
            if candidates[best]["explore"][metric] > req["explore_score"][metric]:
                nominee = best
                next_model = model_type.load(folder/"models"/(candidates[best]["model"]+".npz"))
                before = scores(confirm,predict(model,confirm))[metric]
                after = scores(confirm,predict(next_model,confirm))[metric]
                confirmation = {"parent":before,"candidate":after}
                if after > before+1e-12:
                    served = candidates[best]["model"]
    deploy = model_type.load(folder/"models"/(served+".npz")) if served != parent_sha else model
    outputs = {r.sample_id:int(p) for r,p in zip(service,predict(deploy,service),strict=True)}
    record = {"arm":arm,"round":number,"parent":parent_sha,"served":served,"changed":served!=parent_sha,
        "fixed_ranking":fixed_ranking,"observation_key":req["observation_key"],"response_sha256":response_sha,
        "reason":reason,"candidates":candidates,"nominee_index":nominee,"confirmation":confirmation,
        "explore_parent":req["explore_score"],"served_predictions":outputs,"cpu_seconds":time.process_time()-started}
    atomic_json(receipt,record)
    return record


def report_by_arm(rows, values):
    """四组必须在同一组带身份的预测上配对核验。"""
    truth = np.asarray([r.label for r in rows])
    base = np.asarray(values["original"])
    return {arm:{**scores(rows,p),"correct":int(np.sum(p==truth)),
        "repaired":int(np.sum((base!=truth)&(p==truth))),"broken":int(np.sum((base==truth)&(p!=truth)))}
        for arm,p in values.items()}


def final_score(folder, env, histories):
    _, batches, held, model_type, _, _, _ = env
    order = [r.sample_id for b in batches for r in role(b,"serve")]
    truth_by_id = {r.sample_id:r for r in held}
    if len(set(order))!=len(order) or set(order)!=set(truth_by_id):
        raise ValueError("共同服务分片有缺失或重复")
    ordered = [truth_by_id[i] for i in order]
    issued, models, costs = {},{},{}
    for arm, records in histories.items():
        if any(list(record["served_predictions"]) != [r.sample_id for r in role(batch,"serve")]
               for record,batch in zip(records,batches,strict=True)):
            raise ValueError("已发出的预测与固定分片不匹配")
        issued[arm] = np.asarray([p for r in records for p in r["served_predictions"].values()])
        models[arm] = model_type.load(folder/"models"/(records[-1]["served"]+".npz"))
        trials = [c for r in records for c in r["candidates"]]
        costs[arm] = {"driver_calls":10 if arm in DRIVEN else 0,"attempts":len(trials),
            "successful":sum(c["status"]=="ok" for c in trials),
            "failed":sum(c["status"]=="failed" for c in trials),"invalid":sum(c["status"]=="invalid" for c in trials),
            "adoptions":sum(r["changed"] for r in records),"cpu_seconds_after_proposal":sum(r["cpu_seconds"] for r in records),
            **{name:sum(c["cost"][name] for c in trials) for name in ("fit_rows","reference_rows","steps","epochs")}}
    service = report_by_arm(ordered,issued)
    stress, pressure_rows = {},[]
    public = ROOT/"data/public/gas-drift/gas-drift.zip"
    for source in read_gas_batches(public,expected_sha256=REGISTERED_SHA256):
        if source.batch_number < 3:
            continue
        rows = source.records
        preds = {arm:predict(model,rows) for arm,model in models.items()}
        # 冻结四个末版；每个来源行留下可独立复算的预测，压力批不回流给任何驱动。
        pressure_rows.extend({"batch":source.batch_number,"line":r.source_line,
            **{arm:int(values[i]) for arm,values in preds.items()}} for i,r in enumerate(rows))
        stress[source.batch_number] = {"rows":len(rows),"arms":report_by_arm(rows,preds)}
    with TemporaryDirectory(dir=folder) as tmp:
        path = Path(tmp)/"pressure.jsonl"
        with path.open("w",encoding="utf8",newline="\n") as stream:
            for row in pressure_rows:
                stream.write(json.dumps(row,separators=(",",":"))+"\n")
        pressure_sha = file_sha(path)
        os.replace(path,folder/"pressure-predictions.jsonl")
    means = {name:{arm:float(np.mean([stress[b]["arms"][arm]["macro_f1"] for b in numbers]))
                    for arm in ARMS} for name,numbers in {"batches_3_8":range(3,9),"batches_9_10":range(9,11)}.items()}
    final = {"task":"gas","service_rows":len(order),"official_replay_pooled":service,
        "service_metric":"pooled Macro-F1 on 414 held-out measurements from official validation source batch 2",
        "final_model_sha256":{a:rows[-1]["served"] for a,rows in histories.items()},"cost":costs,
        "stress_mean_batch_macro_f1":means,"stress_by_source_batch":stress,
        "pressure_input_sha256":REGISTERED_SHA256,"pressure_predictions_sha256":pressure_sha,
        "cost_units":"fit_rows are rows used for an Adam epoch; reference_rows are rows read once to construct patch memory",
        "drive_cost":"one local Luna response per request; API tokens and cloud GPU cost not measured"}
    atomic_json(folder/"summary.json",final)
    print(json.dumps({"service":service,"pressure":means},ensure_ascii=False),flush=True)


def run(folder, env, operation, poll_seconds):
    histories = {arm:[] for arm in ARMS}
    while True:
        complete = True
        for arm, history in histories.items():
            while len(history) < len(env[1]):
                number = len(history)+1
                record = run_round(folder,arm,number,env,history)
                if record is None:
                    complete = False
                    break
                history.append(record)
                print(f"{arm} round {number}: {record['served'][:12]}",flush=True)
        if complete:
            final_score(folder,env,histories)
            return
        if operation == "score":
            raise ValueError("评分需要所有已发出的服务预测")
        if operation != "watch":
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=["prepare","advance","watch","score"])
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--poll-seconds",type=float,default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    with LocalStore(output).exclusive():
        environment = prepare(output)
        initial_requests(output,environment)
        if args.operation != "prepare":
            run(output,environment,args.operation,args.poll_seconds)
