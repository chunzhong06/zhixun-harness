"""只读逐条预测和官方答案，交叉核对两项寿命及两次气体对照；不训练或调用驱动。"""
from itertools import combinations
from collections import Counter
from hashlib import sha256
import json
from math import fsum, isclose, sqrt
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.data.official_gas import LABELS
from harness.runtime.storage import atomic_json

OUTPUT = ROOT / "results/experiments/luna-comparison"
PACKAGES = {"gas":"cpu-a-guide-20261110-v1.zip", "bearing":"cpu-b-rul-20261110-v1.zip",
            "engine":"cpu-c-guide-20261110-v1.zip"}


def raw(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate(task, y, p):
    if task != "gas":
        return sqrt(fsum((a-b)**2 for a,b in zip(y,p,strict=True))/len(p))
    actual, predicted = Counter(y), Counter(p)
    hits = Counter(a for a,b in zip(y,p,strict=True) if a == b)
    classes = actual.keys() | predicted.keys()
    return fsum(2*hits[k]/(actual[k]+predicted[k]) for k in classes)/len(classes)


def audit(task):
    gas = task == "gas"
    arms = ["original" if gas else "static", "fixed", "project", "mature"]
    if gas:
        arms.append("mature_format")
    y = {}
    with ZipFile(ROOT/"data/official"/task/PACKAGES[task]) as package, package.open("public/validation.jsonl") as inp:
        for line in inp:
            row = json.loads(line)
            y[row["id"]] = LABELS.index(row["answer"])+1 if gas else row["answer"]
    expected, keys = None, None
    arm_data, by_arm = {}, {}
    for arm in arms:
        arm_id = "mature" if arm == "mature_format" else arm
        branch = OUTPUT/("gas-format" if arm == "mature_format" else "gas" if gas else "rul")
        base = branch/"rounds"/arm_id if gas else branch/"issued"/task/arm_id
        states = [raw(base/f"{n:02d}.json") for n in range(1,11)]
        issued, per_round, per_round_predictions = {}, [], []
        previous = None
        for n,s in enumerate(states,1):
            if s["round"] != n or s["arm"] != arm_id or (previous and s["parent"] != previous):
                raise ValueError(f"{task} {arm} 的父权重或轮次不相符")
            previous = s["served" if gas else "selected"]
            if arm_id in ("mature","project"):
                req, res = [raw(branch/side/task/arm_id/f"{n:02d}.json") for side in ("requests","responses")]
                if req["observation_key"] != res["observation_key"] or req["parent_sha256"] != s["parent"]:
                    raise ValueError("当轮回复未绑定实际父版")
                if not gas and res != s["decision"]:
                    raise ValueError("寿命驱动原始决定有变更")
                if gas and sha256((branch/"responses"/task/arm_id/f"{n:02d}.json").read_bytes()).hexdigest()!=s["response_sha256"]:
                    raise ValueError("气体驱动原始决定有变更")
            preds = s["served_predictions" if gas else "predictions"]
            if not preds or issued.keys() & preds.keys() or not preds.keys() <= y.keys():
                raise ValueError("各轮服务编号重复、遗漏或错配")
            issued.update(preds)
            per_round.append(evaluate(task,[y[k] for k in preds],list(preds.values())))
            per_round_predictions.append(preds)
        timeline = [set(p) for p in per_round_predictions]
        if expected is None:
            expected, keys = timeline, list(issued)
        if timeline != expected:
            raise ValueError("四组收到的服务观测不同")
        value = evaluate(task,[y[k] for k in keys],[issued[k] for k in keys])
        if gas:
            declared = raw(branch/"summary.json")["official_replay_pooled"][arm_id]["macro_f1"]
        else:
            declared = raw(branch/"scores"/(task+".json"))["arms"][arm_id]["rmse"]
        if not isclose(value,declared,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f"{task} {arm} 汇总值和逐条预测不相符")
        arm_data[arm] = {"score":value,"rounds":per_round,"worst_round":(min if gas else max)(per_round)}
        by_arm[arm] = per_round_predictions
    pairings = {}
    for a,b in combinations(arms,2):
        differences = [sum(p[k]!=q[k] for k in p) for p,q in zip(by_arm[a],by_arm[b],strict=True)]
        item = {"changed_predictions":sum(differences),"first_different_round":next((i for i,d in enumerate(differences,1) if d),None)}
        if not gas:
            paired = raw(OUTPUT/"rul"/"scores"/(task+".json"))["per_device"]
            d = [s[a]-s[b] for s in paired.values()]
            item["devices_lower_error_for_first"] = sum(x < -1e-9 for x in d)
            item["devices_lower_error_for_second"] = sum(x > 1e-9 for x in d)
            item["devices_tied"] = sum(abs(x)<=1e-9 for x in d)
        pairings[a+" vs "+b] = item
    return {"rows":len(keys),"per_arm":arm_data,"paired":pairings}


def domain_audit(task):
    """用相同测点确认论文部件与等量抽样对照；独立复算其发布后的预测。"""
    if task == "gas":
        predictions = raw(OUTPUT/"gas-patching/issued.json")
        controls = {}
        for n in range(1,11):
            controls.update(raw(OUTPUT/"gas/rounds/original"/f"{n:02d}.json")["served_predictions"])
        score = raw(OUTPUT/"gas-patching/scores.json")
        if predictions.keys() != controls.keys() or sum(predictions[k]!=p for k,p in controls.items()) != score["different_from_static"]:
            raise ValueError("作者修补器与其他组的服务身份或差异数不符")
        with ZipFile(ROOT/"data/official"/task/PACKAGES[task]) as z:
            src={r["id"]:LABELS.index(r["answer"])+1 for r in map(json.loads,z.read("public/validation.jsonl").splitlines())}
        value=evaluate(task,[src[k] for k in predictions],list(predictions.values()))
        if not isclose(score["macro_f1"],value):
            raise ValueError("作者修补器汇总错误")
        return {"patching":value}
    with ZipFile(ROOT/"data/official"/task/PACKAGES[task]) as z:
        src={r["id"]:r["answer"] for r in map(json.loads,z.read("public/validation.jsonl").splitlines())}
    result={}
    folders=[OUTPUT/"domain-stage"/task]
    if task=="engine":
        folders.append(OUTPUT/"engine-filter-ablation")
    for base in folders:
        scores=raw(base/"scores.json")
        for arm,metrics in scores.items():
            recorded={}
            for n in range(1,11):
                original=raw(OUTPUT/"rul/issued"/task/"mature"/f"{n:02d}.json")
                st=raw(base/"rounds"/arm/f"{n:02d}.json")
                if set(st["predictions"])!=set(original["predictions"]):
                    raise ValueError("补充对照未对齐同轮预测")
                if base.name=="engine-filter-ablation":
                    if [c["rows"] for c in st["trials"]]!=[c["rows"] for c in original["trials"]]:
                        raise ValueError("抽样消融没有匹配每个候选的行数")
                elif st["rows"]!=raw(base/"rounds"/"size_control"/f"{n:02d}.json")["rows"]:
                    raise ValueError("阶段对照资源不符")
                recorded.update(st["predictions"])
            value=evaluate(task,[src[k] for k in recorded],list(recorded.values()))
            if not isclose(value,metrics["rmse"],rel_tol=1e-12):
                raise ValueError("补充对照的评分与逐条预测不符")
            result[arm]=value
    return result


if __name__ == "__main__":
    result = {task:audit(task) for task in PACKAGES}
    for task in PACKAGES:
        result[task]["domain_comparisons"]=domain_audit(task)
    atomic_json(OUTPUT/"逐条复核.json",result)
