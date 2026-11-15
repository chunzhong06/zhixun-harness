"""用同一冻结的气体模型比较三种参照查法，开发选择和已见批次诊断分别运行。"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BASELINE_CPU_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(ROOT / "engineering/src"))
import numpy as np
from harness.data.gas_input import read_gas_batches
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.models.gas_views import pair_model, represent
from harness.runtime.storage import atomic_json

ARMS = ("whole", "dynamic", "separate")


def read(path):
    return json.loads(path.read_bytes())


def checked(path, checksum):
    if sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"冻结文件内容改变：{path}")
    return path


class Lookup:
    """在同一参照库上预备各阶段的特征列。并列按冻结训练行顺序打破。"""

    def __init__(self, training, labels):
        self.labels = labels
        channels = np.arange(128).reshape(16, 8)
        self.columns = {"steady":channels[:, :2].ravel(), "dynamic":channels[:, 2:].ravel(), "rise":channels[:, 2:5].ravel(),
                        "fall":channels[:, 5:8].ravel()}
        views = {"whole":training, **{k:training[:, ix] for k,ix in self.columns.items()}}
        self.banks = {k:(np.ascontiguousarray(v.T), (v*v).sum(axis=1)) for k,v in views.items()}
        self.positions = np.arange(len(labels))

    def distance(self, vector, view):
        weights, norms = self.banks[view]
        query = vector if view == "whole" else vector[self.columns[view]]
        # 同一查询到各参照的距离可省略查询自身的平方范数，排序保持相同。
        return norms - (2 * query) @ weights

    def top_five(self, vector, arm):
        if arm in ("whole","dynamic"):
            return np.argsort(self.distance(vector,arm), kind="stable")[:5]
        rise_dist = self.distance(vector,"rise")
        fall_dist = self.distance(vector,"fall")
        if arm == "full_separate":
            half_steady = self.distance(vector,"steady") / 2
            rise_dist += half_steady
            fall_dist += half_steady
        rise = np.argsort(rise_dist,kind="stable")
        fall = np.argsort(fall_dist,kind="stable")
        # 排名是对同一个训练样本的两次评价。合并前还原训练行编号，避免把不同记录拼在一起。
        rank_sum = np.empty_like(rise)
        rank_sum[rise] = self.positions
        rank_sum[fall] += self.positions
        return np.argsort(rank_sum, kind="stable")[:5]


class FrozenComparison:
    def __init__(self, package, frozen, package_path):
        sys.path.insert(0, str(package_path))
        from baselines.model import Model
        self.models = {}
        for arm in ("base", "double"):
            item = frozen["arms"][arm]
            self.models[arm] = pair_model(Model).load(checked(ROOT / item["path"], item["sha256"]))
        double = self.models["double"]
        memory = double.meta["gas_pair_patch"]
        if (memory["method"], memory["policy"], memory["rules"]) != (
            "shape-blend","protected",[{"reported":1,"target":4},{"reported":1,"target":2}]):
            raise ValueError("当前检查点与两条冻结的修正规则不匹配")
        training = represent([r.values for r in package.train], "shape", double.mean, double.scale)
        labels = np.array([r.label for r in package.train])
        bank = double._prediction_bank(memory)
        if not np.array_equal(training,bank["train"]) or not np.array_equal(labels-1,bank["labels"]):
            raise ValueError("官方训练样本或顺序与冻结参照不一致")
        self.lookup = Lookup(training, labels)

    def predict(self, x, arm):
        """只接收输入；附带五条参照的位置，用于预测后统计。"""
        parent = self.models["base"]
        scores = parent.predict(x)[0]
        if scores.argmax() != 0:
            return scores, ()
        vector = represent(x, "shape", parent.mean, parent.scale)[0]
        neighbors = self.lookup.top_five(vector, arm)
        votes = np.bincount(self.lookup.labels[neighbors], minlength=7)[1:] / 5
        proposal = (scores + votes) / 2
        if int(proposal.argmax())+1 in (2,4):
            scores = proposal
        return scores, neighbors


def observe(rows, archived, comparison, train_ids, arms):
    records = []
    if len(rows) != len(archived) or {sid for sid,_,_ in rows} != set(archived):
        raise ValueError("请求的来源行和历史预测不完全一致")
    for sid, truth, values in rows:
        x = np.asarray([values])
        real = comparison.models["double"].predict(x)[0]
        record = {"id":sid,"answer":truth}
        for arm in arms:
            scores, neighbors = comparison.predict(x, arm)
            if arm == "whole" and (not np.array_equal(scores, real) or int(real.argmax()+1) != archived[sid]):
                raise ValueError(f"整段对照与现用模型不一致：{sid}")
            labels = comparison.lookup.labels[list(neighbors)] if len(neighbors) else ()
            record[arm] = {"label":int(scores.argmax()+1), "acetaldehyde_votes":sum(int(v)==4 for v in labels),
                           "neighbors":[train_ids[i] for i in neighbors]}
        records.append(record)
    return records


def summarize(rows, arms):
    target = [r for r in rows if r["answer"] == 4]
    checked_target = [r for r in target if r["whole"]["neighbors"]]
    scores = {}
    for arm in arms:
        changes = Counter((r["answer"],r["whole"]["label"],r[arm]["label"]) for r in rows
                          if r["whole"]["label"] != r[arm]["label"])
        fixes = Counter()
        breaks = Counter()
        for (truth, before, after), count in changes.items():
            if after == truth:
                fixes[truth] += count
            if before == truth:
                breaks[truth] += count
        votes = [r[arm]["acetaldehyde_votes"] for r in checked_target]
        scores[arm] = {"macro_f1":macro_f1((r["answer"],r[arm]["label"]) for r in rows),
            "target_in_top5":sum(v>0 for v in votes), "target_votes":sum(votes),
            "target_vote_histogram":dict(Counter(votes)), "target_correct":sum(r[arm]["label"]==4 for r in target),
            "correct":sum(r["answer"]==r[arm]["label"] for r in rows),
            "fixed":dict(fixes), "broken":dict(breaks),
            "changes":[[a,b,c,n] for (a,b,c),n in sorted(changes.items())]}
    return {"rows":len(rows), "target":len(target), "target_entered":len(checked_target),
            "entered":sum(bool(r["whole"]["neighbors"]) for r in rows),"scores":scores}


def benchmark(comparison, inputs, passes, variants):
    arms = ("deployed", *variants)
    def predict(x, arm):
        if arm == "deployed":
            return comparison.models["double"].predict(x)[0]
        return comparison.predict(x,arm)[0]
    fingerprints = {}
    times = {k:[] for k in arms}
    for rep in range(passes+1):
        offset = rep % len(arms)
        for arm in arms[offset:]+arms[:offset]:
            start = perf_counter_ns()
            predictions = [predict(x,arm) for x in inputs]
            elapsed = (perf_counter_ns()-start)/len(inputs)/1000
            stamp = sha256(np.asarray(predictions,dtype="<f8").tobytes()).hexdigest()
            if rep:
                if fingerprints[arm] != stamp:
                    raise ValueError("计时轮中预测改变")
                times[arm].append(elapsed)
            else:
                fingerprints[arm] = stamp
    if fingerprints["whole"] != fingerprints["deployed"]:
        raise ValueError("共用比较入口与实际整段输出不一致")
    return {"unit":"microseconds_per_row","passes":times,"median":{k:float(np.median(v)) for k,v in times.items()}}


def choose(primary, remainder, timing, arms):
    def eligible(arm):
        for section in (primary,remainder):
            base, candidate = section["scores"]["whole"],section["scores"][arm]
            if candidate["target_correct"] < base["target_correct"] or candidate["macro_f1"] < base["macro_f1"]:
                return False
        candidate = primary["scores"][arm]
        base = primary["scores"]["whole"]
        return (candidate["target_correct"] > base["target_correct"] and candidate["macro_f1"] > base["macro_f1"]
                and not any(candidate["broken"].get(c,0) for c in (1,6)))
    accepted = [arm for arm in arms[1:] if eligible(arm)]
    return max(accepted,key=lambda arm:(primary["scores"][arm]["macro_f1"],-timing["median"][arm]),default="whole")


def run(protocol, output, stage):
    followup = stage != "development"
    rules = read(ROOT/read(protocol)["initial_protocol"]) if followup else read(protocol)
    arms = (*ARMS,"full_separate") if followup else ARMS
    frozen = read(ROOT/rules["frozen"])
    archive = checked(ROOT/frozen["official"],frozen["official_sha256"])
    package = read_official_gas(archive)
    comparison = FrozenComparison(package,frozen,archive)
    train_ids = [r.sample_id for r in package.train]
    if stage in ("development","matched"):
        previous = read(ROOT/rules["development_rows"])
        held = {r["id"] for r in previous if r["primary"]}
        rows = [(r.sample_id,r.label,r.values) for r in package.validation]
        results = observe(rows,{r["id"]:r["double"] for r in previous},comparison,train_ids,arms)
        primary = summarize([r for r in results if r["id"] in held],arms)
        remainder = summarize([r for r in results if r["id"] not in held],arms)
        timing = benchmark(comparison,[np.asarray([x]) for sid,_,x in rows if sid in held],rules["timing_passes"],arms)
        if followup:
            original = read(output/"development-rows.json")
            if len(original) != len(results) or any(
                {k:v for k,v in new.items() if k!="full_separate"} != old
                for old,new in zip(original,results)):
                raise ValueError("同输入对照改变了首轮逐条比较")
            atomic_json(output/"matched-rows.json",[{"id":r["id"],**r["full_separate"]} for r in results])
        else:
            atomic_json(output/"development-rows.json",results)
        filename = "matched.json" if followup else "development.json"
        atomic_json(output/filename,{"primary":primary,"remainder":remainder,"timing":timing,
                    "choice":choose(primary,remainder,timing,arms),"protocol_sha256":sha256(protocol.read_bytes()).hexdigest()})
        return
    development = output/"matched.json"
    if read(development)["protocol_sha256"] != sha256(protocol.read_bytes()).hexdigest():
        raise ValueError("后续批诊断须沿用已完成的开发协议")
    history = [json.loads(s) for s in (ROOT/rules["diagnostic_reference"]).read_bytes().splitlines()]
    summary = {}
    for n in (9,10):
        batch, = read_gas_batches(ROOT/frozen["source"],expected_sha256=frozen["source_sha256"],batch_number=n)
        rows = [(r.source_line,r.label,r.values) for r in batch.records]
        archived = {r["line"]:r["double"] for r in history if r["batch"]==n}
        results = observe(rows,archived,comparison,train_ids,arms)
        summary[n] = summarize(results,arms)
        atomic_json(output/f"diagnosis-{n}-rows.json",[r for r in results if r["answer"]==4 or
                    any(r[arm]["label"]!=r["whole"]["label"] for arm in arms[1:])])
    atomic_json(output/"diagnosis.json",{"frozen_development_sha256":sha256(development.read_bytes()).hexdigest(),
                "batches":summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--stage",choices=("development","matched","diagnosis"),required=True)
    args = parser.parse_args()
    run(args.protocol,args.output,args.stage)
