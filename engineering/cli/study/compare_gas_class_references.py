"""冻结的整段气体查法与按类别取最近参照的配对比较。开发决定先于后续批诊断。"""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
import json

# 复用上轮相同的检查点加载、特征表示与单线程设置。
from compare_gas_phases import ROOT, FrozenComparison, checked, read
import numpy as np
from harness.data.gas_input import read_gas_batches
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.models.gas_views import represent
from harness.runtime.storage import atomic_json


ARMS = ("whole", "direct", "protected")


class ClassReferenceComparison:
    def __init__(self, frozen, package, path, classes):
        self.frozen = FrozenComparison(package, frozen, path)
        self.parent = self.frozen.models["base"]
        self.current = self.frozen.models["double"]
        self.lookup = self.frozen.lookup
        self.ids = [r.sample_id for r in package.train]
        self.classes = classes
        self.target, self.source = classes["target"], classes["ethanol"]
        self.ethylene = classes["ethylene"]
        self.subsets = {name:np.flatnonzero(self.lookup.labels == classes[name])
                        for name in ("target", "ethanol", "toluene")}

    def evaluate(self, x, arm, *, trace=False):
        """返回分类判断。逐条证据另外保存真实距离及来源身份，不计入服务耗时。"""
        original = self.parent.predict(x)[0]
        base = int(original.argmax()+1)
        current = base
        if base == self.source or trace:
            vec = represent(x,"shape",self.parent.mean,self.parent.scale)[0]
            distances = self.lookup.distance(vec,"whole")
        if base == self.source:
            top = np.argsort(distances,kind="stable")[:5]
            votes = np.bincount(self.lookup.labels[top],minlength=7)[1:] / 5
            proposed = int(((votes + original)/2).argmax()+1)
            if proposed in (self.target,self.ethylene):
                current = proposed
        eligible = base == self.source
        if not trace and (arm == "whole" or not eligible or (arm == "protected" and current != self.source)):
            return current
        nearest = {name:idx[np.argmin(distances[idx])] for name,idx in self.subsets.items()}
        prefer_target = all(distances[nearest["target"]] < distances[nearest[name]]
                            for name in ("ethanol","toluene"))
        direct = self.target if eligible and prefer_target else current
        protected = direct if current == self.source else current
        if not trace:
            return direct if arm == "direct" else protected
        query_norm = np.dot(vec,vec)
        return {"base":base,"whole":current,"direct":direct,"protected":protected,
                "nearest":{name:{"train_id":self.ids[i],
                                 "distance_squared":float(max(0.0,distances[i]+query_norm))}
                           for name,i in nearest.items()}}


def observe(rows, archived, comparison):
    if {sid for sid,_,_ in rows} != set(archived) or len(rows)!=len(archived):
        raise ValueError("配对资料的样本身份不相符")
    records = []
    for sid,label,values in rows:
        x = np.asarray([values])
        r = comparison.evaluate(x,"whole",trace=True)
        if (r["base"],r["whole"]) != archived[sid] or r["whole"] != int(comparison.current.predict(x)[0].argmax()+1):
            raise ValueError(f"整段或原版预测与冻结版本不符：{sid}")
        records.append({"id":sid,"answer":label,**r})
    return records


def summarize(rows, classes, arms=ARMS):
    target, ethylene = classes["target"], classes["ethylene"]
    recovered = [r for r in rows if r["answer"]==ethylene and r["base"]!=ethylene and r["whole"]==ethylene]
    result = {"rows":len(rows), "support":dict(Counter(r["answer"] for r in rows)),
              "recovered_ethylene":len(recovered),"arms":{}}
    for arm in arms:
        fixed, broken, changed = Counter(),Counter(),Counter()
        for r in rows:
            truth, old, new = r["answer"], r["whole"], r[arm]
            if old != new:
                changed[truth,old,new] += 1
                if old == truth:
                    broken[truth] += 1
                if new == truth:
                    fixed[truth] += 1
        result["arms"][arm] = {
            "macro_f1":macro_f1((r["answer"],r[arm]) for r in rows),
            "target_correct":sum(r["answer"] == target and r[arm] == target for r in rows),
            "target_false_positives":dict(Counter(r["answer"] for r in rows if r["answer"]!=target and r[arm]==target)),
            "corrected":dict(fixed),"broken":dict(broken),
            "lost_recovered_ethylene":sum(r[arm]!=ethylene for r in recovered),
            "transitions":[[truth,old,new,n] for (truth,old,new),n in sorted(changed.items())]}
    return result


def decision(primary, remainder, classes):
    good = []
    for arm in ARMS[1:]:
        permitted = True
        for section in (primary,remainder):
            current, trial = section["arms"]["whole"], section["arms"][arm]
            gain = trial["target_correct"]-current["target_correct"]
            delta = trial["macro_f1"]-current["macro_f1"]
            if (gain<0 or delta<0 or (section is primary and (gain==0 or delta==0))
                    or trial["lost_recovered_ethylene"] or any(trial["broken"].get(classes[k],0) for k in ("ethanol","toluene"))):
                permitted = False
        if permitted:
            good.append(arm)
    return max(good,key=lambda arm:(primary["arms"][arm]["macro_f1"],arm=="protected"),default="whole")


def measure(comparison, rows, expected, passes):
    inputs = [np.asarray([values]) for _,_,values in rows]
    arms = ("deployed", *ARMS)
    times = {k:[] for k in arms}
    for step in range(passes+1):
        offset = step % len(arms)
        for arm in arms[offset:]+arms[:offset]:
            start = perf_counter_ns()
            if arm == "deployed":
                predictions = [int(comparison.current.predict(x)[0].argmax()+1) for x in inputs]
            else:
                predictions = [comparison.evaluate(x,arm) for x in inputs]
            duration = (perf_counter_ns()-start)/len(rows)/1000
            if predictions != expected["whole" if arm=="deployed" else arm]:
                raise ValueError("计时入口与逐条分类结果不一致")
            if step:
                times[arm].append(duration)
    return {"median_us":{k:float(np.median(v)) for k,v in times.items()},"passes_us":times}


def run(protocol, output, stage):
    rule = read(protocol)
    config = read(ROOT/rule["frozen"])
    archive = checked(ROOT/config["official"],config["official_sha256"])
    package = read_official_gas(archive)
    comparison = ClassReferenceComparison(config,package,archive,rule["classes"])
    protocol_hash = sha256(protocol.read_bytes()).hexdigest()
    if stage == "development":
        previous = read(ROOT/rule["development"])
        held = {r["id"] for r in previous if r["primary"]}
        rows = [(r.sample_id,r.label,r.values) for r in package.validation]
        results = observe(rows,{r["id"]:(r["base"],r["double"]) for r in previous},comparison)
        phase = {r["id"]:r["label"] for r in read(ROOT/rule["previous_candidate"])}
        for r in results:
            r["phase"] = phase[r["id"]]
        primary = [r for r in results if r["id"] in held]
        remainder = [r for r in results if r["id"] not in held]
        a,b = [summarize(s,rule["classes"],(*ARMS,"phase")) for s in (primary,remainder)]
        choice = decision(a,b,rule["classes"])
        timing = measure(comparison,[r for r in rows if r[0] in held],
                         {k:[r[k] for r in primary] for k in ARMS},rule["timing_passes"])
        atomic_json(output/"development-rows.json",results)
        atomic_json(output/"development.json",{"primary":a,"remainder":b,"choice":choice,"timing":timing,
                                               "protocol_sha256":protocol_hash})
        return
    dev_file = output/"development.json"
    if read(dev_file)["protocol_sha256"]!=protocol_hash:
        raise ValueError("诊断前需冻结同一协议的开发选择")
    archive_rows = [json.loads(line) for line in (ROOT/rule["diagnostic_reference"]).read_bytes().splitlines()]
    out = {}
    for n in (9,10):
        batch, = read_gas_batches(ROOT/config["source"],expected_sha256=config["source_sha256"],batch_number=n)
        rows = [(r.source_line,r.label,r.values) for r in batch.records]
        results = observe(rows,{r["line"]:(r["base"],r["double"]) for r in archive_rows if r["batch"]==n},comparison)
        out[n] = summarize(results,rule["classes"])
        atomic_json(output/f"diagnostic-{n}-rows.json",results)
    atomic_json(output/"diagnosis.json",{"development_sha256":sha256(dev_file.read_bytes()).hexdigest(),"batches":out})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--stage",required=True,choices=("development","diagnosis"))
    args = parser.parse_args()
    run(args.protocol,args.output,args.stage)
