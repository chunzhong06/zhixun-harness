"""只用以前返回的实测为下一批选择尺度，复用封存的两个候选预测。"""
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

from score_gas_probability_fusion import (
    ROOT, BASE, ARCHIVE, SOURCE, SEEDS, LocalAcceptance, LocalStore, atomic_json, digest, metrics, read, read_official_gas,
)
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches

DEST = BASE / "whole-maintenance/gas-scale-choice"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-scale-choice.json"


def decide(history, labels, source_scale, centers):
    if not history:
        return "source","source",{}
    x = np.asarray(history)
    y = np.asarray(labels)
    classes = np.unique(y)
    scores = {}
    for key,scale in (("source",source_scale),("site",StandardScaler().fit(x))):
        z = scale.transform(x)
        difference = np.stack([z[y==c].mean(0)-centers[c] for c in classes])
        frequency = np.array([np.mean(y==c) for c in classes])
        scores[key] = {"classwise":float(np.mean(difference**2)),
                       "marginal":float(np.mean((frequency @ difference)**2))}
    choices = [min(scores,key=lambda name:scores[name][measure]) for measure in ("classwise","marginal")]
    return choices[0],choices[1],scores


def evaluate(site,seed,records,feedback,lookup,truth,source_scale,centers,arms):
    folder = DEST/site/f"seed-{seed}"
    receiver = LocalAcceptance()
    receiver.output = folder/"accepted"
    history,labels,requests = [],[],[]
    inputs = read(records/"scores.json")["requests"]
    for j,key in enumerate(inputs):
        payload = read(records/"accepted"/(key+".json"))["payload"]
        keys = payload["ids"]
        options = {name:payload["predictions"][arm] for name,arm in arms.items()}
        a,b,loss = decide(history,labels,source_scale,centers)
        chosen = {"classwise":options[a],"marginal":options[b],**options}
        submission = {"ids":keys,"prior_feedback":len(labels),"source_request":key,"history_score":loss,
                      "classwise_choice":a,"marginal_choice":b,"predictions":chosen}
        request = digest([digest(read(PROTOCOL)),site,seed,submission])
        ack = receiver.submit(request,submission)
        if not ack["accepted"] or ack["request_id"]!=request:
            raise RuntimeError("未确认交接")
        requests.append(request)
        if j+1<len(inputs):
            items = feedback(j,key)
            if len(items)!=32 or len(set(k for k,_ in items))!=32:
                raise ValueError("反馈样本清单不符")
            for row_id,label in items:
                history.append(lookup[row_id])
                labels.append(label)
    logs = [read(receiver.output/(r+".json"))["payload"] for r in requests]
    ids = [k for row in logs for k in row["ids"]]
    if len(set(ids))!=len(ids) or len(ids)!={"public":414,"late":12221}[site]:
        raise ValueError("计分范围有误")
    y = np.array([truth[k] for k in ids])
    scores = {name:metrics(y,np.concatenate([row["predictions"][name] for row in logs])) for name in chosen}
    correct = {arm:0 for arm in ("source","site","upper","classwise")}
    for row in logs:
        actual = np.array([truth[k] for k in row["ids"]])
        counts = {arm:int(np.sum(actual==row["predictions"][arm])) for arm in ("source","site","classwise")}
        for arm,val in counts.items():
            correct[arm]+=val
        correct["upper"]+=max(counts["source"],counts["site"])
    atomic_json(folder/"scores.json",{"requests":requests,"scores":scores,"correct":correct})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此协议已执行")
        package = read_official_gas(ARCHIVE)
        source = np.asarray([r.values for r in package.train])
        y = np.asarray([r.label for r in package.train])
        scale = StandardScaler().fit(source)
        z = scale.transform(source)
        centers = {c:z[y==c].mean(0) for c in np.unique(y)}
        rows = read(SOURCE/"state.json")["samples"].values()
        first = {r["sample_id"]:r["values"] for r in rows}
        early_truth = {r.sample_id:r.label for r in package.validation}
        pools = [{r["sample_id"]:r["label"] for r in rows if r["available_at"]==j and r["role"] in ("adopt","explore")}
                 for j in range(9)]
        # 读包返回一次性迭代器，同一批实体供特征与评分两张表使用。
        batches = list(read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256))
        last = {r.sample_id:r.values for b in batches if b.batch_number>=3 for r in b.records}
        late_truth = {r.sample_id:r.label for b in batches if b.batch_number>=3 for r in b.records}
        results = {}
        for site in ("public","late"):
            draws = []
            early = site=="public"
            prefix = "gas-cold-start" if early else "gas-label-mass-late"
            arms = {"source":"pooled-source","site":"pooled-separate","current":"fixed-midway"} if early else {
                    "source":"domain-source","site":"domain-site","current":"current"}
            arms["observer"]="observer"
            for seed in SEEDS:
                prior = BASE/"whole-maintenance"/prefix/f"seed-{seed}"
                def feedback(j,key):
                    if early:
                        return [(k,pools[j][k]) for k in sorted(pools[j],key=lambda k:digest([seed,k]))[:32]]
                    saved = read(prior/"feedback"/(key+".json"))
                    return list(zip(saved["ids"],saved["labels"],strict=True))
                draws.append(evaluate(site,seed,prior,feedback,first if early else last,
                                      early_truth if early else late_truth,scale,centers,arms))
            means = {a:{m:float(np.mean([r[a][m] for r in draws])) for m in ("macro","worst")} for a in draws[0]}
            results[site] = {"means":means,"default":draws[0]}
            print(json.dumps({site:means},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",{"sites":results,"protocol":digest(read(PROTOCOL))})


if __name__=="__main__":
    main()
