"""把本地入口每轮接受的实际输出与独立重算逐条核对，再读答案评分。"""
from hashlib import sha256
from statistics import mean
import subprocess
import sys
import json

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,ROOT,REGISTERED_SHA256,read,read_gas_batches,atomic_json,digest
from score_guarded_gas import SEEDS,public_stream
from harness.data.official_gas import LABELS,prediction_label,prediction_request


def verify(cohort,truth,expected_pool):
    folder=OUT/"observer-errors"
    reference=read(folder/cohort/"issued.json")
    if sha256((folder/cohort/"issued.json").read_bytes()).hexdigest()!=read(folder/cohort/"scores.json")["issued_sha256"]:
        raise ValueError("独立预测与评分记录不一致")
    table={}
    for seed in SEEDS:
        source=folder/"live"/cohort/str(seed)
        rep=read(source/"run.json")
        issued={}
        seen=0
        sizes=[]
        seconds=0.
        for evt,(ids,eligible) in zip(rep["accepted"],expected_pool,strict=True):
            receipt=read(source/"accepted"/(evt["request"]+".json"))
            p=receipt["payload"]
            if (receipt["request_id"]!=evt["request"] or digest([rep["identity"],p])!=evt["request"] or
                    set(p["predictions"])!=set(ids) or evt["returned"]!=sorted(eligible,key=lambda k:digest([seed,k]))[:32]
                    or issued.keys()&p["predictions"].keys() or p["feedback_seen"]!=seen):
                raise ValueError("交付身份、服务行或回标时间不符")
            issued.update(p["predictions"])
            sizes.append(len(ids))
            seen+=len(evt["returned"])
            seconds+=evt["seconds"]
        keys=reference["keys"]
        pred=np.asarray([issued[i] for i in keys])
        if not np.array_equal(pred,reference["predictions"][f"uniform-{seed}"]):
            raise ValueError("逐条服务与冻结回放输出不符")
        blocks=np.split(np.arange(len(keys)),np.cumsum(sizes)[:-1])
        result=scores(np.asarray([truth[i] for i in keys]),pred,blocks,6)
        table[str(seed)]={**result,"seconds":seconds,"feedback_seen_before_last":seen-len(rep["accepted"][-1]["returned"]),
                          "rows":len(keys),"run_sha256":sha256((source/"run.json").read_bytes()).hexdigest()}
    val={k:mean(r[k] for r in table.values()) for k in ("pooled","worst_batch","equal_batches","seconds")}
    return {"means":val,"seeds":table}


def verify_standalone(truth):
    root=OUT/"observer-errors"
    folder=root/"standalone"
    report=read(folder/"run.json")
    reference=read(root/"public/issued.json")
    source=read(folder/"regular/state.json")["samples"]
    predictions={}
    launch_seconds=[]
    from time import perf_counter
    for event in report["accepted"]:
        payload=read(folder/"accepted"/(event["request"]+".json"))["payload"]
        snapshot=folder/"snapshots"/(payload["snapshot"]+".json")
        state=read(snapshot)
        if (digest(state)!=payload["snapshot"] or digest(state["counts"])!=payload["rule"]["previous_counts_sha256"]
            or np.asarray(state["counts"])[0].sum()!=payload["feedback_seen"] or state["weights"]!=payload["weights"]):
            raise ValueError("单条服务快照与现场权重、已见反馈不一致")
        keys=list(payload["predictions"])
        requests="".join(json.dumps(prediction_request(source[k]["values"]))+"\n" for k in keys)
        tic=perf_counter()
        proc=subprocess.run([sys.executable,"-X","utf8","-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),
            "gas-predict","--snapshot",str(snapshot)],input=requests,encoding="utf-8",capture_output=True,check=True)
        launch_seconds.append(perf_counter()-tic)
        replies=[json.loads(line) for line in proc.stdout.splitlines()]
        for k,r in zip(keys,replies,strict=True):
            vals=r["outputs"]
            if ([v["label"] for v in vals]!=list(LABELS) or any(v["score"]<0 for v in vals) or
                    abs(sum(v["score"] for v in vals)-1)>1e-12 or prediction_label(r)!=payload["predictions"][k]):
                raise ValueError("单条 JSON 输出不符合接口或与交付类别不同")
            predictions[k]=prediction_label(r)
    keys=reference["keys"]
    if [predictions[k] for k in keys]!=reference["predictions"][f"uniform-{report['seed']}"]:
        raise ValueError("重启推理所得的类别与冻结回放不符")
    sizes=[len(read(folder/"accepted"/(e["request"]+".json"))["payload"]["predictions"]) for e in report["accepted"]]
    blocks=np.split(np.arange(len(keys)),np.cumsum(sizes)[:-1])
    result=scores(np.asarray([truth[i] for i in keys]),np.asarray([predictions[i] for i in keys]),blocks,6)
    result.update(rows=len(keys),seed=report["seed"],feedback_seen_before_last=payload["feedback_seen"],
                  last_snapshot=payload["snapshot"],single_json_process_seconds=launch_seconds,run_sha256=sha256((folder/"run.json").read_bytes()).hexdigest())
    atomic_json(folder/"scores.json",result)
    return result


if __name__=="__main__":
    stream,truth,_=public_stream()
    a=verify("public",truth,[(ids,returned) for ids,_,returned,_,_ in stream])
    batches=[b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
    truth={r.sample_id:r.label for b in batches for r in b.records}
    groups=[[r.sample_id for r in b.records] for b in batches]
    b=verify("late",truth,[(g,g) for g in groups])
    atomic_json(OUT/"observer-errors/live/scores.json",{"public":a,"late":b})
    print(a["means"],b["means"])
    folder=OUT/"observer-errors/standalone"
    if (folder/"run.json").exists() and len(read(folder/"run.json")["accepted"])==10:
        print("单条接口全部十轮",verify_standalone(public_stream()[1])["pooled"])
