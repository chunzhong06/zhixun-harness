"""真实在用权重进入换来源流程；逐窗出具交接回执，揭晓服务标签后分别评价各段。"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import BASE, ROOT, PROFILES, LocalStore, _trainer, atomic_json, delivered, digest, metric, official, read, reference_folders, remaining_cycles, remaining_minutes, summary
from compare_proposal_frequency import OUTPUT, TASKS
from engine_endpoint_check import RAW, SUBSETS, read_cmapss, test_answers
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.methods.planning.rul_continuation import device_biases
from harness.models.gas_views import pair_model
from harness.models.bearing_stumps import stump_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.scoped_calibration import ScopedCalibration, TrainingCalibrationBank, score
from harness.runtime.handoff import LocalAcceptance

OUT = BASE/"calibration-expiry"
POLICIES = ("off","keep","refresh","expiry","row","expiry-row","bank-keep","bank-refresh","bank-expiry","bank-route","bank-route-refresh")
OTHER = {"bearing":"pronostia","engine":"phm08","pronostia":"bearing","phm08":"engine"}


def source_window(state, index):
    rows = [r for r in state.values() if r["available_at"]==index]
    return tuple(sorted((r for r in rows if r["role"]==role),key=lambda r:r["sample_id"])
                 for role in ("explore","adopt","serve"))


def unlabelled(task):
    """只交付无答案的当前窗。目标答案单独留给事后评分。"""
    def serving(key, group, values):
        return {"sample_id":key,"unit_id":group,"label":None,"values":values}
    truth = {}
    if task=="gas":
        blocks = []
        for batch in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256):
            if batch.batch_number>=3:
                blocks.append([serving(r.sample_id,str(batch.batch_number),r.values) for r in batch.records])
                truth.update({r.sample_id:r.label for r in batch.records})
        return blocks,truth
    if task=="bearing":
        raw = read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
        end = datetime.fromisoformat(raw[-1]["time"])
        rows = []
        for r in sorted(raw,key=lambda r:r["time"]):
            if "invalid_nonfinite_values" not in r:
                sid = "KAIST/"+r["file"]
                rows.append(serving(sid,"KAIST",r["features"]))
                truth[sid] = (end-datetime.fromisoformat(r["time"])).total_seconds()/60
    else:
        rows = []
        for subset in SUBSETS:
            units = read_cmapss(RAW,subset=subset,partition="test")
            truth.update(test_answers(RAW,subset,units))
            for t in units:
                rows.append(serving(t.unit_id,t.unit_id,(*t.values[-1][3:],*t.values[-1][:3])))
        rows.sort(key=lambda row:digest([17,row["sample_id"]]))
    return [[rows[int(j)] for j in chunk] for chunk in np.array_split(np.arange(len(rows)),8)],truth


def archived(folder, trainer, work, task):
    records = read(folder/"run.json")["rounds"]
    record = records[3]
    payload = read(folder/record["submitted"])["payload"]
    if payload["model_id"]!=record["selected"]:
        raise ValueError("热身末轮的在用权重未真正交接")
    header,blob = ModelArtifacts(folder/"models").read(record["model_artifact"])
    path = work/(header["sha256"]+".npz")
    path.write_bytes(blob)
    wrapper = pair_model if task=="gas" else (stump_model if "bearing_stump" in header["request"] else None)
    model_type = wrapper(trainer.Model) if wrapper else trainer.Model
    return model_type.load(path),header["sha256"],records


def old_step(task, arm, model, fit, explore, confirm, serve, cached):
    def values(m, rows):
        result = m.predict(np.array([r["values"] for r in rows]))
        return result.argmax(axis=1)+1 if task=="gas" else result
    wanted = bool(explore and confirm) and task!="gas"
    if wanted and arm=="old" and task=="bearing":
        pred = values(model,explore)
        by_device = device_biases([SimpleNamespace(**r) for r in explore],
                                  {r["sample_id"]:v for r,v in zip(explore,pred,strict=True)})
        wanted = any(4*r["matching"]>=3*r["recent_count"] for r in by_device)
    event = {"proposed":wanted,"adopted":False,"physical_fit":False}
    if wanted:
        if cached is not None and task=="bearing":
            trial = cached
        else:
            trial = deepcopy(model)
            trial.fit(*fit,1)
            event["physical_fit"] = True
        cached = trial
        for role,rows in (("explore",explore),("confirm",confirm)):
            labels = np.array([r["label"] for r in rows])
            scores = [score(task,labels,values(m,rows)) for m in (model,trial)]
            event[role] = scores
            if scores[1]+1e-12>=scores[0]:
                break
            if role=="confirm":
                model,cached = trial,None
                event["adopted"]=True
    return model,cached,event,{r["sample_id"]:v for r,v in zip(serve,values(model,serve).tolist(),strict=True)}


def compare(cohort, destination=OUT):
    task = TASKS[cohort]
    package,checksum,_ = official(task)
    fixed,_ = reference_folders(cohort,"mature")
    old = BASE/"selected"/task if cohort==task else fixed.parent/"project"
    folders = {"old":old,"paper_once":OUTPUT/cohort/"mature"}
    states = {c:read(reference_folders(c,"mature")[0]/"state.json")["samples"]
              for c in ([cohort,OTHER[cohort]] if task!="gas" else [cohort])}
    fit_ids = {i for i,r in states[cohort].items() if r["role"]=="fit"}
    if any({i for i,r in s.items() if r["role"]=="fit"}!=fit_ids for s in states.values()):
        raise ValueError("各来源训练行不同")
    train = [r for r in package.train if r.sample_id in fit_ids]
    fit = np.array([r.values for r in train]),np.array([r.label for r in train])
    blocks, target_truth = unlabelled(task)
    schedule = [("before",i+1,source_window(states[cohort],i)) for i in range(4,8)]
    schedule.extend(("unlabelled",i+1,([],[],rows)) for i,rows in enumerate(blocks))
    if task!="gas":
        schedule.extend(("new_feedback",i+1,source_window(states[OTHER[cohort]],i)) for i in range(4))
    schedule.extend(("return",i+1,source_window(states[cohort],i)) for i in (8,9))
    root = destination/cohort
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    with LocalStore(root).exclusive(), TemporaryDirectory(prefix="scope-run-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),PROFILES[task]) as trainer:
        base = {k:archived(f,trainer,Path(tmp),task) for k,f in folders.items()}
        models = {k:v[0] for k,v in base.items()}
        started=perf_counter()
        bank=TrainingCalibrationBank(base["old"][0],fit[0])
        bank_seconds=perf_counter()-started
        calibrators = {k:ScopedCalibration(*base["old"][:2],task,fit[0],k,bank if k.startswith("bank-") else None)
                       for k in POLICIES}
        policies = (*models,*calibrators)
        runtime = {k:{} for k in calibrators}
        cached = dict.fromkeys(models)
        streams = {k:{} for k in policies}
        history = defaultdict(list)
        stages, groups = {},{}
        prior_rows = set()
        for r in range(4):
            serve = source_window(states[cohort],r)[2]
            for k in policies:
                source = k if k in models else "old"
                receipt = base[source][2][r]
                pred = read(folders[source]/receipt["submitted"])["payload"]["predictions"]
                if pred.keys()!={row["sample_id"] for row in serve}:
                    raise ValueError("热身交接样本与分片不符")
                streams[k].update(pred)
                if r==3:
                    calculated = calibrators["off"].predictions(None,np.array([row["values"] for row in serve])) if source=="old" else models[source].predict(np.array([row["values"] for row in serve]))
                    if source=="paper_once" and task=="gas":
                        calculated=calculated.argmax(axis=1)+1
                    if not np.allclose(calculated,[pred[row["sample_id"]] for row in serve],rtol=1e-12,atol=1e-8):
                        raise ValueError("载入的在用权重未复现热身末轮交接")
            for rows in source_window(states[cohort],r):
                prior_rows.update(row["sample_id"] for row in rows)
            for row in serve:
                stages[row["sample_id"]]="warmup"
                groups[row["sample_id"]]=row["unit_id"]
        for stage,index,(explore,confirm,serve) in schedule:
            batch = f"{stage}-{index}"
            ids = [r["sample_id"] for rows in (explore,confirm,serve) for r in rows]
            if len(set(ids))!=len(ids) or prior_rows&set(ids) or any(r["label"] is not None for r in serve):
                raise ValueError("不同阶段或用途的资料重叠、服务标签未遮蔽")
            prior_rows.update(ids)
            for row in serve:
                stages[row["sample_id"]]=stage
                groups[row["sample_id"]]=row["unit_id"]
            for k in policies:
                start=perf_counter()
                if k in models:
                    models[k],cached[k],event,pred = old_step(task,k,models[k],fit,explore,confirm,serve,cached[k])
                    if stage=="before":
                        record=base[k][2][index-1]
                        prior=read(folders[k]/record["submitted"])["payload"]["predictions"]
                        if set(prior)!=set(pred) or not np.allclose(list(pred.values()),[prior[i] for i in pred],rtol=1e-12,atol=1e-8):
                            raise ValueError(f"{cohort}/{k} 未复现切换前第 {index} 轮")
                else:
                    runtime[k],event,pred=calibrators[k].advance(runtime[k],batch,explore,confirm,serve)
                elapsed=perf_counter()-start
                payload={"batch_id":batch,"policy":k,"origin":event.get("using"),"predictions":pred}
                request=digest(payload)
                endpoint=LocalAcceptance()
                endpoint.output=root/"accepted"/k
                if endpoint.submit(request,payload).get("accepted") is not True:
                    raise ValueError("本地交接未获确认")
                history[k].append({**event,"batch":batch,"elapsed_seconds":elapsed,"request":request})
                streams[k].update(pred)
        sealed=root/"issued.json"
        if sealed.exists():
            previous=read(sealed)
            if any(streams[k]!=p for k,p in previous["streams"].items()):
                raise ValueError("重跑未能复现先前已交出的预测")
        atomic_json(sealed,{"package_sha256":checksum,"prefix_sha256":{k:v[1] for k,v in base.items()},
                            "streams":streams,"stages":stages,"history":dict(history),"terminal":runtime})
        truth={r.sample_id:r.label for r in package.validation}
        if task=="engine":
            truth.update(remaining_cycles())
        elif task=="bearing":
            truth.update(remaining_minutes("full-test-manifest.json"))
        truth.update(target_truth)
        # 事后诊断单独存放。这里的挑选结果无法回流到已提交的任何策略。
        oracle={i:v for i,v in streams["off"].items() if stages[i]=="warmup"}
        best_choices={}
        for stage,index,(_,_,serve) in schedule:
            ids=[r["sample_id"] for r in serve]
            x=np.array([r["values"] for r in serve])
            options=[dict(zip(ids,calibrators["off"].predictions(None if j==0 else option,x).tolist(),strict=True))
                     for j,option in enumerate(bank.options)]
            scores=[metric(task,truth,p) for p in options]
            best=(max if task=="gas" else min)(range(len(scores)),key=scores.__getitem__)
            oracle.update(options[best])
            best_choices[f"{stage}-{index}"]={"group":best,"scores":scores}
        atomic_json(root/"bank-opportunity.json",{"choices":best_choices,"whole_score":metric(task,truth,oracle),
                   "issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"diagnostic_only":True})
        results={}
        for k,pred in streams.items():
            if pred.keys()!=stages.keys():
                raise ValueError("分组交接样本不同")
            by_stage={}
            for part in ("whole",*dict.fromkeys(stages.values())):
                subset={i:p for i,p in pred.items() if part=="whole" or stages[i]==part}
                members=defaultdict(dict)
                for i,p in subset.items():
                    members[truth[i] if task=="gas" else groups[i]][i]=p
                by_unit={str(g):(sum(truth[i]==p for i,p in m.items())/len(m) if task=="gas" else metric(task,truth,m))
                         for g,m in members.items()}
                by_stage[part]={"score":metric(task,truth,subset),"rows":len(subset),
                                "worst":(min if task=="gas" else max)(by_unit.values()),"groups":by_unit}
            logs=history[k]
            rounds={}
            for stage,index,(_,_,rows) in schedule:
                ids=[r["sample_id"] for r in rows]
                rounds[f"{stage}-{index}"]=metric(task,truth,{i:pred[i] for i in ids})
            results[k]={"stages":by_stage, "elapsed_seconds":sum(r["elapsed_seconds"] for r in logs),
                        "round_scores":rounds,
                        "proposals":sum(r["proposed"] for r in logs),"adopted":sum(r["adopted"] for r in logs),
                        "expired":sum(r.get("expired",False) for r in logs),"physical_fits":sum(r.get("physical_fit",False) for r in logs)}
        atomic_json(root/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":results,
                   "training_bank":{"seconds":bank_seconds,"size":len(bank.options),"sha256":bank.identity}})
        print(cohort,{k:round(result["stages"]["whole"]["score"],5) for k,result in results.items()})


def public_scores(task):
    """独立汇总十轮真正确认的输出，核对前缀、续跑身份、受保护气体预测和四臂服务分片。"""
    package,_,batches=official(task)
    expected={r.sample_id for b in batches for r in b.samples if r.role=="serve" and r.available_at==b.ordinal}
    truth={r.sample_id:r.label for r in package.validation}
    units={r.sample_id:r.unit_id for r in package.validation}
    source=BASE/"selected"/task
    _,paper=reference_folders(task,"mature")
    streams={"parent":delivered(BASE/"original"/task),"old":delivered(source),"paper":delivered(paper)}
    for policy in ("expiry","refresh","bank-route-refresh"):
        out=OUT/"public"/policy/task
        report=read(out/"run.json")
        if report["rounds"]!=10:
            raise ValueError("公开入口未走完十轮")
        stream={}
        for idx,entry in enumerate([*report["prefix"],*report["accepted"]]):
            prefix=idx<len(report["prefix"])
            folder=source if prefix else out
            ack=read(folder/"accepted"/(entry["request"]+".json"))
            payload=ack["payload"]
            if (ack["request_id"]!=entry["request"] or digest(payload["predictions"])!=entry["outputs_sha256"]
                    or payload["batch_id"]!=batches[idx].batch_id or stream.keys() & payload["predictions"].keys()
                    or (not prefix and digest([report["identity"],payload])!=entry["request"])):
                raise ValueError("公开交接内容或身份与报告不符")
            stream.update(payload["predictions"])
        streams[policy]=stream
    if any(s.keys()!=expected for s in streams.values()):
        raise ValueError("同条件服务分片不符")
    results=summary(task,truth,streams,units)
    atomic_json(OUT/"public"/(task+"-scores.json"),{"arms":results,"predictions_sha256":{k:digest(v) for k,v in streams.items()}})
    print(task,{k:round(v["score"],5) for k,v in results.items()})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=TASKS)
    parser.add_argument("--public-scores",action="store_true")
    args=parser.parse_args()
    if args.public_scores:
        if args.cohort not in PROFILES:
            parser.error("官方入口计分需要官方任务名")
        public_scores(args.cohort)
    else:
        compare(args.cohort)
