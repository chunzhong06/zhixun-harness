"""用相同回标实际训练线性/径向核；旧线性参照重载快照，新权重先交接再评分。"""
from hashlib import sha256
import json
from time import process_time

import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from score_gas_probability_fusion import (ROOT,BASE,ARCHIVE,SEEDS,stream,metrics,read,
    read_official_gas,GasFeedbackBank,model_scores,LocalAcceptance,LocalStore,atomic_json,digest)
from harness.models.field_classification import weights
from harness.models.observer_errors import ObserverErrors
from harness.models.training_tools import SklearnBackend

DEST=BASE/"whole-maintenance/gas-field-kernel"
PRIOR=BASE/"whole-maintenance/gas-domain-augmentation"
PROTOCOL=ROOT/"engineering/configs/experiments/gas-field-kernel.json"
ARMS=tuple(f"svc-{kernel}-{scale}" for kernel in ("linear","rbf") for scale in ("source","separate"))


def fit_models(source,sy,history,answers,folder):
    """训练端的两域变换分别拟合；发布流水线只保留现场需要的一个标准化器。"""
    source_scale=StandardScaler().fit(source)
    z=source_scale.transform(source);y=sy;w=weights(sy,len(sy))
    target_scale=StandardScaler().fit(np.asfortranarray(history)) if history else source_scale
    backend=SklearnBackend(execution_enabled=True)
    forecasts={};identities={};cost=0.
    for scale,scaler in (("source",source_scale),("separate",target_scale)):
        if history:
            x=np.vstack((z,scaler.transform(np.asfortranarray(history))))
            labels=np.r_[y,answers];balance=np.r_[w,weights(np.asarray(answers),len(sy))]
        else:x,labels,balance=z,y,w
        for kernel in ("linear","rbf"):
            arm=f"svc-{kernel}-{scale}"
            model=SVC(C=1.,kernel=kernel,gamma="scale",probability=False,class_weight=None,
                      tol=1e-3,cache_size=128,max_iter=-1)
            start=process_time();model.fit(x,labels,sample_weight=balance);cost+=process_time()-start
            if model.fit_status_!=0:raise ValueError("SVC未完成拟合")
            pipeline=Pipeline([("scale",scaler),("model",model)])
            blob=backend.dump(pipeline);identity=sha256(blob).hexdigest()
            path=folder/(arm+".joblib");path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as out:out.write(blob)
            restored=backend.load(path.read_bytes())
            # 重载后核对实际输入；后续全部输出由落盘权重产生。
            query=source[:3] if not history else np.asarray(history[-3:])
            np.testing.assert_array_equal(restored.predict(query),pipeline.predict(query))
            forecasts[arm]=restored;identities[arm]={"sha256":identity,"gamma":float(model._gamma),
                "support":model.n_support_.tolist(),"file":path.relative_to(DEST).as_posix()}
    return forecasts,identities,cost


def evaluate(seed,pkg,batches,*,destination=DEST,protocol=PROTOCOL,arms=ARMS,
             fitter=fit_models,shared_cold_start=False,control_cold_start=None):
    """复用固定服务/反馈时序；另一个学习器实验可注入拟合器，默认仍复现原核实验。"""
    source=np.asfortranarray([r.values for r in pkg.train]);sy=np.asarray([r.label for r in pkg.train])
    if len(source)!=445:raise ValueError("源训练资料不同")
    folder=destination/f"seed-{seed}";prior=PRIOR/f"seed-{seed}"
    old=read(prior/"scores.json");old_requests=old["requests"]
    observer=ObserverErrors(GasFeedbackBank.names,6)
    history=[];answers=[];received=set();requests=[];cpu=0.
    ep=LocalAcceptance();ep.output=folder/"accepted"
    for j,(ids,x,votes,pool,fx,fv,pool_y,weight) in enumerate(batches):
        ix=sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:32] if j+1<len(batches) else []
        selected=[pool[i] for i in ix]
        if set(selected)&(received|set(ids)):raise ValueError("回标重复或混入服务")
        previous=read(prior/"accepted"/(old_requests[j]+".json"))["payload"]
        snap=read(prior/"snapshots"/(previous["snapshot"]+".json"))
        if digest(snap)!=previous["snapshot"]:raise ValueError("冻结线性参照快照身份不符")
        if previous["ids"]!=ids or snap["source_weight"]!=weight or snap["feedback_seen"]!=len(received):
            raise ValueError("参照输入、权重或回标时点不符")
        both=np.vstack((x,fx[ix]));all_votes={k:np.r_[votes[k],fv[k][ix]] for k in votes}
        pred={"observer":observer.predict(all_votes,"uniform")}
        for scale in ("source","separate"):
            pred["pooled-"+scale]=model_scores(snap["models"]["pool-domain-"+scale+"-c1"],both,6).argmax(1)+1
        pred["fixed-midway"]=pred["pooled-source" if j<5 else "pooled-separate"]
        for arm,p in pred.items():
            np.testing.assert_array_equal(p[:len(ids)],previous["predictions"][arm])
        if control_cold_start is not None:
            if control_cold_start!="observer":raise ValueError("未知冷启动来源")
            # 保留原控制，并把同一个推荐首轮交付提供给全部现场学习器控制。
            for a in ("pooled-source","pooled-separate","fixed-midway"):
                pred[a+"-linear-cold"]=pred[a].copy()
                if not history:pred[a]=pred["observer"].copy()
        if shared_cold_start and not history:
            # 新实验只替换有现场回标后的学习器，各家族共用原线性冷启动。
            identity={a:{"linear_cold_start":previous["snapshot"]} for a in arms}
            pred.update({a:pred["pooled-source"].copy() for a in arms})
        else:
            models,identity,cost=fitter(source,sy,history,answers,folder/"models"/f"round-{j+1:02}");cpu+=cost
            pred.update({a:np.asarray(m.predict(both)).reshape(-1) for a,m in models.items()})
        payload={"round":j+1,"ids":ids,"input_sha256":digest([ids,x.tolist()]),
            "feedback_seen":len(received),"source_weight":weight,"models":identity,
            "predictions":{a:p[:len(ids)].tolist() for a,p in pred.items()}}
        request=digest([digest(read(protocol)),seed,payload]);ack=ep.submit(request,payload)
        if not ack.get("accepted") or ack["request_id"]!=request:raise RuntimeError("未确认服务预测")
        requests.append(request)
        if ix:
            revealed=[pool_y[i] for i in ix]
            actual={"ids":selected,"labels":revealed}
            if actual!=read(prior/f"feedback-{j+1:02}.json"):raise ValueError("参照与新训练取得的反馈不同")
            atomic_json(folder/f"feedback-{j+1:02}.json",actual)
            observer.update(revealed,{k:fv[k][ix] for k in fv})
            history.extend(fx[ix].tolist());answers.extend(revealed)
        received.update(selected)
    logs=[read(ep.output/(r+".json"))["payload"] for r in requests]
    ids=[k for r in logs for k in r["ids"]];truth={r.sample_id:r.label for r in pkg.validation}
    if len(ids)!=414 or len(set(ids))!=414 or len(received)!=288:raise ValueError("服务或反馈数量不符")
    y=np.asarray([truth[k] for k in ids]);p={a:np.concatenate([r["predictions"][a] for r in logs]) for a in logs[0]["predictions"]}
    table={a:metrics(y,v) for a,v in p.items()}
    for a in ("observer","pooled-source","pooled-separate","fixed-midway"):
        key=a+"-linear-cold" if control_cold_start is not None and a!="observer" else a
        if abs(table[key]["macro"]-old["scores"][a]["macro"])>1e-12:raise ValueError("参照分数未复现")
    paired={a:{"rescues":int(np.sum((p[a]==y)&(p["fixed-midway"]!=y))),
               "harms":int(np.sum((p[a]!=y)&(p["fixed-midway"]==y)))} for a in arms}
    result={"seed":seed,"requests":requests,"scores":table,"paired":paired,"fit_cpu_seconds":cpu}
    atomic_json(folder/"scores.json",result)
    print(json.dumps({"seed":seed,"macro":{a:round(v["macro"],5) for a,v in table.items()}}),flush=True)
    return result


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已完成此对照，请复用")
        if sklearn.__version__!="1.7.2":raise ValueError("SVC实现版本未冻结")
        pkg=read_official_gas(ARCHIVE);batches=stream(pkg,GasFeedbackBank(pkg.train))
        runs=[evaluate(s,pkg,batches) for s in SEEDS]
        mean={a:{k:float(np.mean([r["scores"][a][k] for r in runs])) for k in ("macro","worst")}
              for a in runs[0]["scores"]}
        best=max(ARMS,key=lambda a:mean[a]["macro"]);default=runs[0]["scores"]
        eligible=[a for a in ARMS if mean[a]["macro"]>mean["fixed-midway"]["macro"]
                  and default[a]["macro"]>default["fixed-midway"]["macro"]
                  and mean[a]["worst"]>=mean["observer"]["worst"]-.02
                  and default[a]["worst"]>=default["observer"]["worst"]-.02]
        selected=max(eligible,key=lambda a:mean[a]["macro"]) if eligible else "observer"
        atomic_json(DEST/"scores.json",{"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),
            "means":mean,"best":best,"eligible":eligible,"selected":selected,"advances":bool(eligible),
            "fit_cpu_seconds":sum(r["fit_cpu_seconds"] for r in runs),"development_only":True,"seeds":list(SEEDS)})
        print(json.dumps({"best":best,"selected":selected,"advances":bool(eligible),"means":mean}),flush=True)


if __name__=="__main__":main()
