"""固定原公开服务流，对照连续概率、硬票及现场重训；先交付再回标。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np
from scipy.optimize import check_grad
from sklearn.metrics import f1_score,recall_score

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"engineering/src"),str(ROOT)]
from harness.contracts import digest
from harness.data.official_gas import read_official_gas
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _trainer
from harness.models.probability_fusion import ResidualCalibration,evidence,objective,observer_probabilities,softmax
from harness.models.site_training import SiteTraining,RECIPES
from harness.models.field_classification import model_scores
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json

BASE=ROOT/"results/experiments/local-defaults"
DEST=BASE/"whole-maintenance/gas-probability-fusion"
SOURCE=BASE/"selected/gas"
PROTOCOL=ROOT/"engineering/configs/experiments/gas-probability-fusion.json"
ARCHIVE=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip"
SEEDS=(17,47,89,*range(101,111))
CANDIDATES=("arithmetic-source","arithmetic-separate","geometric-source","geometric-separate","residual-soft","residual-hard")


def read(path):
    return json.loads(path.read_bytes())


def mathematics():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(7,6));base=rng.normal(size=(7,3));y=np.array([0,1,2,0,1,2,0])
    theta=rng.normal(scale=.1,size=21)
    weights=np.full(7,1/7)
    err=check_grad(lambda v:objective(v,x,base,y,weights)[0],lambda v:objective(v,x,base,y,weights)[1],theta)
    if err>1e-5:
        raise ValueError("概率残差目标的解析梯度不符")
    probs=softmax(rng.normal(size=(21,3))).reshape(7,3,3)
    learner=ResidualCalibration(3)
    before=learner.predict(probs)
    np.testing.assert_allclose(before,softmax(evidence(probs)[1]),rtol=0,atol=1e-12)
    learner.update(probs,y+1)
    after=learner.predict(probs)
    resumed=ResidualCalibration(3,state=learner.state()).predict(probs)
    np.testing.assert_allclose(after,resumed,rtol=0,atol=1e-12)
    return {"gradient_absolute_error":float(err),"cold_geometric_identity":True,"restore_matches":True}


def stream(package, bank):
    state=read(SOURCE/"state.json")["samples"]
    source=read(SOURCE/"run.json")
    artifacts=ModelArtifacts(SOURCE/"models")
    result=[]
    with TemporaryDirectory(dir=DEST) as tmp,ZipFile(ARCHIVE) as archive,_trainer(archive,Path(tmp),"A") as trainer:
        path=Path(tmp)/"current.npz"
        for j,row in enumerate(source["rounds"]):
            header,blob=artifacts.read(row["model_artifact"])
            path.write_bytes(blob)
            model=pair_model(trainer.Model).load(path)
            roles={role:sorted((r for r in state.values() if r["available_at"]==j and r["role"]==role),key=lambda r:r["sample_id"])
                   for role in ("explore","adopt","serve")}
            serve=roles["serve"];feedback=roles["explore"]+roles["adopt"]
            ids=[r["sample_id"] for r in serve]
            x=np.array([r["values"] for r in serve])
            pool=[r["sample_id"] for r in feedback]
            fx=np.array([r["values"] for r in feedback])
            votes=bank.predict(model,x);fv=bank.predict(model,fx)
            old=read(SOURCE/row["submitted"])["payload"]["predictions"]
            if dict(zip(ids,votes["current"].tolist(),strict=True))!=old or any(r["label"] is not None for r in serve):
                raise ValueError("未复现原服务权重，或服务输入含答案")
            result.append((ids,x,votes,pool,fx,fv,[r["label"] for r in feedback],header["sha256"]))
    return result


def metrics(y,p):
    recall=recall_score(y,p,labels=range(1,7),average=None,zero_division=0)
    return {"macro":float(f1_score(y,p,labels=range(1,7),average="macro",zero_division=0)),
            "recall":recall.tolist(),"worst":float(recall.min())}


def evaluate(seed, package, batches):
    source=np.array([r.values for r in package.train]);labels=np.array([r.label for r in package.train])
    field=SiteTraining(source,labels,GasFeedbackBank.names,6,"pooled-source")
    soft=ResidualCalibration(6);hard=ResidualCalibration(6,hard=True)
    endpoint=LocalAcceptance();folder=DEST/f"seed-{seed}";endpoint.output=folder/"accepted"
    requests=[];received=set()
    for j,(ids,x,votes,pool,fx,fv,feedback_y,weight) in enumerate(batches):
        ix=sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:32] if j+1<len(batches) else []
        selected=[pool[i] for i in ix]
        if set(selected)&(received|set(ids)):
            raise ValueError("反馈重复，或与服务重叠")
        both=np.vstack((x,fx[ix]));combined={k:np.r_[votes[k],fv[k][ix]] for k in votes}
        p0=observer_probabilities(field.observer,combined)
        if (p0.argmax(1)+1).tolist()!=field.observer.predict(combined,"uniform").tolist():
            raise ValueError("连续混淆分数的最大类未复现旧校准")
        models=field.models()
        p={a:model_scores(record,both,6) for a,record in models.items()}
        prob=np.stack((p0,p["pooled-source"],p["pooled-separate"]),axis=1)
        result={"observer":p0.argmax(1)+1,**{k:v.argmax(1)+1 for k,v in p.items()}}
        result["fixed-midway"]=result["pooled-source" if j<5 else "pooled-separate"]
        for suffix in ("source","separate"):
            q=p["pooled-"+suffix]
            result["arithmetic-"+suffix]=(p0+q).argmax(1)+1
            result["geometric-"+suffix]=(np.log(np.clip(p0,1e-6,1))+np.log(np.clip(q,1e-6,1))).argmax(1)+1
        result["residual-soft"]=soft.predict(prob).argmax(1)+1
        result["residual-hard"]=hard.predict(prob).argmax(1)+1
        payload={"round":j+1,"ids":ids,"source_weight":weight,"feedback_seen":len(received),
                 "predictions":{k:v[:len(ids)].tolist() for k,v in result.items()},
                 "probability_evidence":prob[:len(ids)].tolist()}
        request=digest([digest(read(PROTOCOL)),seed,payload]);ack=endpoint.submit(request,payload)
        if ack.get("request_id")!=request or not ack.get("accepted"):
            raise RuntimeError("服务输出未获确认")
        requests.append(request)
        if ix:
            y=np.asarray([feedback_y[i] for i in ix],dtype=int)
            pre={a:result[a][len(ids):] for a in RECIPES}
            fp={k:np.asarray(v)[ix] for k,v in fv.items()}
            soft.update(prob[len(ids):],y);hard.update(prob[len(ids):],y)
            field.update(fx[ix],fp,y,pre)
            atomic_json(folder/f"feedback-{j+1:02}.json",{"ids":selected,"labels":y.tolist(),"probabilities_before_reveal":prob[len(ids):].tolist()})
        received.update(selected)
    logs=[read(endpoint.output/(k+".json"))["payload"] for k in requests]
    ids=[k for row in logs for k in row["ids"]]
    if len(ids)!=len(set(ids)) or len(ids)!=414 or len(received)!=288:
        raise ValueError("公开服务或回标数量与原协议不符")
    truth={r.sample_id:r.label for r in package.validation};y=np.array([truth[k] for k in ids])
    pred={a:np.concatenate([r["predictions"][a] for r in logs]) for a in logs[0]["predictions"]}
    table={a:metrics(y,v) for a,v in pred.items()}
    # 固定配方独立复算，必须复现同一训练/回标条件下的已封存强参照。
    reference=read(BASE/"whole-maintenance/site-training/draws/official/scores.json")
    fixed=next(r for r in reference["draws"] if r["seed"]==seed)["scores"]
    for arm in (*RECIPES,"fixed-midway"):
        if abs(fixed[arm]["macro"]-table[arm]["macro"])>1e-12:
            raise ValueError(f"seed{seed}固定配方{arm}未复现：{table[arm]['macro']} vs {fixed[arm]['macro']}")
    observer=read(BASE/"gas-complementary-feedback/observer-errors/public/scores.json")["arms"][f"uniform-{seed}"]
    if abs(observer["pooled"]-table["observer"]["macro"])>1e-12:
        raise ValueError("原公开校准器未逐条按同一概率最大类复现")
    best=max((*RECIPES,"fixed-midway"),key=lambda a:table[a]["macro"])
    diagnostics={a:{"rescued_vs_observer":int(np.sum((v==y)&(pred["observer"]!=y))),
                    "harmed_vs_observer":int(np.sum((v!=y)&(pred["observer"]==y))),
                    "rescued_vs_fixed":int(np.sum((v==y)&(pred[best]!=y))),
                    "harmed_vs_fixed":int(np.sum((v!=y)&(pred[best]==y)))} for a,v in pred.items() if a in CANDIDATES}
    output={"seed":seed,"requests":requests,"scores":table,"best_fixed":best,"feedback_used":len(received),
            "paired":diagnostics,"meta_cpu_seconds":soft.cpu+hard.cpu,"field_cpu_seconds":field.cpu}
    atomic_json(folder/"scores.json",output)
    print(json.dumps({"seed":seed,"macro":{a:round(r["macro"],5) for a,r in table.items()}}),flush=True)
    return output


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS));args=parser.parse_args()
    with LocalStore(DEST/"lock").exclusive():
        atomic_json(DEST/"mathematics.json",mathematics())
        if (DEST/"scores.json").exists():raise ValueError("已有完整对照，请复用结果")
        package=read_official_gas(ARCHIVE)
        bank=GasFeedbackBank(package.train)
        batches=stream(package,bank)
        runs=[evaluate(s,package,batches) for s in args.seeds]
        mean={a:{"macro":float(np.mean([r["scores"][a]["macro"] for r in runs])),
                 "worst":float(np.mean([r["scores"][a]["worst"] for r in runs]))} for a in runs[0]["scores"]}
        fixed=max((*RECIPES,"fixed-midway"),key=lambda a:mean[a]["macro"])
        selected=max(CANDIDATES,key=lambda a:mean[a]["macro"])
        base=next((r for r in runs if r["seed"]==17),None)
        passes=mean[selected]["macro"]>mean[fixed]["macro"] and mean[selected]["worst"]>=mean["observer"]["worst"]-.02
        if base:
            passes=passes and base["scores"][selected]["macro"]>base["scores"]["observer"]["macro"] and base["scores"][selected]["worst"]>=base["scores"]["observer"]["worst"]-.02
        atomic_json(DEST/"scores.json",{"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"means":mean,"selected":selected,
                                       "best_fixed":fixed,"advances":bool(passes),"seeds":args.seeds,"development_only":True})
        print(json.dumps({"mean":mean,"selected":selected,"best_fixed":fixed,"advances":bool(passes)}),flush=True)


if __name__=="__main__":main()
