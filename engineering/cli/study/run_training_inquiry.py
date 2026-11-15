"""在两套真实资料上让同一提案器实际训练，交齐各组预测后才计算服务误差。"""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

from score_engine_full_condition import ROOT, _trainer, official, read
from harness.contracts import digest
from harness.methods.training_inquiry import LabeledBatch, Trial, investigate
from harness.models.condition_processing import ConditionStandardScaler
from harness.models.engine_training_inquiry import EngineTrainingInquiry,predict
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json

BASE = ROOT/"results/experiments/local-defaults/whole-maintenance/training-inquiry"
RAW = ROOT/"data/public/uci-gas-turbine/uci-551.zip"
SHA = "55fdb1acc25f05bd9c77aac285c424e9154ffb0dbf1bbadb56b69a1142231295"


def issue(folder,stage,ids,predictions):
    payload = {"ids":ids,"predictions":{k:np.asarray(v).tolist() for k,v in predictions.items()}}
    if len(set(ids))!=len(ids) or any(len(v)!=len(ids) or not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("服务清单与预测尺寸或取值不符")
    receiver = LocalAcceptance()
    receiver.output = folder/"accepted"
    request = digest([stage,payload])
    ack = receiver.submit(request,payload)
    if not ack["accepted"] or ack["request_id"]!=request:
        raise RuntimeError("研究候选交接未确认")
    return request


def evaluate(folder,stage,request,labels,groups):
    payload = read(folder/"accepted"/(request+".json"))["payload"]
    if digest([stage,payload])!=request:
        raise ValueError("计分与原交接不同")
    batch = LabeledBatch(np.empty((len(labels),0)),labels,tuple(groups))
    return {k:batch.score(v) for k,v in payload["predictions"].items()}


def engine(folder,trainer):
    pkg,checksum,batches = official("engine")
    first = batches[0]
    ids = {r.sample_id for r in first.samples if r.role=="fit"}
    source = [r for r in pkg.train if r.sample_id in ids]
    explore,confirm = ([r for r in first.samples if r.role==role and r.available_at==first.ordinal]
                       for role in ("explore","adopt"))
    sites = [r for b in batches for r in b.samples if r.role=="serve" and r.available_at==b.ordinal]
    if any(r.label is not None for r in sites) or ({r.unit_id for r in sites}&{r.unit_id for r in explore+confirm}):
        raise ValueError("探索、确认或服务资料交叉")
    pred,trace,times = {},{},{}
    for name in ("adaptive","fixed-order"):
        tic=process_time()
        obj=EngineTrainingInquiry.from_public_engine(pkg.task,source,pkg,checksum,folder/name,trainer,explore,confirm,
                                                    fixed=name=="fixed-order")
        times[name]=process_time()-tic
        pred[name]=predict(obj.bundle,obj.repair.transform([r.values for r in sites]))
        trace[name]=obj.trace
    # 两个固定交付参照均已在相同的正式源包、相同输入上训练；按实际已发出的逐行预测比较。
    for arm,run in (("fixed-reference","engine-full-condition-integrated"),("incumbent","engine-continued-integrated")):
        root=BASE.parent/run/"engine"
        issued={}
        for event in read(root/"run.json")["accepted"]:
            receipt=read(root/"accepted"/(event["request"]+".json"))
            assert digest([read(root/"run.json")["identity"],receipt["payload"]])==event["request"]
            issued.update(receipt["payload"]["predictions"])
        pred[arm]=np.array([issued[r.sample_id] for r in sites])
    ids=[r.sample_id for r in sites]
    req=issue(folder,"public",ids,pred)
    truth={r.sample_id:r.label for r in pkg.validation}
    metrics=evaluate(folder,"public",req,[truth[k] for k in ids],[r.unit_id for r in sites])
    return {"trace":trace,"cpu_seconds":times,"request":req,"scores":metrics}


def turbine(folder,trainer):
    if sha256(RAW.read_bytes()).hexdigest()!=SHA:
        raise ValueError("UCI原件身份不同")
    features=("AT","AP","AH","AFDP","GTEP","TIT","TAT","TEY","CDP")
    def columns(pack,year,fields):
        # 只解析被请求的列。未来年份的CO列在两年预测全部完成后才解析。
        blob=pack.read(f"gt_{year}.csv")
        names=blob.splitlines()[0].decode().strip().split(",")
        return np.loadtxt(BytesIO(blob),delimiter=",",skiprows=1,usecols=[names.index(k) for k in fields],ndmin=2)
    with ZipFile(RAW) as pack:
        train=np.vstack([columns(pack,y,features+("CO",)) for y in (2011,2012)])
        x,y=train[:,:-1],train[:,-1]
        edges=np.quantile(x[:,7],[1/3,2/3])
        def grouped(v):
            return np.column_stack((v,np.searchsorted(edges,v[:,7])))
        cond=ConditionStandardScaler((9,),(0,)).fit(grouped(x))
        def view(v,reading):
            return v if reading=="raw" else cond.transform(grouped(v))[:,:9]
        val=columns(pack,2013,features+("CO",))
        mid=len(val)//2
        def part(rows,prefix):
            return LabeledBatch(rows[:,:-1],rows[:,-1],tuple(f"{prefix}-{j//168}" for j in range(len(rows))))
        explore,confirm=part(val[:mid],"first"),part(val[mid:],"last")
        index=np.arange(0,len(x),16)
        # 同一个赛方模型类在此独立初始化为九输入回归器，参数只由训练年份建立。
        center=StandardScaler().fit(view(x[index],"reference"))
        def build(trial):
            which=index if trial.records=="sample" else slice(None)
            z=view(x[which],trial.reading)
            if trial.learner=="native":
                model=trainer.Model("regression",list(features),[],x,y,hidden=32,seed=42,learning_rate=.002,batch_size=256)
                if trial.reading=="reference":
                    z=center.transform(z)*model.scale+model.mean
                model.fit(z,y[which],10)
            else:
                model=CatBoostRegressor(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1,random_seed=17,
                                        thread_count=1,verbose=False,allow_writing_files=False).fit(z,y[which])
            return trial,model
        def forecast(fitted,data):
            trial,model=fitted
            z=view(data,trial.reading)
            if trial.learner=="native" and trial.reading=="reference":
                z=center.transform(z)*model.scale+model.mean
            values=model.predict(z) if trial.learner=="native" else model.predict(z,thread_count=1)
            return np.maximum(0,values)
        models={"incumbent":build(Trial("native","raw","full"))}
        traces,times={},{}
        for name in ("adaptive","fixed-order"):
            tic=process_time()
            models[name],traces[name]=investigate(build,forecast,models["incumbent"],explore,confirm,len(index),len(x),
                                                 fixed=name=="fixed-order")
            times[name]=process_time()-tic
        for tool in ("native","tree"):
            for reading in ("raw","reference"):
                name=f"fixed-{tool}-{reading}"
                models[name]=models["incumbent"] if (tool,reading)==("native","raw") else build(Trial(tool,reading,"full"))
        requests={}
        for year in (2014,2015):
            target=columns(pack,year,features)
            requests[str(year)]=issue(folder,str(year),[f"{year}/{j}" for j in range(len(target))],
                                      {n:forecast(m,target) for n,m in models.items()})
        scores={}
        for yr,req in requests.items():
            ytest=columns(pack,int(yr),("CO",)).ravel()
            scores[yr]=evaluate(folder,yr,req,ytest,[j//168 for j in range(len(ytest))])
    return {"trace":traces,"cpu_seconds":times,"requests":requests,"scores":scores}


def main(site):
    folder=BASE/site
    with LocalStore(folder/"lock").exclusive(),TemporaryDirectory(dir=folder) as tmp:
        if (folder/"scores.json").exists():
            raise ValueError("本次实验已有评分回执")
        archive,=(ROOT/"data/official/engine").glob("*.zip")
        with ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
            out=(engine if site=="engine" else turbine)(folder,trainer)
        atomic_json(folder/"scores.json",out)
        print(json.dumps({"评分":out["scores"],"构建CPU秒":out["cpu_seconds"],
                          "独立确认采用":{k:t["accepted"] for k,t in out["trace"].items()}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("site",choices=("engine","turbine"))
    main(parser.parse_args().site)
