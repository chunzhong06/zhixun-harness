"""用首次引入的气体阵列检验 H-LLM 分类流程，每批交付后才开放该批回标。"""
import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, PrivateAttr
from river import __version__ as river_version
from river.drift.binary import DDM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.contracts import digest
from harness.models.gas_feedback_bank import discriminate, fit_discriminant
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import majority
from harness.runtime.storage import atomic_json


OUT=ROOT/"results/experiments/local-defaults/whole-maintenance/new-arrays"
UPSTREAM=ROOT/"references/implementations/pauliusrauba/Self_Healing_ML/a48a8323e4f23c2e5fc1a0c06dc5605b34cc2825"
SOURCE=ROOT/"data/public/cernei-voc-v1"
EXPECTED={
    "acetone":"312373929f8fbef1d183fa5a5b75eb2c364466a57780949ad4a4ec875502ffdd",
    "ethanol":"5c85071cd174e8f70ffe200f0455b7959f529b50bf657b037606873a6f13abe3",
    "formaldehyde":"6b862b1120cda2f0359cb551b0b1e01b934e4c7655226b2a06b02e91c14230ee",
    "limonene":"107bb27f1eac0ed470fc62707f36b3c28c3f771dadb01ecae105c7468351c0e9",
    "toluene":"5aeef4d34f9086e049a8b7dcaa124f25173f81bc04fa5e96c3b6214ad48a50f4"}
CHANNELS=["MICS2714_1","MICS2714_2","MICS5524","GGS1430","TGS2620","MP901","TGS2610","Dart_alc","Dart_HCHO"]


def author_definition(filename, identity, symbol, namespace):
    path,=UPSTREAM.rglob(filename)
    data=path.read_bytes()
    if sha256(data).hexdigest()!=identity:
        raise ValueError("作者程序已改变")
    node,=(n for n in ast.parse(data).body if getattr(n,"name",None)==symbol)
    exec(compile(ast.Module([node],type_ignores=[]),str(path),"exec"),namespace)
    return namespace[symbol]


class AwaitingResponse(RuntimeError):
    pass


def author_loop(device, *, output=OUT, columns=CHANNELS, context="Identify why VOC sensor campaign classification may have degraded."):
    """独立现场的原始作者调用；请求目录和合法列随数据源切换，回执按请求摘要复用。"""
    from typing import Dict, Optional
    import re
    import hashlib
    author=author_definition("HLLM.py","d5e742ae970845b48baed4530dbf755c3ae7470448770a0892be44d04ccfb0a0","H_LLM",
                             dict(AzureOpenAI=Any,BaseModel=BaseModel,PrivateAttr=PrivateAttr,Any=Any,Dict=Dict,
                                  Optional=Optional,pd=pd,np=np,ast=ast,hashlib=hashlib,re=re,accuracy_score=accuracy_score))
    log=[]
    class Transport(author):
        # 仅替换网络传输。所有提示、缓存和训练排名都来自固定版本的作者方法。
        def _get_llm_response(self,input_text,system_message=None,metadata_output=False,modelid=None):
            request={"input_text":input_text,"system_message":system_message,"device":device,
                     "stage":len(log)+1,"config":self.config}
            key=digest(request)
            atomic_json(output/device/"requests"/(key+".json"),request)
            log.append(key)
            response=output/device/"responses"/(key+".json")
            if not response.is_file():
                raise AwaitingResponse(key)
            result=json.loads(response.read_bytes())
            if (result["request"]!=key or not result.get("model") or not result.get("text")
                    or not result.get("driver_receipt")):
                raise ValueError("缺少对应请求的真实模型回执")
            if "Return ONLY what is required" in input_text:
                # 上游使用 pandas.query。仅开放本任务所需的数值比较逻辑，禁止把模型文本当任意代码执行。
                queries=ast.literal_eval(result["text"])
                if not isinstance(queries,list) or not 1<=len(queries)<=self.n:
                    raise ValueError("查询数超出冻结上限")
                allowed=(ast.Expression,ast.Compare,ast.Name,ast.Load,ast.Constant,ast.BoolOp,ast.And,ast.Or,
                         ast.BinOp,ast.BitAnd,ast.BitOr,ast.UnaryOp,ast.USub,ast.UAdd,ast.Not,
                         ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq)
                for q in queries:
                    nodes=list(ast.walk(ast.parse(q,mode="eval")))
                    if (any(not isinstance(n,allowed) for n in nodes) or any(n.id not in columns for n in nodes if isinstance(n,ast.Name))
                            or any(type(n.value) not in (int,float) for n in nodes if isinstance(n,ast.Constant))):
                        raise ValueError("条件需由登记的测量列与数值比较组成")
            return result["text"]
    h=Transport(llm=None,config={"seed":0,"temperature":0},context=context)
    method=author_definition("study_2.py","c56695ca92297f5b338af2d8c08dd9f49af71f5de474c925c8daa4ba10b70a70",
                             "adaptive_batch_self_healing",dict(np=np,pd=pd,accuracy_score=accuracy_score,tqdm=lambda iterable,**kw:iterable))
    return h,method,log


def read_arrays():
    features,groups,labels,cycles={},defaultdict(list),{},{}
    for label,(gas,expected) in enumerate(EXPECTED.items(),1):
        f=SOURCE/f"VOC_sensors_{gas}.csv"
        if sha256(f.read_bytes()).hexdigest()!=expected:
            raise ValueError("传感器数据与发布记录不同")
        frame=pd.read_csv(f)
        exposed=frame.loc[frame.Phase=="Exposure"]
        blocks=sorted(exposed.Cycle.unique())
        half=len(blocks)//2
        for idx,c in enumerate(blocks):
            rows=exposed.loc[exposed.Cycle==c].sort_values("Time",kind="stable")
            for device in ([1] if idx<half else [2,3,4,5]):
                x=rows[[f"{s}_{device}" for s in CHANNELS]].to_numpy(float)
                if not np.isfinite(x).all() or len(x)<5:
                    raise ValueError("完整暴露段缺少实测")
                for bin_ix,block in enumerate(np.array_split(x,5)):
                    # 观测身份是预先定义的采集坐标摘要，模型特征只包含九个实际通道。
                    k=digest([gas,int(c),device,bin_ix])
                    features[k]=block.mean(0)
                    labels[k]=label
                    cycles[k]=[gas,int(c)]
                    j=-1 if device==1 else (idx-half)*5+bin_ix
                    groups[device,j].append(k)
    return pd.DataFrame.from_dict(features,orient="index",columns=CHANNELS),pd.Series(labels),groups,cycles


class IssuedForest:
    """作者脚本中的 fit/predict 经此代理逐轮保存，模型只接收相应批次的九个测量。"""
    def __init__(self, keys,folder):
        self.keys=keys
        self.folder=folder
        self.tree=RandomForestClassifier(n_estimators=1,random_state=0,n_jobs=1)
        self.events=[]
        self.fit_count=0
        self.fit_attempts=0
        self.wall=0.

    def fit(self,x,y):
        tic=perf_counter()
        self.fit_attempts += 1
        try:
            self.tree.fit(x,y)
            self.fit_count+=1
        finally:
            self.wall+=perf_counter()-tic
        return self

    def predict(self,x):
        predicted=self.tree.predict(x)
        n=len(self.events)
        if n<len(self.keys) and x.index.tolist()==self.keys[n]:
            event={"keys":self.keys[n],"predictions":predicted.tolist(),"fits_before":self.fit_count}
            atomic_json(self.folder/f"round-{n+1:02}.json",event)
            self.events.append(event)
        return predicted


def metrics(y,values,cycles):
    classes=[1,2,3,4,5]
    cycle_groups=defaultdict(list)
    for i,key in enumerate(y.index):
        cycle_groups[tuple(cycles[key])].append(i)
    return {"macro_f1":float(f1_score(y,values,labels=classes,average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(y,values)),
            "worst_recall":float(min(recall_score(y,values,labels=classes,average=None,zero_division=0))),
            "cycle_errors":{str(k):int(np.sum(y.iloc[ix].to_numpy()!=np.asarray(values)[ix])) for k,ix in cycle_groups.items()}}


def run():
    if river_version!="0.21.2":
        raise ValueError("作者 DDM 必须使用 0.21.2")
    x,y,groups,cycles=read_arrays()
    train=groups[1,-1]; xs,ys=x.loc[train],y.loc[train]
    forest=RandomForestClassifier(n_estimators=1,random_state=0,n_jobs=1).fit(xs,ys)
    logistic=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=1000,random_state=17)).fit(xs,ys)
    rf=RandomForestClassifier(n_estimators=100,max_depth=6,random_state=17,class_weight="balanced",n_jobs=1).fit(xs,ys)
    svm=make_pipeline(StandardScaler(),SVC(C=1.,class_weight="balanced")).fit(xs,ys)
    lda=fit_discriminant(xs.to_numpy(),ys.to_numpy())
    result={}
    for device in range(2,6):
        unit=f"array-{device}"
        folder=OUT/unit
        keys=[groups[device,j] for j in range(10)]
        ids=[k for batch in keys for k in batch]
        if set(map(tuple,(cycles[k] for k in train))) & set(map(tuple,(cycles[k] for k in ids))) or len(ids)!=50:
            raise ValueError("源与目标暴露周期重叠或测量缺失")
        batches=[x.loc[ix] for ix in keys]
        def votes(v):
            return {"current":forest.predict(v),"logistic":logistic.predict(v),"forest":rf.predict(v),
                    "lda":discriminate(lda,v.to_numpy()),"svm":svm.predict(v)}
        obs=ObserverErrors(tuple(votes(xs)),5)
        predicted=defaultdict(list)
        for j,(batch,ix) in enumerate(zip(batches,keys),1):
            tools=votes(batch)
            p={**tools,"project-transfer":obs.predict(tools,"uniform"),"shared-fixed":majority(tools,5)}
            record={"keys":ix,"predictions":{a:v.tolist() for a,v in p.items()},"known":int(obs.counts[0].sum())}
            atomic_json(folder/"project"/f"round-{j:02}.json",record)
            for a,v in record["predictions"].items():
                predicted[a].extend(v)
            if j<10:
                obs.update(y.loc[ix],tools)
        h,method,calls=author_loop(unit)
        model=IssuedForest(keys,folder/"paper")
        status="complete"
        pending=None
        try:
            method(h,model,xs,ys,batches,[y.loc[ix] for ix in keys],None,DDM(drift_threshold=.8,warm_start=3),0,0)
        except AwaitingResponse as e:
            status,pending="waiting_driver",str(e)
        all_y=y.loc[ids]
        table={a:metrics(all_y,v,cycles) for a,v in predicted.items()}
        if status=="complete":
            if len(model.events)!=10:
                raise ValueError("论文实际发放轮数未齐全")
            obs_paper=ObserverErrors(tuple(votes(xs)),5)
            raw,adaptive=[],[]
            for j,(e,b,ix) in enumerate(zip(model.events,batches,keys)):
                tools=votes(b)
                tools["current"]=np.asarray(e["predictions"])
                raw.extend(e["predictions"])
                v=obs_paper.predict(tools,"uniform")
                adaptive.extend(v.tolist())
                if j<9:
                    obs_paper.update(y.loc[ix],tools)
            table["author-study2"]=metrics(all_y,raw,cycles)
            table["author-with-observer"]=metrics(all_y,adaptive,cycles)
        result[unit]={"status":status,"pending_request":pending,"author_scored_rounds":len(model.events),
                      "all_author_requests":calls,"author_fits":model.fit_count,"author_fit_attempts":model.fit_attempts,
                      "author_fit_seconds":model.wall,
                      "train_rows":len(train),"service_rows":len(ids),"physical_test_cycles":len(set(map(tuple,(cycles[k] for k in ids)))),
                      "arms":table}
        atomic_json(folder/"scores.json",result[unit])
        print(unit,status,len(model.events),{a:round(v["macro_f1"],4) for a,v in table.items()},flush=True)
    atomic_json(OUT/"scores.json",result)


if __name__=="__main__":
    run()
