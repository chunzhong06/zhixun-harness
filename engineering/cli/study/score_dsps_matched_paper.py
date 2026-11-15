"""把同一完整稳态读法与留组类别校准交给DS/PDS论文对照，逐次验证其额外贡献。"""
from hashlib import sha256
import json
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler

from verify_dsps_cycle_release import DEST as BASELINE, SOURCE, read, LocalStore, LocalAcceptance, atomic_json, digest, tally
from score_dsps_air_control import DEST as CONTROL
from harness.models.dsps_cycle_release import kernel, target_features

DEST=BASELINE.parent/"dsps-matched-paper"
PRIMARY="按当前校准误差选跨器件映射"


def scores(source,target,y,take,query,method):
    # 两种转换沿每个传感器的20个温度窗排列，不把交错通道当作同一传感器。
    old, new = source[take], target[take]
    before, after = StandardScaler().fit(old), StandardScaler().fit(new)
    oldz, newz = before.transform(old),after.transform(new)
    q=after.transform(query)
    if method=="DS":
        mapping=Ridge(alpha=1).fit(newz,oldz)
        mapped=mapping.predict(q)
    else:
        mapped=np.column_stack([Ridge(alpha=1).fit(newz[:,j:j+20],oldz[:,j:j+20]).predict(q[:,j:j+20])
                                for j in range(0,160,20)])
    model=kernel(oldz,y[take])
    return model.decision_function(mapped)


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("公平论文对照已发出")
        meta=read(SOURCE/"source.json")
        hashes={r["成员"]:r["SHA256"] for r in meta["已核验成员"]}
        with np.load(SOURCE/"development.npz") as z:ids,y,group=z["ids"],z["y"],z["repetition"]
        source=[];target=[]
        for uid in ids:
            path=SOURCE/"members"/uid
            if sha256(path.read_bytes()).hexdigest()!=hashes[uid]:raise ValueError("成员身份不符")
            m=loadmat(path,squeeze_me=True,struct_as_record=False)[path.stem]
            for row,dst in ((0,source),(11,target)):
                inputs=[getattr(m,k)[:,row*128:(row+1)*128].reshape(-1,16,8).transpose(0,2,1) for k in ("Bl","Tr","St")]
                dst.append(target_features(*inputs)[0].reshape(20,8).T.ravel())
        source,target=np.stack(source),np.stack(target)
        previous=read(BASELINE/"scores.json"); receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        truth=[];cols={};events=[]
        for rep,parent in zip(np.unique(group),previous["requests"],strict=True):
            tr=group!=rep;te=~tr
            old=read(BASELINE/"accepted"/(parent+".json"))["payload"]
            if digest([int(rep),old])!=parent or old["ids"]!=ids[te].tolist():raise ValueError("基准回执错误")
            base=np.asarray(old["预测"]["整周期留组类别映射"])
            choices={}
            for name in ("DS","PDS"):
                held=np.empty((tr.sum(),3))
                for g in np.unique(group[tr]):
                    ev=group==g
                    held[ev[tr]]=scores(source,target,y,tr&~ev,target[ev],name)
                fit=LogisticRegression(C=1,max_iter=500).fit(held,y[tr])
                choices[name]=fit.predict_proba(scores(source,target,y,tr,target[te],name))
            pred={k:1+v.argmax(1) for k,v in choices.items()}
            mean=(choices["DS"]+choices["PDS"])/2
            pred["两种论文映射固定各半"]=1+mean.argmax(1)
            pred["论文与原固定各半"]=1+(.5*mean+.5*np.eye(3)[base-1]).argmax(1)
            # 当前两种映射同意且置信度过0.70才允许换掉已校准的本机工具。
            agreed=(pred["DS"]==pred["PDS"])&(mean.max(1)>.7)
            pred[PRIMARY]=np.where(agreed,pred["DS"],base)
            item={"ids":ids[te].tolist(),"预测":{k:v.tolist() for k,v in pred.items()},"基础回执":parent}
            key=digest([int(rep),item]);events.append(key)
            if not receiver.submit(key,item)["accepted"]:raise ValueError("公平校准交付失败")
            sent=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),sent])!=key:raise ValueError("接收端身份改变")
            truth.extend(y[te])
            for k,v in sent["预测"].items():cols.setdefault(k,[]).extend(v)
        actual={k:tally(np.array(truth),np.array(v)) for k,v in cols.items()}
        scores_all={**previous["scores"],**actual}
        p=max([previous["decision"]["论文"],"DS","PDS","两种论文映射固定各半","论文与原固定各半"],key=lambda k:scores_all[k]["F1"])
        f=max([previous["decision"]["固定"],p],key=lambda k:scores_all[k]["F1"])
        gap=actual[PRIMARY]["F1"]-scores_all[f]["F1"]
        d={"主方案":PRIMARY,"固定":f,"论文":p,"最小领先":gap,"进独立":gap>=.01 and actual[PRIMARY]["最差召回"]>=scores_all[f]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"requests":events,"新增读法":actual,"decision":d})
        print(json.dumps({"同工具":actual,"判定":d},ensure_ascii=False))


if __name__=="__main__":main()
