"""源端已知的浓度只用于拟合；每次服务从传感器曲线估计类别并将未知浓度消去。"""
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score

from score_dsps_response import DEST as PREVIOUS, read, tally
from score_dsps_reference import LocalAcceptance, LocalStore, atomic_json, digest

DEST = PREVIOUS.parent/"dsps-concentration"
VIEWS = ("全负载稳态","负载曲线稳态","去总幅度的温度曲线","暴露速度")


def classify(train, target, gas, dose, variant):
    scale = StandardScaler().fit(train)
    z, q = scale.transform(train),scale.transform(target)
    classes = np.unique(gas)
    levels = np.unique(dose)
    curves = np.array([[z[(gas==c)&(dose==v)].mean(axis=0) for v in levels] for c in classes])
    if variant=="类中心":
        centers = curves.mean(axis=1)
        distances = np.mean((q[:,None,:]-centers)**2,axis=-1)
    else:
        # 每个类别都有一整条可预测的浓度轨迹，线上按测量到该轨迹的距离判断。
        centers = curves[:,:-1]
        segment = curves[:,1:]-centers
        offset = q[:,None,None,:]-centers[None,:,:,:]
        alpha = np.sum(offset*segment[None,:,:,:],axis=-1) / np.maximum(1e-9,np.sum(segment**2,axis=-1))
        delta = offset - np.clip(alpha,0,1)[:,:,:,None]*segment
        distances = np.mean(delta**2,axis=-1).min(axis=2)
    return classes[np.argmin(distances,axis=1)]


def bank(views, train, test, gas, dose):
    out={}
    for name in VIEWS:
        old,new=views[name+"_0"],views[name+"_11"]
        for pooled in (False,True):
            x=new[train]
            ty,td=gas[train],dose[train]
            if pooled:
                x=np.vstack([x,old[train]])
                ty,td=np.tile(ty,2),np.tile(td,2)
            for variant in ("类中心","浓度曲线"):
                key="/".join((name,"安装合训" if pooled else "目标训练",variant))
                out[key]=classify(x,new[test],ty,td,variant)
    return out


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已完成该开发试验")
        with np.load(PREVIOUS/"views.npz",allow_pickle=False) as f:
            ids,gas,group = f["ids"],f["y"],f["group"]
            views={k:f[k] for name in VIEWS for k in (name+"_0",name+"_11")}
        dose=np.array([(int(Path(k).stem[1:])-1)*20 for k in ids])
        prior=read(PREVIOUS/"scores.json")
        receiver=LocalAcceptance()
        receiver.output=DEST/"accepted"
        events=[]
        for rep,parent in zip(np.unique(group),prior["requests"],strict=True):
            dev,held=group!=rep,group==rep
            pred=bank(views,dev,held,gas,dose)
            names=list(pred)
            for i in range(1,4):
                pred[f"固定报第{i}类"]=np.full(held.sum(),i)
            inners={k:[] for k in names}
            labels=[]
            for inner in np.unique(group[dev]):
                mask=group==inner
                learned=bank(views,dev&~mask,mask,gas,dose)
                labels.extend(gas[mask].tolist())
                for k in names:
                    inners[k].extend(learned[k].tolist())
            rates={k:f1_score(labels,v,labels=[1,2,3],average="macro",zero_division=0) for k,v in inners.items()}
            best=max(names,key=rates.get)
            pred["源端选曲线"]=pred[best]
            # 多个读法都可以投票，固定方也使用同一批读法的等权票。
            matrix=np.column_stack([pred[k] for k in names])
            for key,w in (("固定曲线票",np.ones(len(names))),("按安装留出加权",np.array([rates[k] for k in names]))):
                ballots=np.stack([((matrix==i)*w).sum(axis=1) for i in (1,2,3)],axis=1)
                pred[key]=1+ballots.argmax(axis=1)
            old=read(PREVIOUS/"accepted"/(parent+".json"))["payload"]
            if digest([int(rep),old])!=parent or old["ids"]!=ids[held].tolist():
                raise ValueError("源端真实比较样本未对齐")
            out={**old["预测"],**{k:v.tolist() for k,v in pred.items()}}
            payload={"ids":ids[held].tolist(),"预测":out,"安装选中":best}
            key=digest([int(rep),parent,payload])
            if not receiver.submit(key,payload)["accepted"]:
                raise ValueError("浓度路线未交接")
            events.append(key)
        truth,actual=[],{}
        for rep,parent,key in zip(np.unique(group),prior["requests"],events,strict=True):
            p=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),parent,p])!=key or p["ids"]!=ids[group==rep].tolist():
                raise ValueError("交接内容改变")
            truth.extend(gas[group==rep].tolist())
            for k,v in p["预测"].items():
                actual.setdefault(k,[]).extend(v)
        table={k:tally(np.array(truth),np.array(v)) for k,v in actual.items()}
        dynamic=("源端选曲线","按安装留出加权")
        excluded=(*dynamic,"按整体开发准确率","按当前分类把握","当前读数邻近选择","源端留整组选定")
        fixed=max((k for k in table if k not in excluded and not k.startswith("干空气对照")),key=lambda k:table[k]["F1"])
        best=max(dynamic,key=lambda k:table[k]["F1"])
        gap=table[best]["F1"]-table[fixed]["F1"]
        decision={"最强固定":fixed,"主候选":best,"开发差值":gap,
                  "进入独立":gap>=.01 and table[best]["最差召回"]>=table[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"requests":events,"scores":table,"decision":decision,
                                        "候选混淆矩阵":confusion_matrix(truth,actual[best],labels=[1,2,3]).tolist()})
        print(json.dumps({"成绩":table,"是否晋级":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
