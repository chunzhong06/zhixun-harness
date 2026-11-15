"""在已见原始暴露上比较同一器件的读法，固定规则也获得所有合格工具。"""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.models.dsps_response import physical_views
from score_dsps_reference import SOURCE, DEST as PRIOR, LocalAcceptance, LocalStore, atomic_json, digest

DEST = PRIOR.parent / "dsps-response"


def read(path):
    return json.loads(path.read_bytes())


def acquire():
    meta = read(SOURCE/"source.json")
    hashes = {r["成员"]:r["SHA256"] for r in meta["已核验成员"]}
    with np.load(SOURCE/"development.npz",allow_pickle=False) as data:
        ids, groups, y, old_views = [data[k].copy() for k in ("ids","repetition","y","x")]
    bank = {}
    for i,key in enumerate(ids):
        path = SOURCE/"members"/key
        if sha256(path.read_bytes()).hexdigest()!=hashes[key]:
            raise ValueError("原件与封存身份不符")
        raw = loadmat(path,squeeze_me=True,struct_as_record=False)[path.stem]
        for device in (0,11):
            stages = [getattr(raw,k)[:,128*device:128*(device+1)].reshape(-1,16,8).transpose(0,2,1) for k in ("Bl","Tr","St")]
            views = physical_views(*stages)
            np.testing.assert_allclose(views["单负载稳态"],old_views[i,device],rtol=1e-11,atol=1e-10)
            for name,vec in views.items():
                bank.setdefault((name,device),[]).append(vec)
    return ids,groups,y,{k:np.stack(v) for k,v in bank.items()}


def models(bank, train, test, y):
    out, confident = {}, {}
    for name,device in bank:
        if device!=11:
            continue
        x = bank[name,11]
        for mode in ("目标线性","目标核","安装合训"):
            key = name+"/"+mode
            tx = x[train]
            labels = y[train]
            if mode=="安装合训":
                tx = np.vstack((tx,bank[name,0][train]))
                labels = np.tile(labels,2)
            clf = (SVC(C=4,gamma="scale",decision_function_shape="ovr") if mode=="目标核"
                   else LogisticRegression(C=1,max_iter=1000,random_state=17))
            pipe = make_pipeline(StandardScaler(),clf).fit(tx,labels)
            scores = pipe.decision_function(x[test])
            out[key] = pipe.predict(x[test])
            top = np.sort(scores,axis=1)[:,-2:]
            confident[key] = top[:,1] - top[:,0]
    return out,confident


def tally(truth,values):
    return {"F1":float(f1_score(truth,values,labels=[1,2,3],average="macro",zero_division=0)),
            "最差召回":float(recall_score(truth,values,labels=[1,2,3],average=None,zero_division=0).min()),
            "正确":int(np.sum(truth==values))}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已有此次原始响应对照")
        ids,group,y,bank = acquire()
        # 用作后续诊断与同信息固定方案的输入，只缓存已登记的开发重复，次序与原件登记一致。
        np.savez_compressed(DEST/"views.npz",ids=ids,y=y,group=group,
                            **{f"{name}_{dev}":v for (name,dev),v in bank.items()})
        receiver=LocalAcceptance()
        receiver.output=DEST/"accepted"
        previous=read(PRIOR/"scores.json")
        conditions=np.unique(group)
        requests=[]
        for rep,old_id in zip(conditions,previous["requests"],strict=True):
            train,test=group!=rep,group==rep
            predictions,margin=models(bank,train,test,y)
            names=[k for k in predictions if not k.startswith("干空气对照/")]
            correct, margins=[],[]
            for validation in conditions[conditions!=rep]:
                held=train & (group==validation)
                fold,conf=models(bank,train & ~held,held,y)
                correct.extend(np.column_stack([fold[k]==y[held] for k in names]))
                margins.extend(np.column_stack([conf[k] for k in names]))
            correct, margins=np.asarray(correct),np.asarray(margins)
            rate=correct.mean(axis=0)
            base=int(rate.argmax())
            decisions=np.column_stack([predictions[k] for k in names])
            q=np.column_stack([margin[k] for k in names])
            level=np.median(margins,axis=0)
            strong=margins>=level
            hi=(np.sum(correct*strong,axis=0)+4*rate)/(strong.sum(axis=0)+4)
            lo=(np.sum(correct*~strong,axis=0)+4*rate)/((~strong).sum(axis=0)+4)
            which=np.where(q>=level,hi,lo).argmax(axis=1)
            old=read(PRIOR/"accepted"/(old_id+".json"))["payload"]
            digest_old = digest([read(SOURCE/"source.json")["特征SHA256"],int(rep),old])
            if old["ids"]!=ids[test].tolist() or digest_old!=old_id:
                raise ValueError("旧论文与固定对照身份不符")
            out={**old["实际分类"],**{k:v.tolist() for k,v in predictions.items()}}
            out["按整体开发准确率"] = decisions[:,base].tolist()
            out["按当前分类把握"] = decisions[np.arange(len(which)),which].tolist()
            payload={"ids":ids[test].tolist(),"预测":out,"源端采用":names[base],"逐条采用":[names[v] for v in which]}
            key=digest([int(rep),payload])
            if not receiver.submit(key,payload)["accepted"]:
                raise ValueError("原始信号候选未交接")
            requests.append(key)
        all_y,all_out=[],{}
        for rep,key in zip(conditions,requests,strict=True):
            p=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),p])!=key or p["ids"]!=ids[group==rep].tolist():
                raise ValueError("实发暴露次序不符")
            all_y.extend(y[group==rep].tolist())
            for k,v in p["预测"].items():
                all_out.setdefault(k,[]).extend(v)
        table={k:tally(np.array(all_y),np.array(v)) for k,v in all_out.items()}
        options=("按当前分类把握","按整体开发准确率")
        fixed=[k for k in table if k not in (*options,"当前读数邻近选择","源端留整组选定") and not k.startswith("干空气对照")]
        best=max(fixed,key=lambda k:table[k]["F1"])
        chosen=max(options,key=lambda k:table[k]["F1"])
        delta=table[chosen]["F1"]-table[best]["F1"]
        decision={"最强固定":best,"候选":chosen,"开发增量":delta,"进入盲测":delta>=.01 and table[chosen]["最差召回"]>=table[best]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"scores":table,"requests":requests,"decision":decision})
        print(json.dumps({"成绩":table,"判定":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
