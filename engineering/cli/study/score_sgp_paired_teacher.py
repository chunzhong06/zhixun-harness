"""相同两台芯片与两次装机预算，重新实测配对输出校准和目标芯片独立监督。"""
from hashlib import md5,sha256
import json
import h5py
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score
from score_sgp_install_cycles import DEST as PRIOR,SOURCE,MD5,SETUPS,raw_part,labels,LocalStore,LocalAcceptance,atomic_json,digest
from harness.models.cycle_install import PAPER,balanced_f1
from harness.models.paired_cycle_teacher import PairedCycleTeacher,PRIMARY,TARGET,PAIRED,INVERSE

DEST=PRIOR.parent/"sgp-paired-teacher"
SELECT="按整混合物装机试验选择校正强度"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有相同训练比较")
        if md5(SOURCE.read_bytes()).hexdigest()!=MD5:raise ValueError("原始芯片包变化")
        prev=json.loads((PRIOR/"scores.json").read_bytes())
        receiver=LocalAcceptance();receiver.output=DEST/"accepted";events={};details={}
        with h5py.File(SOURCE) as f:
            ids,src=raw_part(f,"A","train");y=labels(f,"train",ids);threshold=float(np.median(y))
            for chip in "BC":
                install_id,all_groups=raw_part(f,chip,"train");sid,service=raw_part(f,chip,"test")
                if not np.array_equal(ids,install_id) or set(sid)&set(ids):raise ValueError("混合物边界变化")
                for size in SETUPS:
                    key=f"{chip}-{size}";parent=prev["requests"][key]
                    old=json.loads((PRIOR/"accepted"/(parent+".json")).read_bytes())["payload"]
                    if digest(old)!=parent or old["服务混合物"]!=sid.tolist():raise ValueError("服务或安装回执变化")
                    loc={int(v):i for i,v in enumerate(ids)}
                    ix=np.array([loc[k] for k in old["装机混合物"]]);tx=[all_groups[j] for j in ix]
                    folds=np.arange(len(ix))%5;alphas=(1.,15.,100.)
                    held={};final={};models={}
                    for alpha in alphas:
                        pred=np.empty(len(ix))
                        for k in np.unique(folds):
                            test=folds==k; keep=np.ones(len(src),bool);keep[ix[test]]=False
                            remap=np.cumsum(keep)-1
                            m=PairedCycleTeacher().fit([a for a,t in zip(src,keep) if t],y[keep],
                               [a for a,t in zip(tx,~test) if t],remap[ix[~test]],alpha)
                            pred[test]=m.predict([a for a,t in zip(tx,test) if t])[PRIMARY]
                        held[alpha]=(balanced_f1(y[ix],pred,threshold),-float(np.mean((y[ix]-pred)**2)))
                        m=PairedCycleTeacher().fit(src,y,tx,ix,alpha);models[alpha]=m
                        final[alpha]=m.predict(service)
                    selected=max(alphas,key=lambda a:held[a]);paper=np.asarray(old["实发浓度"][PAPER])
                    output={SELECT:final[selected][PRIMARY]}
                    for a,cols in final.items():
                        for arm,v in cols.items():
                            name=f"{arm}/强度{a:g}"
                            output[name]=v
                            output["与全局校准论文各半/"+name]=(v+paper)/2
                    output["安装选法与论文各半"]=(output[SELECT]+paper)/2
                    weights=DEST/(key+".joblib");joblib.dump({"模型":models[selected],"阈值":threshold},weights)
                    published=joblib.load(weights)
                    emitted=published["模型"].predict(service)[PRIMARY]
                    if published["阈值"]!=threshold or not np.array_equal(emitted,output[SELECT]):
                        raise ValueError("交付权重恢复后的预测改变")
                    output[SELECT]=emitted
                    classes={**old["实发分类"],**{k:(1+(v>threshold).astype(int)).tolist() for k,v in output.items()}}
                    payload={"实发分类":classes,"服务混合物":sid.tolist(),"采用强度":selected,"上轮回执":parent,
                             "实发浓度":{k:v.tolist() for k,v in output.items()},"权重":sha256(weights.read_bytes()).hexdigest()}
                    req=digest(payload)
                    if not receiver.submit(req,payload)["accepted"]:raise ValueError("服务未接收")
                    events[key]=req;details[key]={"强度":selected,"验证F1":held[selected][0]}
            yy=1+(labels(f,"test",sid)>threshold)
        scores={}
        for key,req in events.items():
            payload=json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
            if digest(payload)!=req:raise ValueError("回执改变")
            scores[key]={k:{"F1":float(f1_score(yy,v,average="macro")),"最弱类召回":float(recall_score(yy,v,average=None).min())}
                         for k,v in payload["实发分类"].items()}
        means={k:{m:float(np.mean([d[k][m] for d in scores.values()])) for m in ("F1","最弱类召回")} for k in scores[key]}
        controls=[prev["decision"]["固定"],*[k for k in output if "/强度" in k]]
        fixed=max(controls,key=lambda k:means[k]["F1"])
        paper_arm=max([prev["decision"]["论文特征对照"],*[k for k in output if k.startswith("与全局校准论文各半/")]],key=lambda k:means[k]["F1"])
        gap=means[SELECT]["F1"]-max(means[k]["F1"] for k in (fixed,paper_arm))
        d={"现场":SELECT,"固定":fixed,"论文特征对照":paper_arm,"现场F1":means[SELECT]["F1"],"固定F1":means[fixed]["F1"],
           "论文特征组合F1":means[paper_arm]["F1"],"差值":gap,"过初筛":gap>=.01 and means[SELECT]["最弱类召回"]>=means[fixed]["最弱类召回"]-.02}
        atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":scores,"requests":events,"各安装试验":details})
        print(json.dumps({"配对装机交付":d},ensure_ascii=False))


if __name__=="__main__":main()
