"""冻结规则后的重庆电子鼻迁移比较。目标批只从已经返回的指定校准行学习。"""
from hashlib import sha256
from io import BytesIO
import subprocess

import numpy as np
from catboost import CatBoostClassifier
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,ROOT,atomic_json,digest,lda,read
from harness.models.feedback_routing import apply_choices
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import VoteFeedbackRouter,majority
from score_guarded_gas import SEEDS


def read_split(archive,group,stage):
    blob=subprocess.check_output(["tar","-xOf",str(archive),f"{group}/data_lab_{stage}.mat"])
    data=loadmat(BytesIO(blob))
    suffix="" if stage=="before" else "afterdrift"
    x=data["data"+suffix]
    y=data["numlab"+suffix].ravel().astype(int)
    if len(x)!=len(y) or x.shape[1]!=37 or not np.isfinite(x).all():
        raise ValueError("原始测量形状或数值有误")
    keys=[sha256(row.astype("<f8").tobytes()).hexdigest() for row in x]
    groups={}
    for j,k in enumerate(keys):
        groups.setdefault(k,[]).append(j)
    conflicting=[k for k,ix in groups.items() if len(set(y[ix]))!=1]
    selected=[ix[0] for k,ix in groups.items() if k not in conflicting]
    return x[selected],y[selected],[keys[j] for j in selected],{"rows":len(y),"unique":len(groups),"conflicting_keys":conflicting,"retained":len(selected)}


def fixed(x,y,target):
    scale=StandardScaler().fit(x)
    z=scale.transform(x)
    q=scale.transform(target)
    weights=len(y)/(len(set(y))*np.bincount(y)[y])
    forest=RandomForestClassifier(n_estimators=100,max_depth=6,n_jobs=1,random_state=17)
    boost=CatBoostClassifier(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1,thread_count=1,
                             random_seed=17,verbose=False,allow_writing_files=False)
    logistic=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=17)
    svc=SVC(C=1.,gamma="scale",class_weight="balanced")
    return {"current":logistic.fit(z,y).predict(q),"forest":forest.fit(x,y,sample_weight=weights).predict(target),
            "catboost":boost.fit(x,y,sample_weight=weights).predict(target,thread_count=1).ravel().astype(int),
            "lda":lda(x,target,y),"svm":svc.fit(z,y).predict(q)}


def study(group,name):
    archive=ROOT/"data/public/chongqing-enose/drift.rar"
    if sha256(archive.read_bytes()).hexdigest()!="8a2c6646fa2f63e9048412895abcf0e1dde6924ebb1293292cf9582fd3c4de67":
        raise ValueError("作者原包检验失败")
    x,y,train,before=read_split(archive,group,"before")
    query,truth,keys,after=read_split(archive,group,"after")
    if set(train)&set(keys) or set(y)!=set(range(1,15)) or set(truth)!=set(y):
        raise ValueError("前后批测量重合或类别不匹配")
    if any(np.array_equal(x[:,j],y) or np.array_equal(x[:,j],np.arange(len(x))) for j in range(x.shape[1])):
        raise ValueError("输入含明显标签或行号")
    p=fixed(x,y,query)
    base=OUT/"observer-errors/chongqing"/name
    path=base/"fixed.json"
    atomic_json(path,{"train_rows":len(x),"before_audit":before,"after_audit":after,"service_keys":keys,"source_fit_keys":train,
                      "predictions":{k:v.tolist() for k,v in p.items()}})
    p={k:np.asarray(v) for k,v in read(path)["predictions"].items()}
    groups={};predictions={}
    for seed in SEEDS:
        for budget in (0,32):
            order=sorted(range(len(keys)),key=lambda i:digest([seed,keys[i]]))
            ix=np.asarray(order[:budget],int)
            test=np.asarray(order[budget:],int)
            model=ObserverErrors(tuple(p),14)
            old=VoteFeedbackRouter(tuple(p),14)
            # 先封存所有校准行的各参与者预测，读回后才交给更新函数。
            returned={k:v[ix] for k,v in p.items()}
            model.update(truth[ix],returned)
            old.update(truth[ix],returned)
            inputs={k:v[test] for k,v in p.items()}
            issued={**inputs,"vote":majority(inputs,14),"vote-local":apply_choices(old.bank(inputs),old.choices("vote-local"))}
            for policy in ("uniform","frequency","accuracy"):
                issued[policy]=model.predict(inputs,policy)
            run=f"{budget}-{seed}"
            groups[run]={"returned":[keys[i] for i in ix],"service":[keys[i] for i in test]}
            predictions[run]={k:v.tolist() for k,v in issued.items()}
    sealed=base/"issued.json"
    atomic_json(sealed,{"groups":groups,"predictions":predictions,"fixed_sha256":sha256(path.read_bytes()).hexdigest()})
    saved=read(sealed)
    lookup=dict(zip(keys,truth,strict=True))
    table={}
    for run,arms in saved["predictions"].items():
        ids=saved["groups"][run]["service"]
        target=np.asarray([lookup[i] for i in ids])
        table[run]={k:scores(target,np.asarray(v),[np.arange(len(ids))],14)["pooled"] for k,v in arms.items()}
    atomic_json(base/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"runs":table})
    print(name,{budget:{k:round(float(np.mean([table[f"{budget}-{s}"][k] for s in SEEDS])),4)
                        for k in table[f"{budget}-17"]} for budget in (0,32)},flush=True)


if __name__=="__main__":
    study("board drift","board")
    study("tima drift","time")
