"""按真实日期将同一反馈规则迁到另一年、另一台仪器的五类识别任务。"""
from hashlib import sha256
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from check_feedback_controls import replay
from gas_complementary_feedback import OUT,ROOT,atomic_json,digest,lda


def source_frame():
    path=ROOT/"data/public/electronic-nose-2025/extracted_features.csv"
    fingerprint=sha256(path.read_bytes()).hexdigest()
    if fingerprint!="9933dca26ad0837c25eecca188d9a0499d848bb08e4c27574538e9e8fc69a38d":
        raise ValueError("外部采集特征身份变化")
    frame=pd.read_csv(path,sep=" ")
    dates=pd.to_datetime(frame["Date[yyyy-mm-dd]"],format="%y-%m-%d")
    days=sorted(dates.unique())
    train=np.flatnonzero(dates.isin(days[:5]))
    groups=[np.flatnonzero(dates==day) for day in days[5:]]
    test=np.concatenate(groups)
    labels=sorted(frame.iloc[train]["Class"].unique())
    y=frame["Class"].map(dict(zip(labels,range(1,len(labels)+1),strict=True))).to_numpy()
    fields=[c for c in frame if "_R" in c]
    if len(fields)!=248 or len(labels)!=5 or not np.isfinite(y).all():
        raise ValueError("作者表的传感器或类别不匹配")
    return frame,fields,days,train,groups,test,labels,y,fingerprint


def load_rows():
    _,_,_,_,groups,test,labels,y,_=source_frame()
    return y[test],labels,[[f"row-{i+2}" for i in members] for members in groups]


def main():
    frame,fields,days,train,groups,test,labels,y,fingerprint=source_frame()
    x=frame[fields].to_numpy(float)
    x[~np.isfinite(x)]=np.nan
    available=np.any(np.isfinite(x[train]),axis=0)
    x=x[:,available]
    columns=np.asarray(fields)[available]
    med=np.nanmedian(x[train],axis=0)
    x=np.where(np.isfinite(x),x,med)
    scale=StandardScaler().fit(x[train])
    logistic=LogisticRegression(C=1.,max_iter=1000,class_weight="balanced",random_state=17)
    tic=perf_counter()
    logistic.fit(scale.transform(x[train]),y[train])
    pred={"current":logistic.predict(scale.transform(x[test]))}
    trees={"forest":RandomForestClassifier(n_estimators=100,max_depth=6,random_state=17,n_jobs=1),
           "catboost":CatBoostClassifier(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1,
                                        random_seed=17,thread_count=1,allow_writing_files=False,verbose=False)}
    counts=np.bincount(y[train])
    weights=len(train)/(len(labels)*counts[y[train]])
    duration={"current":perf_counter()-tic}
    for k,m in trees.items():
        tic=perf_counter()
        m.fit(x[train],y[train],sample_weight=weights)
        opts={"thread_count":1} if k=="catboost" else {}
        pred[k]=m.predict(x[test],**opts).reshape(-1).astype(int)
        duration[k]=perf_counter()-tic
    transform=np.maximum(np.median(abs(x[train]),axis=0),1e-9)
    z=np.sign(x)*np.log1p(abs(x)/transform)
    for feature in ("all","mean","rel_diff","sample_startslope","rec_lvl"):
        ix=np.ones(len(columns),bool) if feature=="all" else np.char.startswith(columns,feature+"_R")
        tic=perf_counter()
        pred[feature]=lda(z[train][:,ix],z[test][:,ix],y[train])
        duration[feature]=perf_counter()-tic
    keys=[[f"row-{i+2}" for i in members] for members in groups]
    folder=OUT/"new-instrument"
    atomic_json(folder/"input.json",{"feature_file_sha256":fingerprint,"labels":labels,"training_ids":train.tolist(),
                "training_dates":[str(d.date()) for d in days[:5]],"service_dates":[str(d.date()) for d in days[5:]],
                "training_rows":len(train),"service_rows":len(test),"feature_count":len(columns),"input":digest([keys,x[test].tolist()]),
                "fit_predict_seconds":duration})
    replay(folder,pred,y[test],keys,(0,4,32),len(labels))


if __name__=="__main__":
    main()
