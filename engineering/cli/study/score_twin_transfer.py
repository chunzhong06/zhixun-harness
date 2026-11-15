"""仅在第一块板上拟合，将完整采集日作为新设备的首次交付单位。"""
from hashlib import sha256
from statistics import mean

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gas_complementary_feedback import ROOT,OUT,lda,read,atomic_json
from score_guarded_gas import SEEDS,simulate


DAYS={1:(4,10,15,21),2:(1,7,11,16),3:(2,8,14,17),4:(3,9),5:(18,22)}
FIXED=("current","forest","catboost","all","type-1","type-2","type-3","type-4","shape-svm")


def predict(x,y,train,test):
    a,b=x[train],x[test]
    labels=y[train]
    scaler=StandardScaler().fit(a)
    logistic=LogisticRegression(C=1.,class_weight="balanced",random_state=17,max_iter=1000)
    logistic.fit(scaler.transform(a),labels)
    pred={"current":logistic.predict(scaler.transform(b))}
    models={"forest":RandomForestClassifier(n_estimators=100,max_depth=6,n_jobs=1,random_state=17),
            "catboost":CatBoostClassifier(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1,
                     random_seed=17,thread_count=1,verbose=False,allow_writing_files=False)}
    weights=len(labels)/(4*np.bincount(labels)[labels])
    for name,model in models.items():
        model.fit(a,labels,sample_weight=weights)
        opts={"thread_count":1} if name=="catboost" else {}
        pred[name]=model.predict(b,**opts).reshape(-1).astype(int)
    med=np.maximum(np.median(abs(a),axis=0),1e-9)
    def transform(v):
        return (np.sign(v)*np.log1p(abs(v)/med)).reshape(-1,8,8)
    za,zb=transform(a),transform(b)
    for j,name in enumerate(FIXED[3:8]):
        channels=list(range(8)) if j==0 else [j-1,j+3]
        pred[name]=lda(za[:,channels,:].reshape(len(a),-1),zb[:,channels,:].reshape(len(b),-1),labels)
    def shape(v):
        array=v.reshape(-1,8,8)
        d=array-array.mean(axis=1,keepdims=True)
        return (d/np.maximum(np.sqrt((d*d).mean(axis=1,keepdims=True)),1e-9)).reshape(len(v),-1)
    pred["shape-svm"]=SVC(C=1.,gamma="scale",class_weight="balanced").fit(shape(a),labels).predict(shape(b))
    return pred


def main():
    source=ROOT/"data/public/twin-gas-arrays/twin_features.csv"
    fingerprint=sha256(source.read_bytes()).hexdigest()
    if fingerprint!="81cc1a20a36d5d87dc2d453e75bf5279daf37e76584f7b9ac9ba9e82d418ad65":
        raise ValueError("作者特征表内容已变")
    data=pd.read_csv(source)
    x=data[[f"f{k:02d}" for k in range(64)]].to_numpy(dtype=float)
    if (len(data)!=640 or not np.isfinite(x).all() or data.duplicated(["board","rep","gas","conc"]).any()
        or not (data.groupby(["board","rep","gas"]).size()==10).all()):
        raise ValueError("作者采集记录数量、内容或组划分有误")
    y=data.gas.map({"GCO":1,"GEa":2,"GEy":3,"GMe":4}).to_numpy(dtype=int)
    day=np.array([DAYS[b][r-1] for b,r in zip(data.board,data.rep,strict=True)])
    keys=np.array(["twin-"+sha256(row.astype("<f8").tobytes()).hexdigest() for row in x])
    if len(set(keys))!=len(keys):
        raise ValueError("测量内容重复")
    out=OUT/"twin-author-features"
    summary={}
    for protocol in ("archived","chronological"):
        train=np.flatnonzero((data.board==1)&((protocol=="archived")|(day==4)))
        test=np.flatnonzero((data.board!=1)&((protocol=="archived")|(day>4)))
        predictions=predict(x,y,train,test)
        tables={}
        for board in range(2,6):
            stream=[]
            days=sorted(set(day[test[data.board.to_numpy()[test]==board]]))
            for d in days:
                pos=np.flatnonzero((data.board.to_numpy()[test]==board)&(day[test]==d))
                ids=test[pos]
                group=keys[ids].tolist()
                pred={k:v[pos] for k,v in predictions.items()}
                stream.append((group,pred,group,pred,y[ids]))
            for budget in (4,32):
                tables[board,budget]=simulate(stream,dict(zip(keys[test],y[test],strict=True)),
                    out/protocol/f"board-{board}-labels-{budget}",budget,4,include_fixed=True)
        result={}
        for budget in (4,32):
            per={}
            for arm in (*FIXED,"vote","local","vote-local","vote-confirm","guarded-vote"):
                seeds=SEEDS if arm not in (*FIXED,"vote") else (None,)
                logs=[]
                for seed in seeds:
                    name=arm if seed is None else f"{arm}-{seed}"
                    board_scores=[tables[b,budget][name]["pooled"] for b in range(2,6)]
                    logs.append({"seed":seed,"boards":board_scores,"mean_board":mean(board_scores),"worst_board":min(board_scores)})
                per[arm]={"mean_board":mean(r["mean_board"] for r in logs),"worst_board":mean(r["worst_board"] for r in logs),"runs":logs}
            result[str(budget)]=per
        summary[protocol]={"fit":len(train),"test":len(test),"budgets":result,"input_sha256":fingerprint,
                           "fit_keys":keys[train].tolist()}
        for budget in (4,32):
            print(protocol,budget,{a:round(v["mean_board"],4) for a,v in result[str(budget)].items()},flush=True)
    atomic_json(out/"scores.json",summary)


if __name__=="__main__":
    main()
