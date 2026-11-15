"""在两块真实SGP40芯片的合法配对装机混合物上做训练试验，之后才对服务混合物评分。"""
from hashlib import md5,sha256
import json
import h5py
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score
from sgp40_transfer import SOURCE,MD5
from score_plume_training import DEST as LAST,LocalStore,LocalAcceptance,digest,atomic_json
from harness.models.cycle_install import InstallCycles,PRIMARY,PAPER,CV

DEST=LAST.parent/"sgp-install-cycles"
SETUPS=(20,80)


def features(signal):
    # 12个温度步骤各有5秒高温和7秒低温；分别读两个阶段均值及低温段的后半与前半差。
    arr=np.asarray(signal,dtype=float).reshape(4,12,120)
    if not np.isfinite(arr).all():raise ValueError("SGP40循环非有限")
    return np.stack((arr[:,:,:50].mean(2),arr[:,:,50:].mean(2),
                     arr[:,:,85:].mean(2)-arr[:,:,50:85].mean(2)),axis=2).ravel()


def raw_part(f,s,p):
    ids=np.asarray(f["targets_"+p+"/range"]).ravel()
    unique=np.unique(ids).astype(int)
    if not np.array_equal(ids,ids.astype(int)):raise ValueError("混合物编号非整数")
    layers=np.stack([np.asarray(f[ref]).T for ref in f[f"sensor{s}_{p}"][:].ravel()],axis=1)
    if layers.shape!=(len(ids),4,1440):raise ValueError("周期数据不完整")
    x=np.stack([features(row) for row in layers])
    group=[x[ids==i] for i in unique]
    if any(len(v) not in (3,4) for v in group):raise ValueError("原作者的保留周期数不同")
    return unique,group


def labels(f,p,ids):
    codes=np.asarray(f["targets_"+p+"/range"]).ravel()
    values=np.asarray(f["targets_"+p+"/VOCsum"]).ravel()
    result=[]
    for i in ids:
        v=np.unique(values[codes==i])
        if len(v)!=1 or not np.isfinite(v[0]):raise ValueError("混合物标签不唯一")
        result.append(v[0])
    return np.array(result)


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("本项SGP安装已运行")
        if md5(SOURCE.read_bytes()).hexdigest()!=MD5:raise ValueError("原包MD5改变")
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        events={};choices={};score={}
        with h5py.File(SOURCE) as f:
            ids,source=raw_part(f,"A","train");y=labels(f,"train",ids)
            order=sorted(range(len(ids)),key=lambda i:sha256(f"sgp-install-cycle-v1/{ids[i]}".encode()).digest())
            threshold=float(np.median(y))
            for s in "BC":
                train_ids,target=raw_part(f,s,"train");test_ids,service=raw_part(f,s,"test")
                if not np.array_equal(ids,train_ids) or set(test_ids)&set(ids) or len(test_ids)!=200:raise ValueError("安装和服务混合物重合")
                for size in SETUPS:
                    ix=np.asarray(order[:size]);folds=np.arange(size)%5
                    m=InstallCycles().fit(source,y,[target[i] for i in ix],ix,folds)
                    key=f"{s}-{size}";weight=DEST/(key+".joblib");joblib.dump(m,weight)
                    p=joblib.load(weight).predict(service)
                    payload={"装机设备":s,"装机混合物":ids[ix].tolist(),"服务混合物":test_ids.tolist(),"源阈值":threshold,
                             "实发分类":{k:(1+(v>threshold).astype(int)).tolist() for k,v in p.items()},
                             "实发浓度":{k:v.tolist() for k,v in p.items()},"权重SHA256":sha256(weight.read_bytes()).hexdigest()}
                    request=digest(payload)
                    if not receiver.submit(request,payload)["accepted"]:raise ValueError("预测未接收")
                    events[key]=request;choices[key]={"整体验证赢家":m.names[m.best],"采用分量":dict(zip(m.names,m.weight.tolist()))}
            truth=1+(labels(f,"test",test_ids)>threshold).astype(int)
        for key,request in events.items():
            sent=json.loads((receiver.output/(request+".json")).read_bytes())["payload"]
            if digest(sent)!=request:raise ValueError("已发结果改变")
            score[key]={k:{"F1":float(f1_score(truth,v,average="macro")),"最弱类召回":float(recall_score(truth,v,average=None).min())}
                        for k,v in sent["实发分类"].items()}
        # B/C同一批测试混合物；20/80是预设的两种安装预算，不作为四个独立采集。
        means={k:{v:float(np.mean([s[k][v] for s in score.values()])) for v in ("F1","最弱类召回")} for k in score[key]}
        fixed=max([k for k in means if k not in (PRIMARY,CV)],key=lambda k:means[k]["F1"])
        paper=max([k for k in means if k.startswith("论文与") or k==PAPER],key=lambda k:means[k]["F1"])
        gap=means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper,CV))
        d={"候选":PRIMARY,"固定":fixed,"论文特征对照":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
           "论文特征组合F1":means[paper]["F1"],"固定安装选型F1":means[CV]["F1"],"差值":gap,"过初筛":gap>=.01 and means[PRIMARY]["最弱类召回"]>=means[fixed]["最弱类召回"]-.02}
        atomic_json(DEST/"scores.json",{"scores":score,"means":means,"decision":d,"装机选择":choices,"requests":events})
        print(json.dumps({"跨芯片开发":d,"同一批服务混合物":len(truth),"源混合物":len(ids)},ensure_ascii=False))


if __name__=="__main__":main()
