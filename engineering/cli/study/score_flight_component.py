"""比较各部件可测关系和同工况复测策略，保留按退化轨迹隔离的成熟对照。"""
import json
import h5py
import joblib
import numpy as np
from score_flight_trajectory_split import DEST as SPLIT, BASE, SITES, dataset_path, signals, trajectory_groups, LocalStore, atomic_json, file_hash, issue, fixed_mix
from score_flight_complement import receipt
from score_flight_stage_risk import DEST as PRIOR
from score_ds02_forward import read_flights
from harness.models.flight_component import ComponentFeatures, ComponentReadings, SENSORS, SETTINGS, VIEWS, PRIMARY
from harness.models.flight_masked_readings import ANCHOR

DEST=BASE/"flight-component"
PAPER="整趟序列"
BLEND="部件和整趟论文固定最优"


def other_masks(groups,site):
    used=set(groups[site])
    masks={}
    for s in sorted(set(groups)-{site},key=lambda t:(t=="DS02",t)):
        mask=~np.isin(groups[s],list(used));used.update(groups[s][mask].tolist())
        if mask.any():masks[s]=mask
    return masks


def load_part(s,part,units=None):
    with h5py.File(dataset_path(s)) as f:
        decode=lambda field:tuple(k.decode().strip() for k in f[field][:].ravel())
        if (decode("X_s_var"),decode("W_var"))!=(SENSORS,SETTINGS):raise ValueError("测站顺序改变")
        rows=signals(f,part,units)
        y=read_flights(f,part,answers=True)
    return rows,y


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("此协议已有交付")
        split=json.loads((SPLIT/"scores.json").read_bytes())
        old=json.loads((PRIOR/"scores.json").read_bytes())
        if file_hash(SPLIT/"source.joblib")!=split["源模型SHA256"]:raise ValueError("健康参照权重不同")
        bundle=joblib.load(SPLIT/"source.joblib")
        early,_=load_part("DS01","dev",{4,5,6})
        healthy=np.vstack([early[k][1] for u in (4,5,6) for k in sorted(k for k in early if k[0]==u)[:8]])
        extractor=ComponentFeatures().fit(bundle["prep"],bundle["points"],healthy)
        owners,_=trajectory_groups(old["sites"])
        cache={};x=[];y=[];src_groups=[]
        for s,ids in split["训练设备"].items():
            if not ids:continue
            rows,labels=load_part(s,"dev",set(ids))
            keys=sorted(rows)
            x.extend(extractor.read(rows[k][1]) for k in keys)
            y.extend(labels[k] for k in keys)
            src_groups.extend(owners[s,"dev",k[0]] for k in keys)
        x,y,g=np.stack(x),np.array(y),np.array(src_groups)
        if len(y)!=2219 or len(set(g))!=27:raise ValueError("源退化轨迹改变")
        # 源端留整条退化轨迹验证工具间有无可纠正的余地，所有试验共享同一分折。
        fold=np.array_split(np.unique(g),4)
        oof=np.empty((len(y),len(VIEWS)))
        confidence=np.empty_like(oof)
        for group in fold:
            te=np.isin(g,group)
            sub=ComponentReadings().fit(x[~te],y[~te]);oof[te],confidence[te],_=sub.trial(x[te])
        root=np.sqrt(np.mean((oof-y[:,None])**2,axis=0))
        oracle=float(np.sqrt(np.mean(np.min((oof-y[:,None])**2,axis=1))))
        enabled=oracle<=.99*root.min()
        model=ComponentReadings().fit(x,y)
        cache.update(source_x=x,source_y=y,source_group=g,source_oof=oof,source_instability=confidence)
        weights=DEST/"release.joblib";joblib.dump({"extractor":extractor,"model":model,"enabled":enabled},weights)
        data={};groups={}
        for s,record in old["sites"].items():
            rows,answers=load_part(s,"test")
            keys=sorted(rows); previous=receipt(PRIOR,s,record)
            if [list(k) for k in keys]!=previous["ids"]:raise ValueError("服务机队错位")
            feature=np.stack([extractor.read(rows[k][1]) for k in keys])
            bank,spread,trial=model.trial(feature)
            p=dict(previous["predictions"])
            np.testing.assert_allclose(bank[:,0],p[ANCHOR],atol=1e-8,rtol=0)
            p.update(dict(zip(VIEWS,bank.T,strict=True)))
            p["部件固定等权"]=bank.mean(1)
            p[PRIMARY]=trial if enabled else bank[:,root.argmin()]
            p["联合部件与原读数各半"]=(bank[:,0]+bank[:,-1])/2
            data[s]=(keys,np.array([answers[k] for k in keys]),p,bank)
            groups[s]=np.array([owners[s,"test",k[0]] for k in keys])
            cache.update({s+"_x":feature,s+"_ids":np.array(keys),s+"_y":data[s][1],s+"_group":groups[s],s+"_instability":spread})
        scores={}
        for s,(keys,truth,p,bank) in data.items():
            masks=other_masks(groups,s)
            mat=np.vstack([np.column_stack((data[t][3],data[t][2][PAPER]))[m] for t,m in masks.items()])
            labels=np.concatenate([data[t][1][m] for t,m in masks.items()])
            tags=np.concatenate([np.repeat(t,m.sum()) for t,m in masks.items()])
            joint=fixed_mix(mat,labels,tags)
            p[BLEND]=np.column_stack((bank,p[PAPER]))@joint
            p["部件工具固定最优"]=bank@fixed_mix(mat[:,:-1],labels,tags)
            scores[s]=issue(DEST,s,keys,p,truth)
        means={k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in p}
        fixed=min([old["decision"]["最强固定"],*VIEWS,"部件工具固定最优","部件固定等权","联合部件与原读数各半",BLEND],key=means.get)
        paper=min([PAPER,BLEND,old["decision"]["同工具论文"]],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],
           "论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}
        data_path=DEST/"profiles.npz";np.savez_compressed(data_path,**cache)
        atomic_json(DEST/"scores.json",{"sites":scores,"means":means,"decision":d,"源端四折":dict(zip(VIEWS,root.tolist())),
                    "源端事后最优":oracle,"启用":bool(enabled),"缓存SHA256":file_hash(data_path),"发布权重SHA256":file_hash(weights)})
        print(json.dumps({"开发交付":d,"源端事后最优":oracle,"各固定读法":{k:means[k] for k in VIEWS}},ensure_ascii=False))


if __name__=="__main__":main()
