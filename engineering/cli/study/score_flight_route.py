"""用源机的飞行类别委托训练三种寿命读法；服务阶段只传真实传感器和工况。"""
import json
import h5py
import numpy as np
import joblib
from score_flight_component import DEST as FEATURES, SPLIT, BASE, load_part, dataset_path, other_masks, fixed_mix, VIEWS, PAPER, issue, LocalStore, atomic_json, file_hash, receipt, ComponentReadings
from score_flight_local_trials import DEST as PRIOR
from harness.models.flight_route import PRIMARY, JOINT, RouteReadout, route_features

DEST=BASE/"flight-route"
MIX="航线专家和原固定及论文固定最优"


def source_route(s):
    with h5py.File(dataset_path(s)) as f:
        names=[k.decode().strip() for k in f["A_var"][:].ravel()]
        a=f["A_dev"][:]
        i=np.r_[0,np.flatnonzero(np.any(np.diff(a[:,:2],axis=0),axis=1))+1]
        return {tuple(row[:2].astype(int)):int(row[names.index("Fc")]) for row in a[i]}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("航线读法已完成")
        split=json.loads((SPLIT/"scores.json").read_bytes())
        prev=json.loads((PRIOR/"scores.json").read_bytes())
        meta=json.loads((FEATURES/"scores.json").read_bytes())
        if file_hash(FEATURES/"profiles.npz")!=meta["缓存SHA256"]:raise ValueError("部件缓存变化")
        info=[];routes=[];labels=[]
        for s,units in split["训练设备"].items():
            if not units:continue
            rows,lives=load_part(s,"dev",set(units));fc=source_route(s)
            for k in sorted(rows):
                info.append(route_features(rows[k][1]));routes.append(fc[k]);labels.append(lives[k])
        groups={};data={}
        with np.load(FEATURES/"profiles.npz") as z:
            if not np.array_equal(z["source_y"],labels):raise ValueError("航线与部件训练缓存次序改变")
            source=ComponentReadings.split(z["source_x"][:,0])[-1]
            model=RouteReadout().fit(source,np.stack(info),np.asarray(labels),np.array(routes))
            for s,v in prev["sites"].items():
                rows,truth=load_part(s,"test")
                ids=sorted(rows);old=receipt(PRIOR,s,v)
                if z[s+"_ids"].tolist()!=[list(k) for k in ids] or old["ids"]!=[list(k) for k in ids]:raise ValueError("服务航班不同")
                w=np.stack([route_features(rows[k][1]) for k in ids])
                x=ComponentReadings.split(z[s+"_x"][:,0])[-1]
                pred={**old["predictions"],**model.predict(x,w)}
                bank=np.column_stack([pred[k] for k in (VIEWS[-1],JOINT,"固定航线1","固定航线2","固定航线3",PAPER)])
                data[s]=(ids,np.array([truth[k] for k in ids]),pred,bank)
                groups[s]=z[s+"_group"]
        scores={}
        for s,(ids,y,p,b) in data.items():
            masks=other_masks(groups,s)
            matrices=np.vstack([data[t][3][m] for t,m in masks.items()])
            ys=np.concatenate([data[t][1][m] for t,m in masks.items()])
            tags=np.concatenate([np.repeat(t,m.sum()) for t,m in masks.items()])
            p[MIX]=b@fixed_mix(matrices,ys,tags)
            scores[s]=issue(DEST,s,ids,p,y)
        means={k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in p}
        controls=[prev["decision"]["最强固定"],JOINT,MIX,"三航线固定均值","固定训练航线比例","全局与平均航线各半",
                  *[k for k in p if k.startswith(("固定航线","全局与航线"))]]
        fixed=min(controls,key=means.get);paper=min([PAPER,prev["decision"]["同工具论文"],MIX],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],
           "论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}
        weights=DEST/"route.joblib";joblib.dump(model,weights)
        atomic_json(DEST/"scores.json",{"decision":d,"sites":scores,"means":means,"权重SHA256":file_hash(weights)})
        print(json.dumps({"实际交付":d,"工况作为输入":means[JOINT]},ensure_ascii=False))


if __name__=="__main__":main()
