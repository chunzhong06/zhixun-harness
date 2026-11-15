"""在排重源轨迹学寿命门槛。全量交接六个已见机队的固定、动态与论文组合。"""
import json
import joblib
import numpy as np

from score_flight_calibration_support import DEST as PRIOR,FEATURES,MEAN,NEW_PAPER,other_masks,issue,receipt,file_hash,LocalStore,atomic_json,fixed_mix
from harness.models.flight_horizons import HorizonLife,GLOBAL,LOCAL,PRIMARY

DEST=PRIOR.parent/"flight-horizons"


def run():
    previous=json.loads((PRIOR/"scores.json").read_bytes())
    info=json.loads((FEATURES/"scores.json").read_bytes())
    if file_hash(FEATURES/"profiles.npz")!=info["缓存SHA256"]:
        raise ValueError("源开发工况缓存身份改变")
    data={};groups={}
    with np.load(FEATURES/"profiles.npz") as archive:
        if len(set(archive["source_group"]))!=27:
            raise ValueError("排重源机台数已变化")
        model=HorizonLife().fit(archive["source_x"],archive["source_y"],archive["source_group"])
        path=DEST/"release.joblib"
        joblib.dump(model,path,compress=3)
        model=joblib.load(path)
        for site,row in previous["sites"].items():
            parent=receipt(PRIOR,site,row)
            ids=archive[site+"_ids"]
            if ids.tolist()!=parent["ids"]:
                raise ValueError("开发航班与先前实发预测错位")
            curves=model.predict(archive[site+"_x"])
            data[site]=ids,archive[site+"_y"],parent["predictions"],curves
            groups[site]=archive[site+"_group"]
    fixed_old=previous["decision"]["固定"];paper_old=previous["decision"]["论文"]
    fixed=[MEAN,fixed_old,paper_old];papers=[paper_old,NEW_PAPER]
    names=[k for k in curves if k!=LOCAL]
    reports={}
    for site,(ids,y,parent,curves) in data.items():
        columns={k:parent[k] for k in set([*fixed[:3],paper_old,NEW_PAPER,"第一个取样起点"])}
        anchor=np.array(parent[MEAN]);paper=np.array(parent[NEW_PAPER])
        columns.update(curves)
        for k,v in curves.items():
            for base,label in [(anchor,"块均值"),(paper,"同工具论文")]:
                name=k+"与"+label+"各半"
                columns[name]=(v+base)/2
                if k!=LOCAL:
                    fixed.append(name)
                if label=="同工具论文":
                    papers.append(name)
        columns[PRIMARY]=(curves[LOCAL]+anchor)/2
        def matrix(s):
            _,_,p,c=data[s]
            return np.column_stack((p[MEAN],p[NEW_PAPER],*[c[k] for k in names]))
        masks=other_masks(groups,site)
        training=np.vstack([matrix(s)[take] for s,take in masks.items()])
        labels=np.concatenate([data[s][1][take] for s,take in masks.items()])
        sites=np.concatenate([np.repeat(s,take.sum()) for s,take in masks.items()])
        share=fixed_mix(training,labels,sites)
        columns["论文及门槛工具公共配比"]=matrix(site)@share
        reports[site]=issue(DEST,site,ids.tolist(),columns,y)
    means={k:float(np.mean([r["scores"][k]["rmse"] for r in reports.values()])) for k in columns}
    fixed=min([*fixed,*names,"论文及门槛工具公共配比"],key=means.get)
    paper=min([*papers,"论文及门槛工具公共配比"],key=means.get)
    gain=1-means[PRIMARY]/min(means[fixed],means[paper])
    worst=max(r["scores"][PRIMARY]["worst"] for r in reports.values())
    guard=previous["decision"]["保护参照"]
    d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],"论文误差":means[paper],
       "降幅":gain,"最差设备":worst,"保护参照":guard,"进入独立":gain>=.01 and worst<=1.01*guard}
    atomic_json(DEST/"scores.json",{"decision":d,"sites":reports,"means":means,"模型SHA256":file_hash(path)})
    print(json.dumps({"按剩余飞行数逐档训练":d,"冻结门槛权重":means[GLOBAL+"与块均值各半"],"服务航班":sum(r["rows"] for r in reports.values())},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此项已发放")
        run()
