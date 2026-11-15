"""让完整轨迹的真实预测误差参与安装试验，发放时仅使用当前航班。"""
import json
import joblib
import numpy as np

from score_flight_block import DEST as PRIOR, FEATURES, MIX, MEAN, NEW_PAPER, other_masks, issue, receipt, file_hash, LocalStore, atomic_json, fixed_mix
from harness.models.flight_error_surface import FlightCorrection, evidence, PRIMARY

DEST = PRIOR.parent / "flight-error-surface"


def run():
    parent = json.loads((PRIOR/"scores.json").read_bytes())
    meta = json.loads((FEATURES/"scores.json").read_bytes())
    if file_hash(FEATURES/"profiles.npz")!=meta["缓存SHA256"]:
        raise ValueError("当前航班测量缓存身份变化")
    data, owners = {}, {}
    with np.load(FEATURES/"profiles.npz") as archive:
        for site,row in parent["sites"].items():
            prev = receipt(PRIOR,site,row)
            ids = archive[site+"_ids"]
            if ids.tolist()!=prev["ids"]:
                raise ValueError("航班输入与预测错位")
            old = prev["predictions"]
            anchor,paper = (np.asarray(old[n]) for n in (MEAN,NEW_PAPER))
            x = evidence(archive[site+"_x"],anchor,paper)
            owners[site] = archive[site+"_group"]
            data[site] = (x,anchor,paper,archive[site+"_y"],ids,old)
    results, decisions = {}, {}
    for site,(x,anchor,paper,truth,ids,old) in data.items():
        masks = other_masks(owners,site)
        train = [np.concatenate([data[t][i][m] for t,m in masks.items()]) for i in range(4)]
        groups = np.concatenate([owners[t][m] for t,m in masks.items()])
        if np.intersect1d(groups,owners[site]).size:
            raise ValueError("当前退化轨迹进入误差学习")
        xx,aa,pp,yy = train
        target = FlightCorrection().fit(xx,yy,aa,groups)
        peer = FlightCorrection().fit(xx,yy,pp,groups)
        weights = DEST/(site+".joblib")
        joblib.dump((target,peer),weights)
        target,peer = joblib.load(weights)
        pred,picks = target.predict(x,anchor)
        ref,_ = peer.predict(x,paper)
        columns = dict(old)
        columns.update(pred)
        for name,v in ref.items():
            columns["论文/"+name] = v
        for name,v in pred.items():
            columns["论文和块均值同法各半/"+name] = (v+ref[name])/2
        tags = np.concatenate([np.repeat(t,m.sum()) for t,m in masks.items()])
        mix = fixed_mix(np.column_stack((aa,pp)),yy,tags)
        for name,v in pred.items():
            columns["公共两工具配比/"+name] = v*mix[0]+ref[name]*mix[1]
        results[site] = issue(DEST,site,ids.tolist(),columns,truth)
        decisions[site] = {"装机轨迹":len(set(groups)), "出场次数":dict(zip(target.candidates,np.bincount(picks,minlength=len(target.candidates)).tolist(),strict=True)),
                           "权重SHA256":file_hash(weights)}
    means = {k:float(np.mean([v["scores"][k]["rmse"] for v in results.values()])) for k in columns}
    fixeds = [*target.candidates,"按所有留出轨迹统一选修正",MEAN,MIX,parent["decision"]["最强固定"]]
    papers = [NEW_PAPER,MIX,parent["decision"]["同工具论文"]]
    for prefix in ("论文/","论文和块均值同法各半/","公共两工具配比/"):
        papers.extend(prefix+k for k in [*target.candidates,"按所有留出轨迹统一选修正",PRIMARY])
    fixed = min([*fixeds,*[k for k in papers if PRIMARY not in k]],key=means.get)
    best_paper = min(papers,key=means.get)
    gain = 1-means[PRIMARY]/min(means[k] for k in (fixed,best_paper))
    worst = max(v["scores"][PRIMARY]["worst"] for v in results.values())
    decision = {"现场":PRIMARY,"固定":fixed,"论文":best_paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],
                "论文误差":means[best_paper],"平均降幅":gain,"现场最差设备":worst,
                "进入独立":gain>=.01 and worst<=1.01*parent["decision"]["改前最差"]}
    atomic_json(DEST/"scores.json",{"decision":decision,"means":means,"sites":results,"装机":decisions})
    print(json.dumps({"单航班误差修正":decision},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("该开发比较已完成")
        run()
