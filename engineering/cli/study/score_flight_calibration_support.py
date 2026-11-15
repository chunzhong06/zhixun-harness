"""从原始交接预测及单航班部件量重做整机队试验，比较局部误差的覆盖与限幅。"""
import json
import joblib
import numpy as np

from score_flight_error_surface import DEST as PRIOR, FEATURES, MEAN, NEW_PAPER, other_masks, issue, receipt, file_hash, LocalStore, atomic_json, fixed_mix, evidence
from harness.models.flight_calibration_support import FleetInstallation, PRIMARY

DEST = PRIOR.parent / "flight-calibration-support"


def run():
    old = json.loads((PRIOR/"scores.json").read_bytes())
    meta = json.loads((FEATURES/"scores.json").read_bytes())
    if file_hash(FEATURES/"profiles.npz")!=meta["缓存SHA256"]:
        raise ValueError("原始读数缓存变更")
    data, group = {}, {}
    with np.load(FEATURES/"profiles.npz") as z:
        for s,r in old["sites"].items():
            p = receipt(PRIOR,s,r)
            ids = z[s+"_ids"]
            if ids.tolist()!=p["ids"]:
                raise ValueError("航班服务身份变更")
            bank = p["predictions"]
            a,b = [np.array(bank[key]) for key in (MEAN,NEW_PAPER)]
            data[s] = evidence(z[s+"_x"],a,b),z[s+"_y"],a,b,ids,bank
            group[s] = z[s+"_group"]
    reports, identities = {}, {}
    for s,(x,y,a,b,ids,bank) in data.items():
        masks = other_masks(group,s)
        xx,yy,aa,bb = [np.concatenate([data[t][j][m] for t,m in masks.items()]) for j in range(4)]
        groups = np.concatenate([group[t][m] for t,m in masks.items()])
        sites = np.concatenate([np.repeat(t,m.sum()) for t,m in masks.items()])
        ma = FleetInstallation().fit(xx,yy,aa,groups,sites)
        mb = FleetInstallation().fit(xx,yy,bb,groups,sites)
        w = fixed_mix(np.column_stack((aa,bb)),yy,sites)
        path = DEST/(s+".joblib")
        joblib.dump((ma,mb,w),path)
        ma,mb,w = joblib.load(path)
        out, peer = ma.predict(x,a), mb.predict(x,b)
        all_columns = {**bank,**out}
        paper_arms = [old["decision"]["论文"]]
        for k in out:
            for name,v in (("论文有限修正/"+k,peer[k]),("有限修正两工具各半/"+k,(peer[k]+out[k])/2),
                           ("有限修正公共配比/"+k,out[k]*w[0]+peer[k]*w[1])):
                all_columns[name] = v
                paper_arms.append(name)
        fixed_arms = [old["decision"]["固定"],*[k for k in out if k!=PRIMARY],*[k for k in paper_arms if PRIMARY not in k]]
        reports[s] = issue(DEST,s,ids.tolist(),all_columns,y)
        identities[s] = file_hash(path)
    means = {k:float(np.mean([r["scores"][k]["rmse"] for r in reports.values()])) for k in all_columns}
    fixed = min(fixed_arms,key=means.get)
    paper = min(paper_arms,key=means.get)
    gain = 1-means[PRIMARY]/min(means[fixed],means[paper])
    worst = max(r["scores"][PRIMARY]["worst"] for r in reports.values())
    before = max(r["scores"]["第一个取样起点"]["worst"] for r in reports.values())
    d = {"现场":PRIMARY,"固定":fixed,"论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],"论文误差":means[paper],
         "降幅":gain,"最差设备":worst,"保护参照":before,"进入独立":gain>=.01 and worst<=1.01*before}
    atomic_json(DEST/"scores.json",{"decision":d,"sites":reports,"means":means,"发布权重SHA256":identities})
    print(json.dumps({"整机队试用":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("同项已交付")
        run()
