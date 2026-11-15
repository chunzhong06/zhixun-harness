"""在排重的27条源轨迹上训练退化阶段模型，重新发放六个已见机队的预测。"""
import json
import h5py
import joblib
import numpy as np
from score_flight_trajectory_split import DEST as PRIOR, PROFILES, BASE, SITES, dataset_path, fixed_mix, issue, file_hash, LocalStore, atomic_json, trajectory_groups
from score_flight_complement import receipt
from harness.models.flight_masked_readings import ANCHOR
from harness.models.flight_health_stage import HealthStage, PRIMARY, NORMAL, DEGRADED, CONSTANT

DEST=BASE/"flight-health-stage"
PAPER="整趟序列"
PAPER_MIX="两阶段和原工具及论文固定组合"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("当前协议已交付")
        prior=json.loads((PRIOR/"scores.json").read_bytes())
        profiles=json.loads((PROFILES/"scores.json").read_bytes())
        if file_hash(PROFILES/"profiles.npz")!=profiles["缓存SHA256"] or file_hash(PRIOR/"source.joblib")!=prior["源模型SHA256"]:
            raise ValueError("原始开发航班或源权重身份变化")
        ref=joblib.load(PRIOR/"source.joblib")
        groups,_=trajectory_groups(prior["sites"])
        x,y,h=[],[],[]
        all_sites=list(prior["sites"])
        cohorts={}
        with np.load(PROFILES/"profiles.npz") as z:
            for s in sorted(SITES,key=lambda t:(t=="DS02",t)):
                take=prior["训练设备"][s]
                if not take:continue
                keys=z[f"源_{s}_ids"];mask=np.isin(keys[:,0],take)
                with h5py.File(dataset_path(s)) as f:
                    a=f["A_dev"][:]
                    names=[k.decode().strip() for k in f["A_var"][:].ravel()]
                first=np.r_[0,np.flatnonzero(np.any(np.diff(a[:,:2],axis=0),axis=1))+1]
                last=np.r_[first[1:]-1,len(a)-1]
                hs=names.index("hs")
                if np.any(a[first,hs]!=a[last,hs]) or not set(a[first,hs]).issubset({0,1}):
                    raise ValueError("同趟飞行阶段不恒定或字段缺失")
                tags={tuple(row[:2].astype(int)):bool(row[hs]) for row in a[first]}
                h.extend(tags[tuple(k)] for k in keys[mask])
                x.append(z[f"源_{s}_x"][mask]);y.append(z[f"源_{s}_y"][mask])
            model=HealthStage().fit(np.concatenate(x),np.concatenate(y),np.asarray(h))
            for s in all_sites:
                ids,xs,truth=(z[f"开发_{s}_{k}"] for k in ("ids","x","y"))
                old=receipt(PRIOR,s,prior["sites"][s])
                if ids.tolist()!=old["ids"]:raise ValueError("顺序不同")
                anchor=np.array(old["predictions"][ANCHOR]);check,_=ref["readouts"].readings(xs)
                # 确认缓存的六工况参照与本次隔离重训参照逐条等价。
                np.testing.assert_allclose(check[:,0],anchor,atol=1e-8,rtol=0)
                col,prob=model.predict(xs,anchor)
                bank=np.column_stack((anchor,col[NORMAL],col[DEGRADED],col[CONSTANT],old["predictions"][PAPER]))
                owners=np.array([groups[s,"test",int(u)] for u in ids[:,0]])
                cohorts[s]=(ids,truth,owners,{**old["predictions"],**col},bank,prob)
        results={}
        for s,(ids,truth,owners,col,bank,prob) in cohorts.items():
            used=set(owners);xs=[];ys=[];tags=[]
            for t in sorted((t for t in cohorts if t!=s),key=lambda t:(t=="DS02",t)):
                _,labels,g,_,b,_=cohorts[t]
                m=~np.isin(g,list(used));used.update(g[m].tolist())
                xs.append(b[m]);ys.append(labels[m]);tags.extend([t]*m.sum())
            w=fixed_mix(np.vstack(xs),np.concatenate(ys),np.asarray(tags))
            col[PAPER_MIX]=bank@w
            results[s]=issue(DEST,s,ids.tolist(),col,truth)
        means={k:float(np.mean([r["scores"][k]["rmse"] for r in results.values()])) for k in col}
        fixed=min([prior["decision"]["最强固定"],NORMAL,DEGRADED,CONSTANT,"按训练阶段比例固定相加",
                   *[k for k in col if k.startswith("全局模型与固定阶段配比")],PAPER_MIX],key=means.get)
        paper=min([PAPER_MIX,prior["decision"]["同工具论文"],PAPER],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        decision={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],
                  "论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}
        out=DEST/"release.joblib"
        joblib.dump({"prep":ref["prep"],"points":ref["points"],"anchor":ref["readouts"].readers[0],"stage":model},out)
        atomic_json(DEST/"scores.json",{"sites":results,"means":means,"decision":decision,"发布权重SHA256":file_hash(out),
                    "正常阶段源航班":sum(map(int,h)),"异常阶段源航班":len(h)-sum(map(int,h))})
        print(json.dumps({"阶段开发":decision,"源端正常阶段数":sum(map(int,h)),"现场阶段直接使用":means["直接按现场阶段比例相加"]},ensure_ascii=False))


if __name__=="__main__":main()
