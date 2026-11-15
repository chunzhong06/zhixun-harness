"""用作者原始五组源航班试验遮路误差；在六组已见但整机队分开的开发航班交付。"""
import json
import h5py
import joblib
import numpy as np
from scipy.stats import spearmanr
from score_flight_condition_ablation import DEST as PRIOR, BASE, PREP, SITES, PAPER, fixed_mix, receipt, issue, LocalStore, atomic_json, file_hash
from score_flight_segment_training import dataset_path, read_flights, signals
from harness.models.flight_condition_response import measured_conditions, reference_points
from harness.models.flight_masked_readings import ANCHOR, RESIDUAL, JOINT, PRIMARY, TOOLS, MaskedReadouts, MaskedTrialPolicy

DEST=BASE/"flight-masked-readings"
PAPER_MIX="同工具和成熟序列固定组合"
CONSTANT="现场分量固定为源均值"


def read_features(site,part,prep,points):
    with h5py.File(dataset_path(site)) as f:
        rows=signals(f,part)
        truth=read_flights(f,part,answers=True)
    ids=sorted(rows)
    if set(ids)!=set(truth):raise ValueError("寿命与原始航班不符")
    x=np.stack([measured_conditions(rows[k][1],prep,points) for k in ids])
    return ids,x,np.asarray([truth[k] for k in ids])


def evaluate(means,old,controls):
    fixed=min([*controls,old["固定"]],key=means.get)
    paper=min([PAPER,PAPER_MIX,old["同工具论文"]],key=means.get)
    gain=1-means[PRIMARY]/min(means[fixed],means[paper])
    return {"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],
            "固定误差":means[fixed],"论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同遮路试验已有结果")
        pack=joblib.load(PREP)
        prep=pack.prep
        points=reference_points(prep)
        prev=json.loads((PRIOR/"scores.json").read_bytes())
        source={s:read_features(s,"dev",prep,points) for s in SITES}
        targets={s:read_features(s,"test",prep,points) for s in prev["sites"]}
        cache={}
        for part,data in (("源",source),("开发",targets)):
            for s,(ids,x,y) in data.items():
                cache[f"{part}_{s}_ids"]=np.array(ids)
                cache[f"{part}_{s}_x"]=x
                cache[f"{part}_{s}_y"]=y
        np.savez_compressed(DEST/"profiles.npz",**cache)
        x,y,tags=(np.concatenate(q) for q in ([v[1] for v in source.values()],[v[2] for v in source.values()],
                                              [np.repeat(s,len(v[2])) for s,v in source.items()]))
        trials,contexts,correlations=[],[],{}
        for s in source:
            take=tags!=s
            m=MaskedReadouts().fit(x[take],y[take])
            predictions,context=m.readings(x[~take])
            e=np.abs(predictions-y[~take,None])
            cor=spearmanr(context[:,-1],e[:,0]).statistic
            pair=spearmanr(context[:,-1],e[:,0]-e[:,2]).statistic
            correlations[s]={"还原偏差与寿命绝对误差":float(cor),"还原偏差与联合读法优势":float(pair)}
            contexts.append(context);trials.append(predictions)
        joined=np.vstack(trials); evidence=np.vstack(contexts)
        gate=MaskedTrialPolicy().fit(joined,evidence,y,tags)
        # 安装时先核对遮路偏差与寿命误差是否有可重复的关联，再决定是否启用现场规则。
        enabled=sum(abs(v["还原偏差与寿命绝对误差"])>=.15 for v in correlations.values())>=3
        model=MaskedReadouts().fit(x,y)
        weights=DEST/"release.joblib"
        joblib.dump({"readouts":model,"policy":gate,"启用":enabled,"prep":prep,"points":points},weights)
        cols={}; matrices={}
        for s,(ids,xs,ys) in targets.items():
            old=receipt(PRIOR,s,prev["sites"][s])
            if old["ids"]!=[list(uid) for uid in ids]:raise ValueError("原回执航班身份不同")
            bank,ctx=model.readings(xs)
            np.testing.assert_allclose(bank[:,0],old["predictions"][ANCHOR],atol=1e-8,rtol=0)
            pred=dict(old["predictions"])
            pred.update(dict(zip(TOOLS,bank.T,strict=True)))
            pred["三读法固定等权"]=bank.mean(1)
            pred[CONSTANT]=bank@gate.mean
            pred[PRIMARY]=(bank*gate.weights(ctx)).sum(1) if enabled else bank[:,0]
            cols[s]=pred
            matrices[s]=np.column_stack((bank,pred[PAPER]))
        scores={}
        for s,(ids,_,truth) in targets.items():
            others=[t for t in targets if t!=s]
            mix=fixed_mix(np.vstack([matrices[t] for t in others]),np.concatenate([targets[t][2] for t in others]),
                          np.concatenate([np.repeat(t,len(targets[t][2])) for t in others]))
            cols[s][PAPER_MIX]=matrices[s]@mix
            scores[s]=issue(DEST,s,ids,cols[s],truth)
        means={k:float(np.mean([z["scores"][k]["rmse"] for z in scores.values()])) for k in cols[s]}
        d=evaluate(means,prev["decision"],[*TOOLS,"三读法固定等权",PAPER_MIX,CONSTANT])
        atomic_json(DEST/"scores.json",{"sites":scores,"means":means,"decision":d,"源端整机队试验":correlations,
                    "启用":enabled,"缓存SHA256":file_hash(DEST/"profiles.npz"),"发布权重SHA256":file_hash(weights)})
        print(json.dumps({"六机队开发":d,"源端关联":correlations,"启用":enabled},ensure_ascii=False))


if __name__=="__main__":main()
