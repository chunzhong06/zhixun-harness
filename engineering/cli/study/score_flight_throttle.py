"""在排重的源航班训练升降油门读法，在六个已见机队逐批交接同工具比较。"""
import json
import joblib
import numpy as np
from score_flight_component import DEST as FEATURES,SPLIT,BASE,load_part,other_masks,issue,receipt,file_hash,LocalStore,atomic_json,fixed_mix,PAPER,VIEWS
from score_flight_route import DEST as PRIOR
from harness.models.flight_throttle import PRIMARY,FIXED,CORRECTED,CONTROL,read_throttle,ThrottleReaders

DEST=BASE/"flight-throttle"
MIX="方向读法与成熟论文固定组合"


def profiles(extractor,rows):
    keys=sorted(rows)
    views,quality=zip(*(read_throttle(extractor,rows[k][1]) for k in keys),strict=True)
    return keys,[np.stack(x) for x in zip(*views,strict=True)],np.array(quality)


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("本项已交付")
        old=json.loads((PRIOR/"scores.json").read_bytes())
        src=json.loads((SPLIT/"scores.json").read_bytes())
        meta=json.loads((FEATURES/"scores.json").read_bytes())
        if file_hash(FEATURES/"release.joblib")!=meta["发布权重SHA256"] or file_hash(FEATURES/"profiles.npz")!=meta["缓存SHA256"]:
            raise ValueError("部件权重或划分身份变化")
        extractor=joblib.load(FEATURES/"release.joblib")["extractor"]
        chunks=[];source_y=[];quality=[];cache={}; groups={};data={}
        for s,units in src["训练设备"].items():
            if not units:continue
            rows,answers=load_part(s,"dev",set(units))
            keys,x,q=profiles(extractor,rows)
            chunks.append(x);source_y.extend(answers[k] for k in keys);quality.extend(q)
        x=[np.concatenate(v) for v in zip(*chunks,strict=True)]
        y=np.asarray(source_y);model=ThrottleReaders().fit(x,y)
        f=DEST/"release.joblib";joblib.dump({"extractor":extractor,"model":model},f)
        with np.load(FEATURES/"profiles.npz") as z:
            if not np.array_equal(z["source_y"],y):raise ValueError("源训练寿命不同")
            cache.update(source_base=x[0],source_contrast=x[1],source_corrected=x[3],source_selected=x[4],
                         source_group=z["source_group"],source_quality=np.array(quality),source_y=y)
            for s,entry in old["sites"].items():
                rows,answers=load_part(s,"test")
                ids,v,q=profiles(extractor,rows)
                old_pred=receipt(PRIOR,s,entry)
                if old_pred["ids"]!=[list(k) for k in ids] or z[s+"_ids"].tolist()!=old_pred["ids"]:raise ValueError("开发航班错位")
                out=model.predict(v)
                np.testing.assert_allclose(out["不分方向部件模型"],old_pred["predictions"][VIEWS[-1]],atol=1e-8,rtol=0)
                p={**old_pred["predictions"],**out}
                groups[s]=z[s+"_group"]
                truth=np.array([answers[k] for k in ids])
                bank=np.column_stack([p[k] for k in (VIEWS[-1],FIXED,CONTROL,CORRECTED,PAPER)])
                data[s]=(ids,truth,p,bank)
                for name,values in zip(("base","contrast","corrected","selected","quality","group","y","ids"),
                                       (v[0],v[1],v[3],v[4],q,groups[s],truth,np.array(ids)),strict=True):
                    cache[s+"_"+name]=values
        scores={}
        for s,(ids,y,p,b) in data.items():
            masks=other_masks(groups,s)
            train,labels=[np.concatenate([data[t][i][m] for t,m in masks.items()]) for i in (3,1)]
            tags=np.concatenate([np.repeat(t,m.sum()) for t,m in masks.items()])
            p[MIX]=b@fixed_mix(train,labels,tags)
            scores[s]=issue(DEST,s,ids,p,y)
        means={k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in p}
        fixed=min([old["decision"]["最强固定"],VIEWS[-1],FIXED,CONTROL,CORRECTED,"方向差与统一校正各半","方向差与原工具各半",MIX],key=means.get)
        paper=min([old["decision"]["同工具论文"],MIX,PAPER],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        worst=max(r["scores"][PRIMARY]["worst"] for r in scores.values())
        baseline_worst=max(r["scores"][VIEWS[-1]]["worst"] for r in scores.values())
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],"论文误差":means[paper],
           "降幅":gain,"现场最差":worst,"改前最差":baseline_worst,"进独立":gain>=.01 and worst<=1.01*baseline_worst}
        cachepath=DEST/"profiles.npz";np.savez_compressed(cachepath,**cache)
        coverage={s:np.median(cache[s+"_quality"],axis=0).tolist() for s in scores}
        atomic_json(DEST/"scores.json",{"sites":scores,"means":means,"decision":d,"各机队中位配对数和校正通道":coverage,
                    "缓存SHA256":file_hash(cachepath),"权重SHA256":file_hash(f)})
        print(json.dumps({"开发交付":d,"方向差":means[FIXED],"同配对均值":means[CONTROL],"统一校正":means[CORRECTED]},ensure_ascii=False))


if __name__=="__main__":main()
