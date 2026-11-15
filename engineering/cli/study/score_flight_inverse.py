"""真实训练源整机的反向寿命读法，与所有固定轨迹匹配及同条件论文逐航班对照。"""
import json
import joblib
import numpy as np
from score_flight_throttle import DEST as PRIOR, BASE, other_masks, issue, receipt, file_hash, LocalStore, atomic_json, fixed_mix, PAPER, VIEWS
from harness.models.flight_inverse import InverseLife, PRIMARY, ALL

DEST=BASE/"flight-inverse"
MIX="反推读法与成熟论文固定组合"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已发放本次反推读法")
        prev=json.loads((PRIOR/"scores.json").read_bytes())
        if file_hash(PRIOR/"profiles.npz")!=prev["缓存SHA256"]:raise ValueError("部件轨迹身份不同")
        data={};groups={}
        with np.load(PRIOR/"profiles.npz") as z:
            model=InverseLife().fit(z["source_base"],z["source_y"],z["source_group"])
            for s,record in prev["sites"].items():
                ids=z[s+"_ids"].tolist()
                published=receipt(PRIOR,s,record)
                if ids!=published["ids"]:raise ValueError("源交付的航班次序不同")
                p=dict(published["predictions"])
                outputs=model.predict(z[s+"_base"],np.array(p[VIEWS[-1]]))
                p.update(outputs)
                fixed_preds=[k for k in outputs if PRIMARY not in k]
                bank=np.column_stack([p[k] for k in [VIEWS[-1],*fixed_preds,PAPER]])
                data[s]=(ids,z[s+"_y"],p,bank);groups[s]=z[s+"_group"]
        scores={}
        for s,(ids,y,p,b) in data.items():
            m=other_masks(groups,s)
            source,labels=[np.concatenate([data[t][i][mask] for t,mask in m.items()]) for i in (3,1)]
            teams=np.concatenate([np.repeat(t,mask.sum()) for t,mask in m.items()])
            p[MIX]=b@fixed_mix(source,labels,teams)
            scores[s]=issue(DEST,s,ids,p,y)
        means={k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in p}
        fixed=min([prev["decision"]["最强固定"],MIX,*fixed_preds],key=means.get)
        paper=min([PAPER,prev["decision"]["同工具论文"],MIX],key=means.get)
        arm="与联合部件各半/"+PRIMARY
        worst=max(v["scores"][arm]["worst"] for v in scores.values())
        gain=1-means[arm]/min(means[fixed],means[paper])
        d={"现场":arm,"最强固定":fixed,"同工具论文":paper,"现场误差":means[arm],"固定误差":means[fixed],"论文误差":means[paper],
           "降幅":gain,"现场最差":worst,"进独立":gain>=.01 and worst<=1.01*prev["decision"]["改前最差"]}
        f=DEST/"inverse.joblib";joblib.dump(model,f)
        atomic_json(DEST/"scores.json",{"sites":scores,"means":means,"decision":d,"权重SHA256":file_hash(f)})
        print(json.dumps({"开发交付":d,"固定反推":means[ALL],"留工况反推":means[PRIMARY]},ensure_ascii=False))


if __name__=="__main__":main()
