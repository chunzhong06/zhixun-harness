"""按退化轨迹隔离，实跑在相似历史航班上试用纠错范围的策略。"""
import json
import joblib
import numpy as np
from score_flight_component import DEST as PRIOR, BASE, LocalStore, atomic_json, file_hash, receipt, other_masks, issue, fixed_mix, VIEWS, PAPER
from harness.models.flight_local_trials import LocalTrials, context, PRIMARY, NONE, GLOBAL, WIDTHS

DEST=BASE/"flight-local-trials"
MIX="近邻固定工具和论文组合"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有相同近邻试用结果")
        old=json.loads((PRIOR/"scores.json").read_bytes())
        if file_hash(PRIOR/"profiles.npz")!=old["缓存SHA256"]:raise ValueError("部件特征身份变化")
        data={};groups={}
        with np.load(PRIOR/"profiles.npz") as z:
            for s,report in old["sites"].items():
                ids=z[s+"_ids"].tolist();payload=receipt(PRIOR,s,report)
                if payload["ids"]!=ids:raise ValueError("历史航班错位")
                pred=payload["predictions"]
                bank=np.column_stack([pred[k] for k in VIEWS])
                data[s]=(ids,z[s+"_y"],bank,context(z[s+"_x"],bank),pred)
                groups[s]=z[s+"_group"]
        scores={};hist={};policies={}
        for s,(ids,truth,bank,features,pred) in data.items():
            masks=other_masks(groups,s)
            collect=lambda i:np.concatenate([data[t][i][mask] for t,mask in masks.items()])
            group=np.concatenate([groups[t][mask] for t,mask in masks.items()])
            y,b,x=collect(1),collect(2),collect(3)
            m=LocalTrials().fit(x,b,y,group);policies[s]=m
            outputs,sel=m.predict(features,bank)
            out={**pred,**outputs}
            hist[s]=np.bincount(sel,minlength=5).tolist()
            # 将真实留整台试用得到的各固定尺度交给论文组合；训练组合时不能用自拟合纠错值。
            offsets=m.corrections(m.refs,leave_group=group)
            paper_ref=np.concatenate([np.asarray(data[t][4][PAPER])[mask] for t,mask in masks.items()])
            teams=np.concatenate([np.repeat(t,mask.sum()) for t,mask in masks.items()])
            mix=fixed_mix(np.column_stack((b[:,3],np.maximum(0,b[:,3:4]+offsets),paper_ref)),y,teams)
            out[MIX]=np.column_stack((bank[:,3],*[outputs[f"固定邻域宽度{w:g}"] for w in WIDTHS],pred[PAPER]))@mix
            scores[s]=issue(DEST,s,ids,out,truth)
        means={k:float(np.mean([r["scores"][k]["rmse"] for r in scores.values()])) for k in out}
        fixed=min([old["decision"]["最强固定"],GLOBAL,MIX,*[f"固定邻域宽度{w:g}" for w in WIDTHS]],key=means.get)
        paper=min([PAPER,MIX,old["decision"]["同工具论文"]],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],
           "论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}
        joblib.dump(policies,DEST/"folds.joblib")
        atomic_json(DEST/"scores.json",{"decision":d,"sites":scores,"means":means,"各宽度采用数":hist})
        print(json.dumps({"六机队实际交付":d,"现场半量":means["邻域现场修正半量"]},ensure_ascii=False))


if __name__=="__main__":main()
