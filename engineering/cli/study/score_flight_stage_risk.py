"""真实阶段只供发放后的瓶颈诊断，现场用整机队隔离的既有寿命误差训练。"""
import json
import h5py
import joblib
import numpy as np
from score_flight_health_stage import BASE, DEST as PRIOR, PROFILES, SITES, dataset_path, file_hash, LocalStore, atomic_json, trajectory_groups, receipt, issue, fixed_mix, PAPER, ANCHOR, NORMAL, DEGRADED
from harness.models.flight_stage_risk import PRIMARY, LEVEL, CONSTANT, StageRisk

DEST=BASE/"flight-stage-risk"
PAPER_MIX="按阶段代价修正与论文固定组合"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已完成本次风险试用")
        prior=json.loads((PRIOR/"scores.json").read_bytes())
        meta=json.loads((PROFILES/"scores.json").read_bytes())
        if file_hash(PRIOR/"release.joblib")!=prior["发布权重SHA256"] or file_hash(PROFILES/"profiles.npz")!=meta["缓存SHA256"]:
            raise ValueError("阶段输入或发布权重身份不同")
        frozen=joblib.load(PRIOR/"release.joblib")
        group,_=trajectory_groups(prior["sites"])
        data={}
        with np.load(PROFILES/"profiles.npz") as z:
            for s,v in prior["sites"].items():
                ids,x,y=(z[f"开发_{s}_{key}"] for key in ("ids","x","y"))
                issued=receipt(PRIOR,s,v)
                if ids.tolist()!=issued["ids"]:raise ValueError("相同航班身份未复现")
                pred=issued["predictions"]
                a,h,d=(np.array(pred[k]) for k in (ANCHOR,NORMAL,DEGRADED))
                generated,p=frozen["stage"].predict(x,a)
                np.testing.assert_allclose(generated[prior["decision"]["现场"]],pred[prior["decision"]["现场"]],rtol=0,atol=1e-8)
                evidence=np.column_stack((a,h,d,p,h-d,x[:,:,:14].mean(1),x[:,:,:14].std(1)))
                owners=np.array([group[s,"test",int(u)] for u in ids[:,0]])
                data[s]=(ids,y,owners,pred,np.column_stack((a,h,d,p)),evidence)
        scores={};diagnostics={};policies={}
        for s,(ids,truth,owners,pred,bank,ctx) in data.items():
            used=set(owners);past=[]
            for t in sorted(set(data)-{s},key=lambda k:(k=="DS02",k)):
                item=data[t]
                m=~np.isin(item[2],list(used));used.update(item[2][m].tolist())
                if m.any():past.append((item[4][m],item[5][m],item[1][m],np.repeat(t,m.sum()),np.array(item[3][PAPER])[m]))
            b,c,y,teams,paper=[np.concatenate(k) for k in zip(*past,strict=True)]
            m=StageRisk().fit(b,c,y,teams);policies[s]=m
            outputs={**pred,**m.predict(bank,ctx)}
            # 固定论文混合只用基础工具，避免让同一批内拟合的代价门控带来不对称训练优势。
            w=fixed_mix(np.column_stack((b[:,:3],(b[:,1]*b[:,3]+b[:,2]*(1-b[:,3])),paper)),y,teams)
            outputs[PAPER_MIX]=np.column_stack((bank[:,:3],bank[:,1]*bank[:,3]+bank[:,2]*(1-bank[:,3]),pred[PAPER]))@w
            scores[s]=issue(DEST,s,ids.tolist(),outputs,truth)
            # 当前航班的答案只在交接之后使用，这项分析不成为下一航班输入。
            with h5py.File(dataset_path(s)) as f:
                a=f["A_test"][:];idx=np.r_[0,np.flatnonzero(np.any(np.diff(a[:,:2],axis=0),axis=1))+1]
                hs=list(k.decode().strip() for k in f["A_var"][:].ravel()).index("hs")
                labels={tuple(row[:2].astype(int)):bool(row[hs]) for row in a[idx]}
            flag=np.array([labels[tuple(k)] for k in ids])
            perfect=(bank[:,0]+np.where(flag,bank[:,1],bank[:,2]))/2
            diagnostics[s]={"真实阶段已知的参照":float(np.sqrt(np.mean((perfect-truth)**2))),"现场阶段估对":int(sum((bank[:,3]>.5)==flag)),"航班":len(flag)}
        means={k:float(np.mean([r["scores"][k]["rmse"] for r in scores.values()])) for k in outputs}
        fixed=min([prior["decision"]["最强固定"],CONSTANT,LEVEL,PAPER_MIX],key=means.get)
        paper=min([PAPER_MIX,prior["decision"]["同工具论文"],PAPER],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],"论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}
        joblib.dump(policies,DEST/"folds.joblib")
        atomic_json(DEST/"scores.json",{"sites":scores,"means":means,"decision":d,"事后诊断":diagnostics})
        print(json.dumps({"开发交付":d,"真实阶段已知的参照":float(np.mean([t["真实阶段已知的参照"] for t in diagnostics.values()]))},ensure_ascii=False))


if __name__=="__main__":main()
