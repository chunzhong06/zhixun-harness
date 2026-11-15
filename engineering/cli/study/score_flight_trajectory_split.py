"""核验逐航班健康演化是否重复；按相同演化隔离训练与校准，重训同工具的论文和固定比较。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import h5py
import joblib
import numpy as np
from score_flight_sensor_trials import BASE, LocalStore, atomic_json, file_hash, fixed_mix, issue, PAPER
from score_flight_masked_readings import DEST as PROFILES, SITES
from score_flight_load_neighbors import dataset_path, signals, batch_sequence, call_paper
from harness.models.flight_masked_readings import MaskedReadouts, TOOLS
from harness.models.flight_sensor_trials import TrialCorrection, PRIMARY, FIXED_BIAS, MASK_BIAS, SWITCH
from harness.models.flight_condition_response import measured_conditions, reference_points
from score_engine_flight_load import fit_tools

DEST=BASE/"flight-trajectory-split"
PAPER_MIX="同源论文与三读法固定最优"


def trajectory_groups(sites):
    """身份只在构造划分时使用；跨子集采样率可能不同，逐航班比较健康参数的首、末时刻。"""
    variants=[]
    owners={}
    for s in sites:
        parts=("dev","test") if s in SITES else ("test",)
        with h5py.File(dataset_path(s)) as f:
            fields=[k.decode().strip() for k in f["T_var"][:].ravel()]
            if set(fields)!=set(variants[0]["fields"] if variants else fields):raise ValueError("健康参数定义变化")
            order=np.argsort(fields)
            for part in parts:
                a=f["A_"+part][:,:2].astype(int)
                starts=np.r_[0,np.flatnonzero(np.any(np.diff(a,axis=0),axis=1))+1]
                ends=np.r_[starts[1:]-1,len(a)-1]
                # 两个采样率可能令航班末样本相差一个物理时刻；使用首次与末次的均值表达每趟健康状态。
                h=(f["T_"+part][starts][:,order]+f["T_"+part][ends][:,order])/2
                for unit in np.unique(a[starts,0]):
                    take=a[starts,0]==unit
                    series=h[take]
                    hits=[i for i,v in enumerate(variants) if series.shape==v["series"].shape and np.allclose(series,v["series"],rtol=0,atol=1e-8)]
                    if len(hits)>1:raise ValueError("轨迹身份匹配到多条")
                    if not hits:
                        hits=[len(variants)]
                        variants.append({"series":series,"fields":fields,"members":[]})
                    group=hits[0]; owner=(s,part,int(unit))
                    variants[group]["members"].append(owner);owners[owner]=group
    return owners,[v["members"] for v in variants if len(v["members"])>1]


def main():
    with LocalStore(DEST/"lock").exclusive(),TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("当前隔离协议已实跑")
        orig=json.loads((PROFILES/"scores.json").read_bytes())
        sites=tuple(orig["sites"])
        ids_by_group,overlap=trajectory_groups(sites)
        held={g for (s,part,u),g in ids_by_group.items() if part=="test"}
        with np.load(PROFILES/"profiles.npz") as source:
            if file_hash(PROFILES/"profiles.npz")!=orig["缓存SHA256"]:raise ValueError("航班身份缓存变化")
            fleet={}
            for s in sites:
                for part in (("dev","test") if s in SITES else ("test",)):
                    prefix=("源_" if part=="dev" else "开发_")+s
                    fleet[s,part]=(source[prefix+"_ids"],source[prefix+"_y"])
        # DS01和DS03是原始单一故障子集；混合子集DS02排在最后，重复的源轨迹只保留一份。
        ranking=sorted(SITES,key=lambda s:(s=="DS02",s))
        seen=set(); retained={}; counts={}
        for s in ranking:
            ids,_=fleet[s,"dev"]
            keep=[]
            for u in sorted(set(ids[:,0])):
                g=ids_by_group[s,"dev",int(u)]
                if g not in held and g not in seen:
                    seen.add(g);keep.append(int(u))
            retained[s]=keep;counts[s]=int(np.isin(ids[:,0],keep).sum())
        if seen&held or not seen:raise ValueError("训练留出身份冲突")
        stage=Path(temp)
        dev_x=[];dev_y=[];seq=[];valid=[];early=[];raw_by_site={}
        for s in ranking:
            if not retained[s]:continue
            with h5py.File(dataset_path(s)) as f: raw_by_site[s]=signals(f,"dev",set(retained[s]))
        # 重建原健康模型的4、5、6号开工参照，核对它们均通过了源/目标去重。
        if not {4,5,6}.issubset(retained["DS01"]):raise ValueError("健康参照涉及留出轨迹")
        start_flights={k:v for k,v in raw_by_site["DS01"].items() if k[0] in (4,5,6)}
        source_ids,source_life=fleet["DS01","dev"]
        prep,_,_=fit_tools(start_flights,dict(zip(map(tuple,source_ids.tolist()),source_life,strict=True)))
        points=reference_points(prep)
        for s,flights in raw_by_site.items():
            ids,y=fleet[s,"dev"]; answers=dict(zip(map(tuple,ids.tolist()),y,strict=True))
            held_units=set(sorted(retained[s])[::5])
            for u in sorted(retained[s]):
                for i,k in enumerate(sorted(k for k in flights if k[0]==u)):
                    rows=flights[k][1]
                    dev_x.append(measured_conditions(rows,prep,points));dev_y.append(answers[k])
                    seq.append(batch_sequence(rows));early.append(i<8);valid.append(u in held_units)
        raw_by_site.clear()
        src=np.stack(dev_x);y=np.asarray(dev_y)
        readouts=MaskedReadouts().fit(src,y)
        joblib.dump({"readouts":readouts,"points":points,"prep":prep},DEST/"source.joblib")
        np.savez(stage/"source.npz",x=np.stack(seq),y=y,early=early,valid=valid)
        paper_path=DEST/"paper.pt"
        call_paper(["train",str(stage/"source.npz"),str(paper_path)])
        cohorts={};testseq=[]
        for s in sites:
            ids,truth=fleet[s,"test"]
            with h5py.File(dataset_path(s)) as f: flights=signals(f,"test")
            keys=list(map(tuple,ids.tolist()))
            if sorted(flights)!=keys:raise ValueError("重训交付与登记航班不一致")
            x=np.stack([measured_conditions(flights[k][1],prep,points) for k in keys])
            bank,ctx=readouts.readings(x)
            # 对齐现有修正器的原始、联合、残差列。其他工具和论文获得同样的装机资料。
            bank=bank[:,[0,2,1]]
            testseq.extend(batch_sequence(flights[k][1]) for k in keys)
            owners=np.array([ids_by_group[s,"test",int(k[0])] for k in keys])
            cohorts[s]=(ids,truth,bank,ctx,owners)
        np.save(stage/"flight.npy",np.stack(testseq))
        call_paper(["predict",str(stage/"flight.npy"),str(stage/"paper.npy"),str(paper_path)])
        p=np.load(stage/"paper.npy"); pos=0; paper={}
        for s,(ids,*_) in cohorts.items():
            paper[s]=p[pos:pos+len(ids)];pos+=len(ids)
        if pos!=len(p):raise ValueError("论文预测条数变化")
        scores={};cal_counts={};policies={}
        for s,(ids,truth,b,ctx,owners) in cohorts.items():
            masks={t:~np.isin(v[4],owners) for t,v in cohorts.items() if t!=s}
            # 同一轨迹若在两个其他开发机队出现，只能贡献一次校准标签。
            used=set()
            for t in sorted(masks,key=lambda t:(t=="DS02",t)):
                masks[t] &= ~np.isin(cohorts[t][4],list(used))
                used.update(cohorts[t][4][masks[t]].tolist())
            if used.intersection(owners):raise ValueError("留出轨迹进入校准")
            hist=lambda i:np.concatenate([cohorts[t][i][mask] for t,mask in masks.items()])
            train,info,label=hist(2),hist(3),hist(1)
            teams=np.concatenate([np.repeat(t,mask.sum()) for t,mask in masks.items()])
            p_hist=np.concatenate([paper[t][mask] for t,mask in masks.items()])
            policy=TrialCorrection().fit(train,info,label,teams);policies[s]=policy
            mixture=fixed_mix(np.column_stack((train,p_hist)),label,teams)
            pred={TOOLS[0]:b[:,0],TOOLS[2]:b[:,1],TOOLS[1]:b[:,2],PAPER:paper[s],**policy.predict(b,ctx)}
            pred[PAPER_MIX]=np.column_stack((b,paper[s]))@mixture
            pred["固定三读法等权"]=b.mean(1)
            best=fixed_mix(train,label,teams)
            pred["三种树固定最优"]=b@best
            scores[s]=issue(DEST,s,ids.tolist(),pred,truth)
            cal_counts[s]={"轨迹数":len(used),"校准航班":len(label)}
        means={k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in pred}
        fixed=min([*TOOLS,PAPER,PAPER_MIX,FIXED_BIAS,MASK_BIAS,"固定校准与原版各半","全部诊断固定校准","固定三读法等权","三种树固定最优"],key=means.get)
        paper_control=min([PAPER,PAPER_MIX],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper_control])
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper_control,"现场误差":means[PRIMARY],"固定误差":means[fixed],
           "论文误差":means[paper_control],"同条件单论文":means[PAPER],"降幅":gain,"进独立":gain>=.01}
        joblib.dump(policies,DEST/"folds.joblib")
        atomic_json(DEST/"scores.json",{"重叠组":overlap,"训练设备":retained,"训练航班":counts,"校准":cal_counts,"sites":scores,
                  "means":means,"decision":d,"论文权重SHA256":file_hash(paper_path),"源模型SHA256":file_hash(DEST/"source.joblib")})
        print(json.dumps({"隔离后":d,"训练轨迹":len(seen),"训练航班":len(y),"重叠组数":len(overlap)},ensure_ascii=False))


if __name__=="__main__":main()
