"""比较两种来源切换顺序，复用同一套近邻查询，把反馈、预测、事后评分分开。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import OUT as MIXED, OTHER, archived, old_step, source_window, unlabelled
from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, digest, metric, official, read, reference_folders, remaining_cycles
from harness.data.cmapss_input import condition_key
from harness.models.device_neighbors import DeviceNeighborUpdate

DEST=BASE/"engine-neighbor-updates"/"switch"


def fifth_distance(action,data):
    """与预测器用同一标准化和欧氏距离，每台训练设备仅保留一个最近观测。"""
    q=np.clip((data[:,:21]-action.mean)/action.scale,-10,10)
    order=np.argsort(action.units,kind="stable")
    units=action.units[order]
    starts=np.r_[0,1+np.flatnonzero(units[1:]!=units[:-1])]
    bank=action.bank[order]
    norms=(bank**2).sum(axis=1)
    distances=[]
    for i in range(0,len(q),128):
        chunk=q[i:i+128]
        squared=norms[None,:]+(chunk**2).sum(axis=1)[:,None]-2*chunk@bank.T
        per_device=np.minimum.reduceat(squared,starts,axis=1)
        distances.extend(np.sqrt(np.maximum(np.partition(per_device,4,axis=1)[:,4],0)/bank.shape[1]))
    return np.array(distances)


def observations(action,rows):
    if not rows:
        return None
    data=np.array([r["values"] for r in rows])
    plain=action.model.predict(data)
    blend=action.predict(rows,True)
    return plain,blend,fifth_distance(action,data),[condition_key(v[21:]) for v in data]


def run(cohort):
    package,checksum,_=official("engine")
    fixed,_=reference_folders(cohort,"mature")
    states={c:read(reference_folders(c,"mature")[0]/"state.json")["samples"] for c in (cohort,OTHER[cohort])}
    folder=BASE/"live-first"/("public" if cohort=="engine" else "")/cohort
    ref=read(MIXED/cohort/"issued.json")
    fit_ids={i for i,r in states[cohort].items() if r["role"]=="fit"}
    train=[r for r in package.train if r.sample_id in fit_ids]
    fit=np.array([r.values for r in train]),np.array([r.label for r in train])
    blocks,target_truth=unlabelled("engine")
    seq=[("before",source_window(states[cohort],r)) for r in range(4,8)]
    seq.extend(("unlabelled",([],[],b)) for b in blocks)
    seq.extend(("new_feedback",source_window(states[OTHER[cohort]],r)) for r in range(4))
    seq.extend(("return",source_window(states[cohort],r)) for r in (8,9))
    arms=("current","anchor","keep","confirm","coverage")
    outputs={k:{} for k in arms}
    stages,units,history,trace={},{},[],{}
    counts=defaultdict(int)
    archive,=(ROOT/"data/official/engine").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z,_trainer(z,Path(tmp),"C") as trainer:
        anchor,identity,records=archived(folder,trainer,Path(tmp),"engine")
        action=DeviceNeighborUpdate(anchor,identity,train)
        live=anchor
        for r,record in enumerate(records[:4]):
            payload=read(folder/record["submitted"])["payload"]
            if payload["model_id"]!=record["selected"]:
                raise ValueError("热身交接身份不同")
            for k in arms:
                outputs[k].update(payload["predictions"])
            for row in source_window(states[cohort],r)[2]:
                stages[row["sample_id"]],units[row["sample_id"]]="warmup",row["unit_id"]
        used=False
        radius=None
        settings=set()
        query_seconds=update_seconds=0.
        for stage,(explore,confirm,serve) in seq:
            ids=[r["sample_id"] for r in serve]
            if len(set(ids))!=len(ids) or set(ids)&stages.keys() or any(r["label"] is not None for r in serve):
                raise ValueError("服务样本与合法时序不匹配")
            start=perf_counter()
            live,_,receipt,pred=old_step("engine","fixed",live,fit,explore,confirm,serve,None)
            update_seconds+=perf_counter()-start
            counts["regular_fits"]+=int(receipt["physical_fit"])
            outputs["current"].update(pred)
            start=perf_counter()
            confirmed=False
            event={"stage":stage,"rows":len(ids),"current_adopted":receipt["adopted"]}
            if explore and confirm:
                radius,settings=None,set()
                observed=[]
                for label,rows in (("explore",explore),("confirm",confirm)):
                    obs=observations(action,rows)
                    observed.append(obs)
                    truth=np.array([r["label"] for r in rows])
                    a,b=[float(np.sqrt(np.mean((v-truth)**2))) for v in obs[:2]]
                    event[label]=[a,b]
                    if b>=a-1e-12:
                        break
                if len(observed)==2 and all(event[k][1]<event[k][0]-1e-12 for k in ("explore","confirm")):
                    confirmed=used=True
                    settings=set(observed[0][3])&set(observed[1][3])
                    radius=float(np.quantile(observed[1][2],.95,method="higher"))
            base,blend,dist,conditions=observations(action,serve)
            query_seconds+=perf_counter()-start
            reuse=np.full(len(ids),confirmed,dtype=bool)
            if not explore and not confirm and radius is not None:
                reuse=(dist<=radius)&np.array([k in settings for k in conditions])
            for arm,mask in (("anchor",False),("keep",used),("confirm",confirmed),("coverage",reuse)):
                selected=np.broadcast_to(mask,base.shape)
                outputs[arm].update(zip(ids,map(float,np.where(selected,blend,base)),strict=True))
                counts[arm]+=int(selected.sum())
            for j,r in enumerate(serve):
                i=r["sample_id"]
                stages[i],units[i]=stage,r["unit_id"]
                trace[i]={"base":float(base[j]),"blend":float(blend[j]),"distance":float(dist[j]),
                          "condition":conditions[j],"coverage":bool(reuse[j])}
            history.append({**event,"confirmed":confirmed,"kept":used,"radius":radius,
                            "settings":sorted(settings),"coverage_used":int(reuse.sum())})
    if stages!=ref["stages"] or ref["prefix_sha256"]["paper_once"]!=identity:
        raise ValueError("历史对照的时序或当前默认首提后的父版不符")
    for k in ("old","paper_once","refresh"):
        outputs[k]=ref["streams"][k]
    # 论文首提后的规则和当前默认在这两条流有相同父权重；独立计算完整更新序列。
    delta=max(abs(v-outputs["paper_once"][i]) for i,v in outputs["current"].items())
    if delta>1e-8:
        raise ValueError(f"当前默认的独立运行未复现历史同策略输出 {delta}")
    out=DEST/cohort
    sealed=out/"issued.json"
    atomic_json(sealed,{"source":checksum,"parent":identity,"training":digest(sorted(fit_ids)),"stages":stages,"units":units,
                        "history":history,"trace":trace,"predictions":outputs,"max_abs_ref_difference":delta})
    saved=read(sealed)
    truth={r.sample_id:r.label for r in package.validation}
    truth.update(remaining_cycles())
    truth.update(target_truth)
    stats={}
    for k,p in saved["predictions"].items():
        if set(p)!=set(stages):
            raise ValueError("实际交出分片不同")
        groups=defaultdict(dict)
        for i,v in p.items():
            groups[units[i]][i]=v
        per_unit={u:metric("engine",truth,g) for u,g in groups.items()}
        stats[k]={"score":metric("engine",truth,p),"worst_device":max(per_unit.values()),"device_scores":per_unit,
                  "stages":{s:metric("engine",truth,{i:v for i,v in p.items() if stages[i]==s}) for s in set(stages.values())},
                  "window_scores":[metric("engine",truth,{r["sample_id"]:p[r["sample_id"]] for r in serve}) for _,(_,_,serve) in seq]}
    evidence={}
    for keep in (True,False):
        ids=[i for i,t in trace.items() if stages[i]=="unlabelled" and t["coverage"]==keep]
        evidence[str(keep)]={"rows":len(ids),"rmse":{k:metric("engine",truth,{i:trace[i][k] for i in ids}) for k in ("base","blend")} if ids else {}}
    atomic_json(out/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":stats,
                "after_warmup_counts":dict(counts),"shared_query_and_scoring_seconds":query_seconds,
                "regular_update_and_scoring_seconds":update_seconds,"unlabelled_coverage":evidence})
    print(cohort,{k:(round(v["score"],3),round(v["stages"]["unlabelled"],3),round(v["worst_device"],3)) for k,v in stats.items()},evidence,flush=True)


if __name__=="__main__":
    for name in ("engine","phm08"):
        run(name)
