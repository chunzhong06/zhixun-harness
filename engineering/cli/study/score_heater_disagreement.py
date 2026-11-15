"""从相同源端和冻结的原始阶段读数出发，各条查询轨迹真实训练并先发预测再回标。"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score

from score_gas_global_calibration import DEST as PRIOR, ENTRY, FIRST, CHANNELS, INPUT, ANCHOR, extract, by_type, ordered_label
from score_temperature_heater_response import read, digest, LocalStore, LocalAcceptance, atomic_json, train_votes, read_views
from engineering.cli.workflow.gas_heater_stream import gas_query_stream, audit, checked
from harness.methods.heater_disagreement import POLICIES, DisagreementFeedback

DEST=PRIOR.parent/"heater-disagreement"
PRIMARY="学习各类分歧后发布"
PAPER="共享洁净校准"


def run_site(site,policy,phase,calibration):
    rel,source_channels,target_channels=CHANNELS[site]
    folder=INPUT/rel
    spec=read(folder/"stream.json")
    src=checked(folder/spec["source"],spec["source_sha256"])
    ids=[r["id"] for r in src];sy=np.array([r["label"] for r in src])
    a=read_views(np.stack([phase[k] for k in ids]),source_channels)
    b=np.stack([calibration[k][1,source_channels] for k in ids])
    query_src=[{"values":v.tolist(),"label":int(y)} for v,y in zip(a["相位电阻"],sy,strict=True)]
    prior_run=read(PRIOR/site/"BADGE"/"加热"/"run.json")
    components=prior_run["components"]
    ctx=DisagreementFeedback()
    all_x,seen,history_b,y_seen,history_q,events=[],[],[],[],[],[]
    out=DEST/site/policy
    receiver=LocalAcceptance();receiver.output=out/"accepted"
    identity=digest([spec,policy,components,Path(__file__).read_text(encoding="utf8")])
    for i,batch in enumerate(spec["batches"]):
        rows=checked(folder/batch["inputs"],batch["inputs_sha256"])
        ids=[r["id"] for r in rows]; data=np.stack([phase[k] for k in ids]);all_x.append(data)
        x=read_views(data,target_channels)
        history=read_views(np.concatenate(seen),target_channels) if seen else {k:[] for k in x}
        pred=train_votes(a,sy,read_views(np.concatenate(all_x),target_channels),history,y_seen,x)
        target=np.stack([calibration[k][1,target_channels] for k in ids])
        raw,spread=by_type(np.concatenate([b,*history_b]),np.r_[sy,np.asarray(y_seen,dtype=int)],target,components)
        anchor=pred[ANCHOR]
        cal=ordered_label(raw)
        alt=np.mean(np.stack([pred[k] for k in ("相位比仅源","相位电阻仅源","相位电阻仅现场")])==cal,axis=0)
        context=ctx.context(anchor,cal,raw,spread,alt)
        pred.update({PAPER:cal,PRIMARY:ctx.publish(context)})
        for frac in (.25,.5,.75):
            pred[f"固定论文分量{frac}"]=ordered_label(raw*frac+anchor*(1-frac))
        item={"ids":ids,"预测":{k:v.tolist() for k,v in pred.items()},"已见回标":len(y_seen)}
        key=digest([identity,item])
        if not receiver.submit(key,item)["accepted"]:raise ValueError("未完成当批交接")
        ix=[]
        if i+1<len(spec["batches"]):
            if policy in POLICIES:
                ix=ctx.query(policy,ids,x["相位电阻"],context,8)
            else:
                scale,model,_=gas_query_stream.train(query_src,history_q,3)
                z=scale.transform(x["相位电阻"])
                ix,_=audit.sample(policy,ids,x["相位电阻"],z,model.predict_proba(z),8,17+i)
            if len(set(ix))!=min(8,len(ids)):raise ValueError("实际回标数有误")
            truth=checked(folder/batch["feedback"],batch["feedback_sha256"])
            y=np.array([truth[ids[j]] for j in ix])
            ctx.receive(context[ix],y);seen.append(data[ix]);history_b.append(target[ix])
            y_seen.extend(y.tolist())
            history_q.extend({"x":x["相位电阻"][j].tolist(),"y":int(v)} for j,v in zip(ix,y,strict=True))
        events.append({"key":key,"selected":[ids[j] for j in ix],"分歧查询数":int(np.sum(anchor[ix]!=cal[ix]))})
    y=[];columns={};oracle=read(folder/"evaluation.json")
    for e in events:
        p=read(out/"accepted"/(e["key"]+".json"))["payload"]
        if digest([identity,p])!=e["key"]:raise ValueError("交接身份不匹配")
        y.extend(oracle[k] for k in p["ids"])
        for k,v in p["预测"].items():columns.setdefault(k,[]).extend(v)
    table={k:{"F1":float(f1_score(y,v,average="macro")),"最差召回":float(recall_score(y,v,average=None).min())} for k,v in columns.items()}
    atomic_json(out/"run.json",{"identity":identity,"events":events,"scores":table,"回标":len(y_seen)})
    return table


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("本项已经交付")
        with np.load(FIRST/"exposures.npz") as z:phase=dict(zip(z["ids"],z["features"],strict=True))
        cal=extract()
        policies=(*POLICIES,"BADGE","混合抽查")
        results={s:{p:run_site(s,p,phase,cal) for p in policies} for s in CHANNELS}
        means={p:{k:{m:float(np.mean([v[p][k][m] for v in results.values()])) for m in ("F1","最差召回")}
                  for k in results["全通道"][p]} for p in policies}
        get=lambda a:means[a[0]][a[1]]["F1"]
        fixed=max(((p,k) for p in policies if p not in ("按历史分歧分配","BADGE") for k in means[p] if k!=PRIMARY),key=get)
        paper=max((("BADGE",k) for k in means["BADGE"]),key=get)
        old=read(PRIOR/"scores.json")
        old_entry=read(ENTRY/"scores.json")
        arm=("按历史分歧分配",PRIMARY)
        f=max(get(fixed),old["decision"]["固定F1"])
        p=max(get(paper),old["decision"]["论文F1"])
        gap=get(arm)-max(f,p)
        protect=max(means[fixed[0]][fixed[1]]["最差召回"] if get(fixed)>=old["decision"]["固定F1"] else 0,
                    old["means"][old["decision"]["固定"][0]][old["decision"]["固定"][1]]["最差召回"] if f==old["decision"]["固定F1"] else 0)
        for s in CHANNELS:
            expected=old_entry["streams"][s]["BADGE-加热"][ANCHOR]["F1"]
            if abs(results[s]["BADGE"][ANCHOR]["F1"]-expected)>1e-12:raise ValueError("同查询原基准未复现")
        decision={"主候选":arm,"现场F1":get(arm),"本项固定":fixed,"含原最强固定F1":f,"论文":paper,"含原最强论文F1":p,
                  "最小差值":gap,"进独立":gap>=.01 and means[arm[0]][PRIMARY]["最差召回"]>=protect-.02}
        atomic_json(DEST/"scores.json",{"streams":results,"means":means,"decision":decision})
        print(json.dumps({"开发交付":decision},ensure_ascii=False))


if __name__=="__main__":main()
