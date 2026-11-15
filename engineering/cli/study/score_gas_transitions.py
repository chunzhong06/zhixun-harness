"""从既有全流交接追到真正切换的窗口，以相同32条回标试验两种状态转移模型。"""
from hashlib import file_digest
import json
from zipfile import ZipFile
import numpy as np
from sklearn.metrics import f1_score,recall_score
from score_dynamic_gas_memory import RAW,DATA,EXPECTED,DEST as PRIOR,PRIOR as RECEIPTS,seconds,LocalStore,LocalAcceptance,atomic_json,digest,DynamicEncoder,StreamRegression
from harness.models.gas_transitions import EdgeEncoder,TransitionModels,PRIMARY

DEST=PRIOR.parent/"gas-transitions"


def extract(raw,edge,dyn):
    times,x,_=raw;window=times//10
    split=np.r_[0,np.flatnonzero(np.diff(window))+1,len(window)]
    a,b,c=[],[],[]
    for i,j in zip(split[:-1],split[1:],strict=True):
        a.append(edge.encode(x[i:j]))
        views,_=dyn.encode(x[i:j])
        b.append(views[1]);c.append(views[-1])
    return np.stack(a),np.stack(b),np.stack(c)


def run_stream(site,origin,raw,labels,meta):
    edge=EdgeEncoder(raw[origin][1]);enc=DynamicEncoder(raw[origin][1])
    s,d,p=extract(raw[origin],edge,enc)
    y=np.array([v["label"]==2 for v in labels[origin]])
    models=TransitionModels().fit(s,y)
    fixed=StreamRegression().fit(d,y);paper=StreamRegression().fit(p,y)
    edge.past=[];enc.reset();tx,td,tp=extract(raw[site],edge,enc)
    lookup={r["id"]:r["label"] for r in labels[site]}
    events=json.loads((RECEIPTS/site/"ongoing/run.json").read_bytes())["accepted"]
    receiver=LocalAcceptance();receiver.output=DEST/site/"accepted"
    issued=[];cursor=0
    for prev,event in zip(meta["requests"],events,strict=True):
        parent=json.loads((PRIOR/site/"accepted"/(prev+".json")).read_bytes())["payload"]
        if digest([site,parent])!=prev or parent["上轮"]!=event["request"]:raise ValueError("源回执变化")
        block=parent["ids"];sl=slice(cursor,cursor+len(block))
        if parent["回标已用"]!=len(models.site) or block!=[r["id"] for r in labels[site][sl]]:raise ValueError("成本或窗口变化")
        main=fixed.outputs(td[sl])[:,1];pp=paper.outputs(tp[sl])[:,1]
        for val,ref in ((main,"近期响应差分/累计回标"),(pp,"差分与论文记忆合用/累计回标")):
            if (1+(val>=.5)).tolist()!=parent["预测"][ref]:raise ValueError("动态对照输出变化")
        out,bank,z=models.predict(tx[sl],np.clip(main,0,1))
        cols=dict(parent["预测"])
        for k,v in out.items():
            cols[k]=(1+(v>=.5)).tolist()
            cols["状态与论文各半/"+k]=(1+((v+np.clip(pp,0,1))/2>=.5)).tolist()
        pkt={"ids":block,"上轮":prev,"预测":cols,"累计反馈":len(models.site),"已有相邻回标对":sum(i-1 in models.ref for i in models.site)}
        h=digest([site,pkt])
        if not receiver.submit(h,pkt)["accepted"]:raise ValueError("状态预测未发放")
        issued.append(h)
        if event["returned"]:
            ix=np.array([block.index(i) for i in event["returned"]]);yy=np.array([lookup[i]==2 for i in event["returned"]])
            fixed.update(td[sl][ix],yy);paper.update(tp[sl][ix],yy)
            models.update(cursor+ix,z[ix],yy,bank[ix])
        cursor+=len(block)
    true=[];cols={}
    for h in issued:
        pkt=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
        if digest([site,pkt])!=h:raise ValueError("状态交付记录改变")
        true.extend(lookup[i] for i in pkt["ids"])
        for k,v in pkt["预测"].items():cols.setdefault(k,[]).extend(v)
    y=np.array(true);switch=np.r_[False,np.diff(y)!=0]
    scores={k:{"F1":float(f1_score(y,v,average="macro")),"最差召回":float(recall_score(y,v,average=None).min()),
               "切换错":int(np.count_nonzero((y!=v)&switch)),"稳定错":int(np.count_nonzero((y!=v)&~switch))} for k,v in cols.items()}
    controls=[meta["decision"]["固定"],*[k for k in out if k!=PRIMARY],*["状态与论文各半/"+k for k in out if k!=PRIMARY]]
    winner=max(controls,key=lambda k:scores[k]["F1"])
    best=max([meta["decision"]["论文"],*["状态与论文各半/"+k for k in out]],key=lambda k:scores[k]["F1"])
    gap=scores[PRIMARY]["F1"]-max(scores[k]["F1"] for k in (winner,best))
    dec={"现场":PRIMARY,"固定":winner,"论文":best,"现场F1":scores[PRIMARY]["F1"],"固定F1":scores[winner]["F1"],"论文F1":scores[best]["F1"],
         "差值":gap,"通过":gap>=.01 and scores[PRIMARY]["最差召回"]>=scores[winner]["最差召回"]-.02}
    return {"decision":dec,"scores":scores,"requests":issued,"切换窗口":int(switch.sum()),"窗口":len(y),"已回标":len(models.site)}


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已完成状态试验")
        with RAW.open("rb") as f:
            if file_digest(f,"sha256").hexdigest()!=EXPECTED:raise ValueError("原始混合物文件不同")
        with ZipFile(RAW) as archive:
            raw={}
            for site,filename in (("co","ethylene_CO.txt"),("methane","ethylene_methane.txt")):
                with archive.open(filename) as stream:raw[site]=seconds(stream)
        hist=json.loads((PRIOR/"scores.json").read_bytes())
        rows={s:json.loads((DATA/s/"source.json").read_bytes()) for s in raw}
        result={s:run_stream(s,t,raw,rows,hist[s]) for s,t in (("co","methane"),("methane","co"))}
        atomic_json(DEST/"scores.json",result)
        print(json.dumps({s+"状态切换":r["decision"] for s,r in result.items()},ensure_ascii=False))
