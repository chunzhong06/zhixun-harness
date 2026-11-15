"""在各查询办法自己的反馈轨迹上运行同一模型库，核算查询和发布的联合收益。"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from engineering.cli.workflow.field_classification import checked
from harness.contracts import digest
from harness.methods.gas_class_reference import ClassReference
from harness.models.field_classification import weights
from harness.methods.gas_response_votes import kernel_models
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from score_gas_query_risk import BASE, DEST as PRIOR, STREAMS, read

DEST = PRIOR.parent / "gas-joint-query"
QUERIES = ("随机","边界","均匀覆盖","BADGE","历史辨别力")
TRAIN = ("原刻度", "独立刻度", "固定校准", "只用现场", "固定源核", "固定合训核", "固定现场核")


def publish(site, relative, query):
    folder = BASE / relative
    spec = read(folder / "stream.json")
    classes = spec["classes"]
    klist = np.arange(1, classes+1)
    src = checked(folder / spec["source"],spec["source_sha256"])
    sx,sy = np.array([r["values"] for r in src]),np.array([r["label"] for r in src])
    learner = ClassReference(sx,sy)
    source_svm = SVC(C=4,gamma="scale",class_weight="balanced",random_state=17).fit(learner.source,sy)
    qfolder = PRIOR / site / query / "17"
    frozen = read(qfolder / "run.json")
    out = DEST / site / query
    receiver = LocalAcceptance()
    receiver.output = out / "accepted"
    ident = digest([frozen["identity"], digest(Path(__file__).read_text(encoding="utf-8"))])
    shown, labels, historical, observed, events = [],[],[],[],[]
    for b,e in zip(spec["batches"],frozen["accepted"],strict=True):
        rows = checked(folder/b["inputs"],b["inputs_sha256"])
        ids, x = [r["id"] for r in rows], np.array([r["values"] for r in rows])
        past = np.asarray(shown).reshape(-1,x.shape[1])
        y = np.asarray(labels,dtype=int)
        observed.append(x)
        pool = np.vstack(observed)
        fixed = {k:learner.predict(pool,past,y,x,k).argmax(1)+1 for k in TRAIN[:4]}
        fixed.update(kernel_models(sx,sy,x,past,y,pool,source_svm))
        original = read(qfolder/"accepted"/(e["request"]+".json"))["payload"]
        if (original["ids"] != ids or digest([frozen["identity"],original]) != e["request"]
                or fixed["原刻度"].tolist()!=original["prediction"]):
            raise ValueError("查询器冻结输入或原模型预测不一致")
        preds = np.column_stack([fixed[k] for k in TRAIN])
        best = 0
        best_local = np.zeros(len(x),dtype=int)
        if historical:
            hist = np.vstack(historical)
            rates = np.array([f1_score(y,v,labels=klist,average="macro",zero_division=0) for v in hist.T])
            best = rates.argmax()
            scale = StandardScaler().fit(pool)
            ds = cdist(scale.transform(x),scale.transform(past),"sqeuclidean") / x.shape[1]
            ix = np.argsort(ds,axis=1,kind="stable")[:,:min(12,len(past))]
            correct = (hist == y[:,None]).astype(float)
            # 同一回标有各工具当时的真实预测；类频率和少量近邻只用既有反馈估计。
            weight = weights(y,len(y))[ix]
            support = np.exp(-np.take_along_axis(ds,ix,axis=1))*weight
            local = np.einsum("nk,nkm->nm",support,correct[ix])
            best_local = np.argmax(local + 8*rates,axis=1)
        outputs = {k:v.tolist() for k,v in fixed.items()}
        outputs["全局历史发布"] = preds[:,best].tolist()
        outputs["近邻历史发布"] = preds[np.arange(len(x)),best_local].tolist()
        p = {"ids":ids,"predictions":outputs,"全局采用":TRAIN[best],"查询回执":e["request"],"已见标签":len(y)}
        request = digest([ident,p])
        if not receiver.submit(request,p)["accepted"]:
            raise ValueError("现场模型未完成交付")
        events.append(request)
        if e["selected"]:
            returned = checked(folder/b["feedback"],b["feedback_sha256"])
            ix = [ids.index(k) for k in e["selected"]]
            shown.extend(x[ix].tolist())
            labels.extend(returned[k] for k in e["selected"])
            historical.append(preds[ix])
    oracle = read(folder/"evaluation.json")
    yall, cols = [], {k:[] for k in outputs}
    for req in events:
        p = read(receiver.output/(req+".json"))["payload"]
        if digest([ident,p]) != req:
            raise ValueError("已发预测发生变化")
        yall.extend(oracle[i] for i in p["ids"])
        for k,v in p["predictions"].items():
            cols[k].extend(v)
    scores = {k:{"F1":float(f1_score(yall,v,labels=klist,average="macro",zero_division=0)),
                 "最差召回":float(recall_score(yall,v,labels=klist,average=None,zero_division=0).min())} for k,v in cols.items()}
    atomic_json(out/"run.json",{"identity":ident,"requests":events,"scores":scores,"回标":len(labels)})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已有交叉实验")
        streams = {}
        for site,relative in STREAMS.items():
            streams[site] = {q:publish(site,relative,q) for q in QUERIES}
            print(json.dumps({site:"完整查询轨迹与发布完成"},ensure_ascii=False),flush=True)
        means = {q:{m:{metric:float(np.mean([r[q][m][metric] for r in streams.values()]))
                       for metric in ("F1","最差召回")} for m in (*TRAIN,"全局历史发布","近邻历史发布")} for q in QUERIES}
        fixed = [(q,m) for q in QUERIES if q not in ("BADGE","历史辨别力") for m in TRAIN]
        paper = [("BADGE",m) for m in (*TRAIN,"全局历史发布","近邻历史发布")]
        dynamic = [(q,m) for q in QUERIES if q!="BADGE" for m in (*TRAIN,"全局历史发布","近邻历史发布")
                   if q=="历史辨别力" or m not in TRAIN]
        metric = lambda a:means[a[0]][a[1]]["F1"]
        a,b,c = (max(group,key=metric) for group in (dynamic,fixed,paper))
        decision = {"主候选":a,"最强固定":b,"最强论文查询":c,"主候选F1":metric(a),"固定F1":metric(b),
                    "论文查询F1":metric(c),"对固定增量":metric(a)-metric(b),"对论文增量":metric(a)-metric(c),
                    "可进独立":metric(a)-max(metric(b),metric(c))>=.01 and means[a[0]][a[1]]["最差召回"]>=means[b[0]][b[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":decision})
        print(json.dumps({"联合取舍":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
