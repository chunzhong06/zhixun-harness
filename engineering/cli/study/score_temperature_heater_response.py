"""重读13日原始传感器，在完整已见暴露上公平比较恒定读法、回标选法及同工具查询论文。"""
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from score_gas_audit_feedback import DEST as AUDIT, read, digest, LocalStore, atomic_json, f1_score, recall_score
from prepare_temperature_trials import RAW, IDENTITY, DEST as INPUT
from harness.models.gas_heater_response import train_votes, read_views, VIEWS
from harness.models.gas_heater_archive import day_exposures
from harness.runtime.handoff import LocalAcceptance

DEST = AUDIT.parent / "gas-heater-response"
ANCHOR = "原入口局部纠错"
PRIMARY = "回标选择加热读法"
POLICIES = ("均匀覆盖","混合抽查","随机","BADGE")
CHANNELS = {"全通道":("full",slice(None),slice(None)), "换件":("replacement",slice(3),slice(3,6))}


def features():
    with RAW.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=IDENTITY:
            raise ValueError("已登记的原始温控归档身份不符")
    bank = {}
    with ZipFile(RAW) as archive, ZipFile(BytesIO(archive.read("gas-sensor-array-temperature-modulation.zip"))) as inner:
        for name in sorted(n for n in inner.namelist() if n.endswith(".csv")):
            bank.update(day_exposures(inner,name))
    return bank


def evaluate(site,policy,bank):
    rel, sc, tc = CHANNELS[site]
    root = INPUT/rel
    spec = read(root/"stream.json")
    def checked(path, checksum):
        import hashlib
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=checksum:
            raise ValueError("输入身份不符")
        return json.loads(data)
    src = checked(root/spec["source"],spec["source_sha256"])
    source = read_views(np.stack([bank[r["id"]] for r in src]),sc)
    np.testing.assert_allclose(source["各路平均"],[r["values"] for r in src],rtol=1e-8,atol=1e-8)
    sy = np.array([r["label"] for r in src])
    qdir = AUDIT/site/policy
    queries = read(qdir/"run.json")
    identity = digest([queries["identity"],Path(__file__).read_text(encoding="utf8"),IDENTITY])
    observed,history,labels = [],[],[]
    receiver = LocalAcceptance()
    out = DEST/site/policy
    receiver.output = out/"accepted"
    events = []
    evidence = {}
    for b,e in zip(spec["batches"],queries["events"],strict=True):
        rows = checked(root/b["inputs"],b["inputs_sha256"])
        ids = [r["id"] for r in rows]
        original = read(qdir/"accepted"/(e["request"]+".json"))["payload"]
        if digest([queries["identity"],original])!=e["request"] or original["ids"]!=ids or original["已见回标"]!=len(labels):
            raise ValueError("比较使用的查询、目标输入或回标时序不符")
        x = np.stack([bank[k] for k in ids])
        observed.append(x)
        views = read_views(x,tc)
        np.testing.assert_allclose(views["各路平均"],[r["values"] for r in rows],rtol=1e-8,atol=1e-8)
        seen = read_views(np.concatenate(history) if history else x[:0],tc) if history else {k:[] for k in VIEWS}
        pred = train_votes(source,sy,read_views(np.concatenate(observed),tc),seen,labels,views)
        pred[ANCHOR] = np.array(original["预测"]["相近读数残差"])
        order = list(pred)
        if labels:
            ref = evidence[ANCHOR]
            def quality(name):
                # 依已回标的事前预测估计各类F1，用相同八个有效样本的强基准作收缩。
                matrix = evidence[name] + 8*ref/max(1,ref.sum())
                denom = matrix.sum(0)+matrix.sum(1)
                return np.mean(np.divide(2*matrix.diagonal(),denom,out=np.zeros(3),where=denom>0))
            best = max(order,key=quality)
            if quality(best)<=quality(ANCHOR)+.01:
                best = ANCHOR
        else:
            best = ANCHOR
        pred[PRIMARY] = pred[best]
        payload = {"ids":ids,"预测":{k:v.tolist() for k,v in pred.items()},"采用":best,"已见反馈":len(labels),"查询回执":e["request"]}
        key = digest([identity,payload])
        if not receiver.submit(key,payload)["accepted"]:
            raise ValueError("温控读法未交付")
        events.append(key)
        if e["selected"]:
            feedback = checked(root/b["feedback"],b["feedback_sha256"])
            ix = [ids.index(k) for k in e["selected"]]
            y = np.array([feedback[k] for k in e["selected"]])
            if not np.array_equal(y,e["labels"]):
                raise ValueError("取回的答案与同预算查询不一致")
            history.append(x[ix])
            labels.extend(y.tolist())
            for k in order:
                cm = evidence.setdefault(k,np.zeros((3,3)))
                np.add.at(cm,(y-1,pred[k][ix]-1),e["逆包含概率或单位权"])
    oracle = read(root/"evaluation.json")
    y, cols = [],{}
    for key in events:
        p = read(receiver.output/(key+".json"))["payload"]
        if digest([identity,p])!=key:
            raise ValueError("已发结果身份不符")
        y.extend(oracle[k] for k in p["ids"])
        for k,v in p["预测"].items():
            cols.setdefault(k,[]).extend(v)
    scores = {k:{"F1":float(f1_score(y,v,average="macro")),"最差召回":float(recall_score(y,v,average=None).min())} for k,v in cols.items()}
    atomic_json(out/"run.json",{"identity":identity,"requests":events,"scores":scores,"回标":len(labels)})
    return scores


def run():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("温控开发已完成")
        bank=features()
        ids=sorted(bank)
        np.savez_compressed(DEST/"exposures.npz",ids=ids, features=np.stack([bank[k] for k in ids]))
        streams = {s:{q:evaluate(s,q,bank) for q in POLICIES} for s in CHANNELS}
        old = read(AUDIT/"scores.json")
        arms = next(iter(next(iter(streams.values())).values()))
        means = {q:{k:{m:float(np.mean([v[q][k][m] for v in streams.values()])) for m in ("F1","最差召回")}
                    for k in arms} for q in POLICIES}
        # 旧方法在两个相同装置读法上的实际分数也参与比较，不把缺少温控字段的其他来源算作本实验。
        for q in POLICIES:
            for k in old["streams"]["全通道"][q]:
                means[q].setdefault(k,{m:float(np.mean([old["streams"][site][q][k][m] for site in CHANNELS])) for m in ("F1","最差召回")})
        from score_gas_rank_readout import DEST as rank_folder
        rank = read(rank_folder/"scores.json")
        extra = []
        for q in ("均匀覆盖","随机","边界","BADGE","历史辨别力"):
            old_runs = [rank["streams"][site][q] for site in CHANNELS]
            names = set.intersection(*(set(v) for v in old_runs))
            for k in sorted(names):
                means.setdefault(q,{}).setdefault(k,{m:float(np.mean([v[k][m] for v in old_runs])) for m in ("F1","最差召回")})
                if q not in ("BADGE","历史辨别力") and not k.startswith(("按现场","全局历史","近邻历史")):
                    extra.append((q,k))
        fixed = extra+[(q,k) for q in ("混合抽查","均匀覆盖","随机") for k in [*arms,*audit_fixed()] if k!=PRIMARY]
        metric=lambda a:means[a[0]][a[1]]["F1"]
        a=("混合抽查",PRIMARY)
        f=max(fixed,key=metric)
        p=max((("BADGE",k) for k in means["BADGE"]),key=metric)
        gap=metric(a)-max(metric(f),metric(p))
        d={"主候选":a,"固定":f,"论文":p,"现场F1":metric(a),"固定F1":metric(f),"论文F1":metric(p),"最小领先":gap,
           "进独立":gap>=.01 and means[a[0]][a[1]]["最差召回"]>=means[f[0]][f[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":d})
        print(json.dumps({"温控开发":d,"源与开发暴露":len(bank)},ensure_ascii=False))


def audit_fixed():
    from harness.methods.gas_audit_feedback import ARMS
    return ARMS


if __name__=="__main__":
    run()
