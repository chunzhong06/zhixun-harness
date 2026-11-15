"""弱响应训练只读源和开发原件，论文用同样的衰减训练样本重新拟合。"""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, recall_score

from score_plume_training import DEST as PRIOR, DATA, PAPER, LocalStore, LocalAcceptance, atomic_json, digest, read, split_members
from score_turbulent_contrast import DEST as CONTRAST, PairedGasContrast, exposure_response
from score_turbulent_source import DEST as PAPER_DIR, paper
from harness.models.weak_plume import VIEWS, PRIMARY, WeakInstallation, augment, sample_cycle

DEST = PRIOR.parent / "weak-plume"
ANCHOR = "固定去噪线性"


def read_parts():
    meta = read(DATA / "source.json")
    recorded = DATA / "development.json"
    archive = DATA / "uci309.zip"
    if sha256(archive.read_bytes()).hexdigest()!=meta["原包SHA256"] or sha256(recorded.read_bytes()).hexdigest()!=meta["开发文件SHA256"]:
        raise ValueError("原件或开发清单变化")
    records = read(recorded)
    waves, legacy, groups = {}, {}, []
    with ZipFile(archive) as handle:
        selected = split_members(handle.infolist())
        for part in ("source","development"):
            waves[part], legacy[part] = [], []
            for (uid,item,_), row in zip(selected[part],records[part],strict=True):
                raw = handle.read(item)
                if uid in meta["封存编号"] or uid != row["id"] or sha256(raw).hexdigest()!=row["原件SHA256"]:
                    raise ValueError("风洞释放身份变化")
                waves[part].append(sample_cycle(np.loadtxt(BytesIO(raw), delimiter=",")))
                legacy[part].append(exposure_response(raw))
                if part == "source":
                    groups.append(Path(item.filename).name.lower().split("_")[2:])
    return {k:np.stack(v) for k,v in waves.items()}, {k:np.stack(v) for k,v in legacy.items()}, np.asarray(groups), records


def paper_transfer(x, y, group, service, original):
    from scipy.special import expit
    out = {}
    weights = {}
    for view in VIEWS:
        augmented, truth = augment(x,y,group,view)
        a = (3110*expit(augmented)).reshape(len(truth),-1)
        b = (3110*expit(service)).reshape(len(service),-1)
        distance = cdist(a,a,"sqeuclidean") / a.shape[1]
        band = float(np.median(distance[np.triu_indices(len(a),1)]))
        factor = original["gamma"] / band
        dual = paper(np.exp(-distance*factor),truth,original["cost"])
        values = np.exp(-cdist(b,a,"sqeuclidean")*(factor/a.shape[1])) @ dual
        out["论文重训/"+view] = 1+(values>=0)
        weights[view] = {"templates":a, "dual":dual, "factor":factor/a.shape[1]}
    return out, weights


def run():
    x, old_x, groups, records = read_parts()
    previous = read(PRIOR / "scores.json")
    parent = previous["request"]
    old = read(PRIOR/"accepted"/(parent+".json"))["payload"]
    ids = [r["id"] for r in records["development"]]
    if digest(old)!=parent or ids!=old["ids"]:
        raise ValueError("开发交付身份变化")
    y = np.array([r["label"] for r in records["source"]])
    anchor, cuts = joblib.load(CONTRAST/"models.joblib")
    _, code = np.unique(groups,axis=0,return_inverse=True)
    held = np.empty(len(y),int)
    for fold in range(5):
        test = code%5 == fold
        model = PairedGasContrast().fit(old_x["source"][~test],groups[~test])
        # 只调用线性读法；留整种配方时有些背景的参照组会为空。
        whitened = model.scale.transform(old_x["source"][test]) @ model.whitener
        held[test] = 1+(model.classifier.predict_proba(whitened)[:,1]>=.5)
    base = 1+(anchor.outputs(old_x["development"])[ANCHOR]>=cuts[ANCHOR])
    if base.tolist()!=old["预测"][ANCHOR]:
        raise ValueError("原强固定未复现")
    adaptive = WeakInstallation().fit(x["source"],y,groups,held)
    pp = joblib.load(PAPER_DIR/"models.joblib")["models"][PAPER]
    papers, paper_weights = paper_transfer(x["source"],y,groups,x["development"],pp)
    if papers["论文重训/"+VIEWS[0]].tolist()!=old["预测"][PAPER]:
        raise ValueError("原始训练的论文读法未复现")
    bundle = DEST/"release.joblib"
    joblib.dump({"安装":adaptive,"固定":anchor,"阈值":cuts[ANCHOR],"论文":paper_weights},bundle,compress=3)
    issued = joblib.load(bundle)["安装"].predict(x["development"],base)
    columns = {k:np.asarray(v) for k,v in old["预测"].items()}
    columns.update(issued)
    columns.update(papers)
    fixed = [previous["decision"]["固定"], *[k for k in issued if k!=PRIMARY],*papers]
    paperarms = [previous["decision"]["论文"], *papers]
    for first in (ANCHOR, *papers):
        for second in [*VIEWS, "三种训练固定平均"]:
            for mode, op in (("任一报有",np.maximum),("均报有",np.minimum)):
                name = f"{first}/{second}/{mode}"
                columns[name] = op(columns[first],columns[second])
                fixed.append(name)
                if first in papers:
                    paperarms.append(name)
    # 最强固定与新增论文各自也可对对方取并集或交集。
    for p in papers:
        for name, op in (("任一报有",np.maximum),("均报有",np.minimum)):
            k = f"{p}/原去噪/{name}"
            columns[k] = op(columns[p],base)
            fixed.append(k)
            paperarms.append(k)
    payload = {"ids":ids,"预测":{k:v.tolist() for k,v in columns.items()},"权重SHA256":sha256(bundle.read_bytes()).hexdigest(),"上轮回执":parent}
    key = digest(payload)
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    if not receiver.submit(key,payload)["accepted"]:
        raise ValueError("未交付")
    sent = read(receiver.output/(key+".json"))["payload"]
    if digest(sent)!=key:
        raise ValueError("回执改变")
    truth = np.array([v["label"] for v in records["development"]])
    scores = {k:{"F1":float(f1_score(truth,v,average="macro")),"最弱类召回":float(recall_score(truth,v,average=None).min()),
                 "正确":int(np.sum(truth==v)),"修回":int(np.sum((truth==v)&(truth!=base))),
                 "改错":int(np.sum((truth!=v)&(truth==base)))} for k,v in sent["预测"].items()}
    top = max(fixed,key=lambda k:scores[k]["F1"])
    best_paper = max(paperarms,key=lambda k:scores[k]["F1"])
    gap = scores[PRIMARY]["F1"]-max(scores[k]["F1"] for k in (top,best_paper))
    d = {"现场":PRIMARY,"固定":top,"论文":best_paper,"现场F1":scores[PRIMARY]["F1"],"固定F1":scores[top]["F1"],
         "论文F1":scores[best_paper]["F1"],"差值":gap,"进入独立":gap>=.01 and scores[PRIMARY]["最弱类召回"]>=scores[top]["最弱类召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"scores":scores,"request":key})
    print(json.dumps({"弱释放训练":d},ensure_ascii=False))


if __name__ == "__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("同项训练已有回执")
        run()
