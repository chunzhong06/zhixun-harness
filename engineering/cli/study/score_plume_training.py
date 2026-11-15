"""从UCI309原始完整释放进行装机试验，只读登记的源和开发成员，按接收结果评分。"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score
from score_turbulent_disagreement import DEST as PRIOR,DATA,PAPER,LocalAcceptance,LocalStore,atomic_json,digest,read,split_members
from harness.models.plume_training import FIXED,CV,PRIMARY,PlumeInstallation,read_windows

DEST=PRIOR.parent/"plume-training"


def data():
    meta=read(DATA/"source.json"); archive=DATA/"uci309.zip"; records=read(DATA/"development.json")
    if sha256(archive.read_bytes()).hexdigest()!=meta["原包SHA256"] or sha256((DATA/"development.json").read_bytes()).hexdigest()!=meta["开发文件SHA256"]:
        raise ValueError("冻结的归档或源开发清单发生变化")
    inputs={};groups=[]
    with ZipFile(archive) as z:
        members=split_members(z.infolist())
        for part in ("source","development"):
            row=[]
            for (uid,m,_),r in zip(members[part],records[part],strict=True):
                if uid!=r["id"] or uid in meta["封存编号"]:raise ValueError("源开发与封存身份冲突")
                raw=z.read(m)
                if sha256(raw).hexdigest()!=r["原件SHA256"]:raise ValueError("完整释放校验失败")
                row.append(read_windows(np.loadtxt(BytesIO(raw),delimiter=",")))
                if part=="source":groups.append(Path(m.filename).name.lower().split("_")[2:])
            inputs[part]=np.stack(row)
    return inputs,np.asarray(groups),records


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已交付当前协议")
        x,g,records=data()
        y=np.asarray([v["label"] for v in records["source"]])
        policy=PlumeInstallation().fit(x["source"],y,g)
        weight=DEST/"release.joblib";joblib.dump(policy,weight)
        previous=read(PRIOR/"scores.json");parent=previous["request"]
        old=read(PRIOR/"accepted"/(parent+".json"))["payload"]
        ids=[v["id"] for v in records["development"]]
        if digest(old)!=parent or ids!=old["ids"]:raise ValueError("交接测量范围不符")
        pred,prob=joblib.load(weight).predict(x["development"]);columns=dict(old["预测"])
        oldfixed=previous["decision"]["固定"];oldpaper=previous["decision"]["同工具论文"]
        for name,values in pred.items():columns[name]=values.tolist()
        new_controls=[k for k in pred if k!=PRIMARY]
        # 原固定与论文可使用同一组新工具作二分类并集或交集，不低估对手。
        for a in (oldfixed,PAPER):
            for b in new_controls:
                for n,op in (("至少一方报有",np.maximum),("双方都报有",np.minimum)):
                    columns[f"{a}合用/{b}/{n}"]=op(columns[a],pred[b]).tolist()
        payload={"ids":ids,"预测":columns,"各工具概率":{k:v.tolist() for k,v in prob.items()},
                 "上次回执":parent,"发布权重SHA256":sha256(weight.read_bytes()).hexdigest(),"装机选择":FIXED[policy.selected]}
        key=digest(payload);store=LocalAcceptance();store.output=DEST/"accepted"
        if not store.submit(key,payload)["accepted"]:raise ValueError("未确认交接")
        sent=read(store.output/(key+".json"))["payload"]
        if digest(sent)!=key:raise ValueError("回执内容改变")
        truth=np.array([v["label"] for v in records["development"]])
        scores={k:{"F1":float(f1_score(truth,v,average="macro")),"最差召回":float(recall_score(truth,v,average=None).min()),"正确":int((truth==v).sum())} for k,v in sent["预测"].items()}
        fixed=max([oldfixed,*new_controls,*[k for k in columns if k.startswith(oldfixed+"合用/")]],key=lambda k:scores[k]["F1"])
        paper=max([oldpaper,PAPER,*[k for k in columns if k.startswith(PAPER+"合用/")]],key=lambda k:scores[k]["F1"])
        gap=scores[PRIMARY]["F1"]-max(scores[k]["F1"] for k in (fixed,paper))
        d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":scores[PRIMARY]["F1"],"固定F1":scores[fixed]["F1"],"论文F1":scores[paper]["F1"],
           "差值":gap,"进独立":gap>=.01 and scores[PRIMARY]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"scores":scores,"decision":d,"request":key,"装机选择":FIXED[policy.selected],
                    "装机试验F1":dict(zip(FIXED,policy.quality.tolist(),strict=True))})
        print(json.dumps({"开发交付":d,"各训练读法":{k:scores[k] for k in [*FIXED,CV]}},ensure_ascii=False))


if __name__=="__main__":main()
