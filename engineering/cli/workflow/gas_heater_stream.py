"""从原始测量逐日按加热阶段发布浓度类别；每批预测接受后再取限定答案。"""
from argparse import ArgumentParser
from hashlib import file_digest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"engineering/src"),str(ROOT)]
from engineering.cli.workflow.gas_residual_stream import checked, LocalAcceptance, LocalStore, atomic_json, digest, gas_query_stream
from harness.models import gas_heater_archive, gas_heater_response
from harness.methods import gas_audit_feedback as audit

FIXED = "相位电阻合训"
LEARN = "用现场F1选择加热模型"
LEGACY_RAW_SHA = "cb5ce4a6af1a51b933d1979952d7845f0e9baac54e15a81a1e3599e8b85905d4"
LAYOUTS = {"full":(slice(None),slice(None)), "replacement":(slice(3),slice(3,6))}
QUERIES = ("均匀覆盖","随机","混合抽查","BADGE")


def batch_values(inner, rows):
    ids = [r["id"] for r in rows]
    filenames = sorted({k.rsplit("/",1)[0]+".csv" for k in ids})
    bank = {}
    for name in filenames:
        bank.update(gas_heater_archive.day_exposures(inner,name))
    if set(bank)!=set(ids) or len(ids)!=len(set(ids)):
        raise ValueError("一个采集日的实际暴露与登记批次不一致")
    return np.stack([bank[k] for k in ids])


def choose(confusion):
    if not confusion:
        return FIXED
    base = confusion[FIXED]
    def score(k):
        cm = confusion[k] + 16*base/base.sum()
        denom = cm.sum(0)+cm.sum(1)
        return np.mean(np.divide(2*cm.diagonal(),denom,out=np.zeros(3),where=denom>0))
    best = max(confusion,key=score)
    return best if score(best)>score(FIXED)+.01 else FIXED


def run(archive, manifest, out, query="混合抽查", view="均值", seed=17, budget=8, through=None):
    spec = json.loads(manifest.read_bytes())
    root = manifest.parent
    if root.name not in LAYOUTS or query not in QUERIES or view not in ("均值","加热"):
        raise ValueError("未识别的安装或查询读法")
    source = checked(root/spec["source"],spec["source_sha256"])
    with archive.open("rb") as raw:
        if file_digest(raw,"sha256").hexdigest()!=LEGACY_RAW_SHA:
            raise ValueError("采集归档身份改变")
    stop = len(spec["batches"]) if through is None else through
    if not 0<=stop<=len(spec["batches"]) or budget<0:
        raise ValueError("运行范围或反馈预算无效")
    modules = (gas_heater_archive,gas_heater_response,audit,gas_query_stream)
    identity = digest([spec,LEGACY_RAW_SHA,root.name,query,view,seed,budget,Path(__file__).read_text(encoding="utf8"),
                       *[Path(m.__file__).read_text(encoding="utf8") for m in modules]])
    store = LocalStore(out/"checkpoint")
    receiver = LocalAcceptance()
    receiver.output = out/"accepted"
    sc,tc = LAYOUTS[root.name]
    with store.exclusive(), ZipFile(archive) as z, ZipFile(BytesIO(z.read("gas-sensor-array-temperature-modulation.zip"))) as inner:
        state = store.load() or {"identity":identity,"events":[]}
        if state["identity"]!=identity or len(state["events"])>stop:
            raise ValueError("输入身份或断点已改变")
        src = gas_heater_response.read_views(batch_values(inner,source),sc)
        np.testing.assert_allclose(src["各路平均"],[r["values"] for r in source],rtol=1e-8,atol=1e-8)
        sy = np.asarray([r["label"] for r in source])
        source_view = src["各路平均" if view=="均值" else "相位电阻"]
        query_source = [{"values":row.tolist(),"label":int(y)} for row,y in zip(source_view,sy,strict=True)]
        history,known_y,qhistory,observed = [],[],[],[]
        known_ids = {r["id"] for r in source}
        confusion = {}
        for i,b in enumerate(spec["batches"][:stop]):
            rows = checked(root/b["inputs"],b["inputs_sha256"])
            ids = [r["id"] for r in rows]
            if known_ids.intersection(ids) or any(set(r)!={"id","values"} for r in rows):
                raise ValueError("本批含已提交测量或额外字段")
            known_ids.update(ids)
            x = batch_values(inner,rows)
            observed.append(x)
            current = gas_heater_response.read_views(x,tc)
            np.testing.assert_allclose(current["各路平均"],[r["values"] for r in rows],rtol=1e-8,atol=1e-8)
            available = gas_heater_response.read_views(np.concatenate(observed),tc)
            returned = gas_heater_response.read_views(np.concatenate(history),tc) if history else {k:[] for k in current}
            pred = gas_heater_response.train_votes(src,sy,available,returned,known_y,current)
            adoption = choose(confusion)
            payload = {"ids":ids,"预测":{k:v.tolist() for k,v in pred.items()},"采用":adoption,
                       "prediction":pred[FIXED].tolist(),"现场选择预测":pred[adoption].tolist(),"已见反馈":len(known_y)}
            key = digest([identity,payload])
            replay = i<len(state["events"])
            if replay:
                existing = json.loads((receiver.output/(key+".json")).read_bytes())
                if existing["payload"]!=payload or state["events"][i]["request"]!=key:
                    raise ValueError("断点前的发放无法复现")
            elif not receiver.submit(key,payload)["accepted"]:
                raise ValueError("发放未完成")
            selected, inverse = [],np.empty(0)
            y = np.empty(0,dtype=int)
            if i+1<len(spec["batches"]) and budget:
                qx = current["各路平均" if view=="均值" else "相位电阻"]
                scale,model,_ = gas_query_stream.train(query_source,qhistory,3)
                zq = scale.transform(qx)
                selected,inverse = audit.sample(query,ids,qx,zq,model.predict_proba(zq),budget,seed+i)
                if len(set(selected))!=min(budget,len(ids)):
                    raise ValueError("查询重复或数量不符")
                answers = checked(root/b["feedback"],b["feedback_sha256"])
                y = np.array([answers[ids[j]] for j in selected])
                known_y.extend(y.tolist())
                history.append(x[selected])
                qhistory.extend({"x":qx[j].tolist(),"y":int(t)} for j,t in zip(selected,y,strict=True))
                for k,v in pred.items():
                    cm = confusion.setdefault(k,np.zeros((3,3)))
                    np.add.at(cm,(y-1,v[selected]-1),inverse)
            event = {"request":key,"selected":[ids[j] for j in selected],"labels":y.tolist()}
            if replay and event!=state["events"][i]:
                raise ValueError("恢复后实际查询或答案改变")
            if not replay:
                state["events"].append(event)
                store.save(state)
        report = {"identity":identity,"events":state["events"],"returned":len(known_y)}
        atomic_json(out/"run.json",report)
        return report


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--archive",required=True,type=Path)
    cli.add_argument("--stream",required=True,type=Path)
    cli.add_argument("--output",required=True,type=Path)
    cli.add_argument("--query",choices=QUERIES,default="混合抽查")
    cli.add_argument("--query-view",choices=("均值","加热"),default="均值")
    cli.add_argument("--seed",type=int,default=17)
    cli.add_argument("--budget",type=int,default=8)
    cli.add_argument("--through",type=int)
    args=cli.parse_args()
    run(args.archive,args.stream,args.output,args.query,args.query_view,args.seed,args.budget,args.through)
