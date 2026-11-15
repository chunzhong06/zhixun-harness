"""按独立采集批交接预测，在两个预先登记的反馈时钟下运行同一冻结策略。"""
import csv
import io
import json
import re
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from score_chongqing_feedback import fixed
from score_gas_probability_fusion import BASE, ROOT, LocalAcceptance, LocalStore, atomic_json, digest
from harness.models.site_training import SiteTraining

OUTPUT = BASE / "whole-maintenance/flow-transfer"
ARCHIVE = ROOT / "data/public/uci-flow-modulation/flow-modulation.zip"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-flow-transfer.json"
GASES = ("air", "acetone", "ethanol", "mixture")
DAYS = ("day-1-morning", "day-2-morning", "day-2-afternoon", "day-3-morning", "day-4-afternoon")


def read_source():
    if sha256(ARCHIVE.read_bytes()).hexdigest() != "7b062960dbef1a9e8aefc62bc7dbac09bdadb8cbb40793cf2b8e122da1862f90":
        raise ValueError("采集原件身份发生变化")
    with ZipFile(ARCHIVE) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("features.csv").decode())))
    features = [key for key in rows[0] if re.fullmatch(r"S\d+_(max|r\d+_A[lh]f)", key)]
    rows.sort(key=lambda row: int(row["exp"]))
    ids = [row["exp"] for row in rows]
    if len(set(ids)) != 58 or len(features) != 432:
        raise ValueError("采集身份或特征列不符")
    x = np.asarray([[float(row[col]) for col in features] for row in rows])
    days = [np.flatnonzero([row["batch"] == day for row in rows]) for day in DAYS]
    if [len(day) for day in days] != [19,10,10,11,8] or not np.isfinite(x).all():
        raise ValueError("真实批次或测量不符")
    def answers(indices):
        return np.array([1 + GASES.index(rows[i]["gas"]) for i in indices])
    return x, ids, days, answers


def blocks(days, cadence):
    for day, indices in zip(DAYS[1:], days[1:], strict=True):
        groups = [indices[j:j+4].tolist() for j in range(0,len(indices),4)]
        parts = [(g[:-1],g[-1:]) if len(g)>1 else (g,[]) for g in groups]
        if cadence == "batch":
            yield day, [i for service,_ in parts for i in service], [i for _,feedback in parts for i in feedback]
        else:
            for service,feedback in parts:
                yield day,service,feedback


def measure(y,p):
    matrix = confusion_matrix(y,p,labels=[1,2,3,4])
    counts = matrix.sum(1)
    recall = np.divide(matrix.diagonal(),counts,out=np.zeros(4),where=counts>0)
    return {"macro":float(f1_score(y,p,labels=[1,2,3,4],average="macro",zero_division=0)),
            "support":counts.tolist(),"recall":recall.tolist(),"worst":float(recall.min())}


def evaluate(cadence, x, ids, days, answers, bank):
    source = days[0]
    field = SiteTraining(x[source],answers(source),tuple(bank),4,"pooled-midway-cold")
    folder = OUTPUT / cadence
    endpoint = LocalAcceptance()
    endpoint.output = folder / "accepted"
    requests = []
    feedback_ids = []
    for day,service,feedback in blocks(days,cadence):
        votes = {k:v[service] for k,v in bank.items()}
        tools = field.predict(x[service],votes)
        fallback = tools["observer"]
        selected = field.current()
        models = field.models()
        snapshot = {"source_models_votes":{k:v.tolist() for k,v in votes.items()},
                    "models":models,"observed_counts":field.observer.counts.tolist()}
        identity = digest(snapshot)
        atomic_json(folder/"snapshots"/(identity+".json"),snapshot)
        preds = {"candidate":tools[selected],"observer":fallback,
                 "linear-start-midway":tools["pooled-source" if field.rounds<5 else "pooled-separate"]}
        for name in ("pooled-source","pooled-separate"):
            preds[name]=tools[name]
            preds[name+"-cold"]=tools[name] if field.history else fallback
        preds.update({"source-"+k:v for k,v in votes.items()})
        expected = preds["linear-start-midway"] if field.history else fallback
        np.testing.assert_array_equal(preds["candidate"],expected)
        payload = {"day":day,"indices":service,"ids":[ids[i] for i in service],"selected":selected,
                   "feedback_before":len(feedback_ids),"feedback_rounds_before":field.rounds,
                   "snapshot":identity,"predictions":{k:v.tolist() for k,v in preds.items()}}
        request = digest([digest(json.loads(PROTOCOL.read_bytes())),cadence,payload])
        ack = endpoint.submit(request,payload)
        if not ack["accepted"] or ack["request_id"] != request:
            raise RuntimeError("服务交接未确认")
        requests.append(request)
        if feedback:
            feedback_votes = {k:v[feedback] for k,v in bank.items()}
            before = field.predict(x[feedback],feedback_votes)
            labels = answers(feedback)
            field.update(x[feedback],feedback_votes,labels.tolist(),before)
            feedback_ids.extend(feedback)
            atomic_json(folder/"feedback"/(request+".json"),{"ids":[ids[i] for i in feedback],"labels":labels.tolist()})
    logs = [json.loads((endpoint.output/(r+".json")).read_bytes())["payload"] for r in requests]
    indices = [i for row in logs for i in row["indices"]]
    if len(set(indices+feedback_ids))!=39 or len(indices)+len(feedback_ids)!=39:
        raise ValueError("反馈与服务划分有重复或缺失")
    y = answers(indices)
    prediction = {k:np.concatenate([row["predictions"][k] for row in logs]) for k in logs[0]["predictions"]}
    scores = {k:measure(y,v) for k,v in prediction.items()}
    paired = {k:{"rescued":int(np.sum((prediction["candidate"]==y)&(p!=y))),
                 "harmed":int(np.sum((prediction["candidate"]!=y)&(p==y)))} for k,p in prediction.items()}
    per_day = {}
    for day in DAYS[1:]:
        mask = np.array([row["day"]==day for row in logs for _ in row["indices"]])
        per_day[day]={k:{"correct":int(np.sum(p[mask]==y[mask])),"rows":int(mask.sum())} for k,p in prediction.items()}
    result = {"service_ids":[ids[i] for i in indices],"feedback_ids":[ids[i] for i in feedback_ids],
              "requests":requests,"scores":scores,"paired":paired,"by_day":per_day}
    atomic_json(folder/"scores.json",result)
    return result


def main():
    with LocalStore(OUTPUT/"lock").exclusive():
        if (OUTPUT/"scores.json").exists():
            raise ValueError("请复用已完成的新来源评价")
        x,ids,days,answers = read_source()
        bank = fixed(x[days[0]],answers(days[0]),x)
        records = {key:evaluate(key,x,ids,days,answers,bank) for key in ("batch","early")}
        if any(records["batch"][key]!=records["early"][key] for key in ("service_ids","feedback_ids")):
            raise ValueError("时钟对照使用了不同测量")
        atomic_json(OUTPUT/"scores.json",{"input_sha256":sha256(ARCHIVE.read_bytes()).hexdigest(),
            "protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"classes":GASES,
            "scores":{k:v["scores"] for k,v in records.items()},"scope":"新仪器四类维护方法迁移"})
        for name,record in records.items():
            print(json.dumps({name:{k:{"分类分":v["macro"],"最弱类召回":v["worst"]}
                                    for k,v in record["scores"].items()}},ensure_ascii=False))


if __name__=="__main__":
    main()
