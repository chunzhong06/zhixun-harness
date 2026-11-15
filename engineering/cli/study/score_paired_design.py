"""检查现场预测未测标气波形所产生的选瓶与发布增量，用同一训练库重跑各固定顺序。"""
import json
from hashlib import sha256
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score

from score_calibration_query import DEST as PRIOR, GASES, measurements, chemistry, GasBase, LocalStore,LocalAcceptance,atomic_json,digest
from harness.models.calibration_query import query_checkpoints,ACTIVE
from harness.models.paired_design import RULES,ADAPT,CHOICE,PairedDesign,QueryDeployment

DEST = PRIOR.parent/"paired-design"


def run():
    ids,x,truth = measurements()
    src,*chips = sorted(x)
    source = x[src][:120]
    c = chemistry(ids[:120])
    parent = json.loads((PRIOR/"scores.json").read_bytes())
    events = {}
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    lookup = {int(uid):j for j,uid in enumerate(ids[:120])}
    for gas in GASES:
        from score_calibration_query import CHEM
        j = CHEM.index(gas)
        base = GasBase().fit(source,c[:,j])
        query = PairedDesign(source,c,j,base)
        for chip in chips:
            readings = x[chip]
            def buy(index):
                if not 0<=index<120:raise ValueError("配方池以外的请求")
                return readings[index].copy()
            histories, outputs, parents = {},{},{}
            for size in (20,80):
                key = f"{chip}-{gas}-{size}"
                request = parent["requests"][key]
                old = json.loads((PRIOR/"accepted"/(request+".json")).read_bytes())["payload"]
                if digest(old)!=request or old["配方"]!=ids[120:].tolist():
                    raise ValueError("原始服务范围或询问记录变化")
                histories[size] = dict(old["装机"])
                outputs[size] = {r:dict(v) for r,v in old["浓度"].items()}
                parents[size] = request
            for rule in [*histories[20],*RULES]:
                if rule in RULES:
                    observations = query_checkpoints(query,rule,(20,80),buy)
                else:
                    def replay():
                        for size in (20,80):
                            ix = np.array([lookup[uid] for uid in histories[size][rule]])
                            yield ix,np.stack([buy(i) for i in ix])
                    observations = replay()
                for ix,measured in observations:
                    size = len(ix)
                    installed = QueryDeployment().fit(source,c,ix,measured,j)
                    if rule==ADAPT:
                        path = DEST/f"{chip}-{gas}-{size}.joblib"
                        joblib.dump(installed,path,compress=3)
                        installed = joblib.load(path)
                    vals = installed.predict(readings[120:])
                    histories[size][rule] = ids[ix].tolist()
                    outputs[size].setdefault(rule,{}).update({name:v.tolist() for name,v in vals.items()})
                    if rule in RULES:
                        from harness.models.year_install_readout import InstallReadout
                        baseline = InstallReadout().fit(base,source,c,ix,measured,j,gas)
                        outputs[size][rule].update({name:v.tolist() for name,v in baseline.predict(readings[120:]).items()})
            for size in (20,80):
                key = f"{chip}-{gas}-{size}"
                payload = {"配方":ids[120:].tolist(),"装机":histories[size],"气体":gas,"阈值":float(query.cut),"浓度":outputs[size],"上轮":parents[size],
                           "现场权重":sha256((DEST/(key+".joblib")).read_bytes()).hexdigest()}
                request = digest(payload)
                if not receiver.submit(request,payload)["accepted"]:raise ValueError("新询问没有完成交付")
                events[key] = request
    scores = {}
    for key,req in events.items():
        record = json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
        if digest(record)!=req or record["配方"]!=ids[120:].tolist():raise ValueError("交付记录变更")
        y = truth[record["气体"]][120:]
        yy = y>record["阈值"]
        scores[key] = {}
        for rule,tools in record["浓度"].items():
            chosen = record["装机"][rule]
            if (len(chosen)!=int(key.split("-")[-1]) or len(set(chosen))!=len(chosen)
                    or not set(chosen)<=lookup.keys()):raise ValueError("询问范围或预算不符")
            scores[key][rule] = {}
            for name,v in tools.items():
                pred = np.array(v)>record["阈值"]
                scores[key][rule][name] = {"F1":float(f1_score(yy,pred,average="macro")),
                                          "弱类召回":float(recall_score(yy,pred,average=None).min())}
    controls = {"固定":{},"论文":{}}
    from harness.models.paired_design import GEOMETRY
    for gas in GASES:
        for size in (20,80):
            suffix = f"{gas}-{size}"
            keys = [k for k in scores if k.endswith(suffix)]
            for typ in controls:
                options = [(r,n) for r,tools in scores[keys[0]].items() for n in tools
                           if (typ=="固定" and r not in (ACTIVE,ADAPT,GEOMETRY) or typ=="论文" and ("论文" in n or n=="全曲线回源"))]
                controls[typ][suffix] = max(options,key=lambda pair:np.mean([scores[k][pair[0]][pair[1]]["F1"] for k in keys]))
    parts = []
    for k,v in scores.items():
        result = {"现场":v[ADAPT][CHOICE]}
        for typ,policy in controls.items():
            rule,name = policy[k.split("-",1)[1]]
            result[typ] = v[rule][name]
        parts.append(result)
    means = {typ:{m:float(np.mean([r[typ][m] for r in parts])) for m in ("F1","弱类召回")} for typ in parts[0]}
    gap = means["现场"]["F1"]-max(means[k]["F1"] for k in controls)
    d = {"现场F1":means["现场"]["F1"],"固定F1":means["固定"]["F1"],"论文F1":means["论文"]["F1"],
         "差值":gap,"进入独立":gap>=.01 and means["现场"]["弱类召回"]>=means["固定"]["弱类召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"controls":controls,"scores":scores,"requests":events})
    print(json.dumps({"芯片反馈选瓶与训练":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同选瓶已经实测")
        run()
