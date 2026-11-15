"""在同一120瓶源标气中逐瓶请求真实目标波形，封存询问序列并交付79瓶服务预测。"""
import json
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score,recall_score

from score_year_gas_calibration import DATA,NAME,ROOT,GASES,measurements,GasBase,LocalAcceptance,LocalStore,atomic_json,digest
from score_mixture_response import CHEM
from verify_year_known_controls import DEST as PRIOR
from harness.models.calibration_query import RULES,ACTIVE,HASH,CalibrationQuery,query_checkpoints
from harness.models.year_install_readout import InstallReadout

DEST = PRIOR.parent/"calibration-query"
READOUT = "按气体与预算固定"
PAPER = "按气体与预算论文"


def chemistry(ids):
    raw = loadmat(DATA/NAME,variable_names=["target"],simplify_cells=True)["target"]
    return np.stack([[raw[n][np.flatnonzero(raw["states"]==uid)[0]] for n in CHEM] for uid in ids])


def run():
    ids,features,answer = measurements()
    source,*chips = sorted(features)
    sx = features[source][:120]
    chem = chemistry(ids[:120])
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    events = {}
    for gas in GASES:
        j = CHEM.index(gas)
        base = GasBase().fit(sx,chem[:,j])
        query = CalibrationQuery(sx,chem,j,base)
        for chip in chips:
            tx = features[chip]
            outputs = {20:{},80:{}}
            traces = {20:{},80:{}}
            for rule in RULES:
                # 源波形及配方公开。每个策略的目标端接口只释放它自己选中的一瓶波形。
                def buy(i):
                    if not 0<=i<120:raise ValueError("装机池以外的查询")
                    return tx[i].copy()
                for ix,observed in query_checkpoints(query,rule,(20,80),buy):
                    size = len(ix)
                    model = InstallReadout().fit(base,sx,chem,ix,observed,j,gas)
                    forecasts = model.predict(tx[120:])
                    if rule==HASH:
                        key = f"{chip}-{gas}-{size}"
                        request = previous["安装"][key]["回执"]
                        parent = json.loads((PRIOR/"accepted"/(request+".json")).read_bytes())["payload"]
                        if digest([key,parent])!=request or ids[120:].tolist()!=parent["ids"]:
                            raise ValueError("原开发交付或配方身份变化")
                        for name in (READOUT,PAPER):
                            if not np.allclose(forecasts[name],parent["浓度"][name],rtol=1e-10,atol=1e-8):
                                raise ValueError("原散列规则与已发布权重输出不符")
                    outputs[size][rule] = {k:v.tolist() for k,v in forecasts.items()}
                    traces[size][rule] = ids[ix].tolist()
            for size in (20,80):
                key = f"{chip}-{gas}-{size}"
                payload = {"配方":ids[120:].tolist(),"装机":traces[size],"气体":gas,"阈值":float(query.cut),"浓度":outputs[size]}
                request = digest(payload)
                if not receiver.submit(request,payload)["accepted"]:raise ValueError("服务未交付")
                events[key] = request
    scores = {}
    for key,request in events.items():
        p = json.loads((receiver.output/(request+".json")).read_bytes())["payload"]
        if digest(p)!=request or p["配方"]!=ids[120:].tolist():raise ValueError("预测身份异常")
        y = answer[p["气体"]][120:]
        yy = y>p["阈值"]
        scores[key] = {}
        for rule,cols in p["浓度"].items():
            samples = p["装机"][rule]
            if len(samples)!=len(set(samples)) or len(samples)!=int(key.split("-")[-1]) or set(samples)&set(p["配方"]):
                raise ValueError("标气预算或隔离异常")
            scores[key][rule] = {}
            for arm,v in cols.items():
                v = np.asarray(v)
                pred = v>p["阈值"]
                scores[key][rule][arm] = {"F1":float(f1_score(yy,pred,average="macro")),
                    "弱类召回":float(recall_score(yy,pred,average=None).min()),"浓度误差":float(np.mean((y-v)**2)**.5)}
    controls = {}
    for typ in ("固定","论文"):
        chosen = {}
        for gas in GASES:
            for size in (20,80):
                suffix = f"{gas}-{size}"
                keys = [k for k in scores if k.endswith(suffix)]
                choices = [(rule,arm) for rule in RULES for arm in scores[keys[0]][rule] if
                           (typ=="固定" and rule!=ACTIVE or typ=="论文" and arm in (PAPER,"全曲线回源","单用论文续训"))]
                best = max(choices,key=lambda pair:np.mean([scores[k][pair[0]][pair[1]]["F1"] for k in keys]))
                chosen[suffix] = list(best)
        controls[typ] = chosen
    by_case = {}
    for k,s in scores.items():
        suffix = k.split("-",1)[1]
        by_case[k] = {"现场":s[ACTIVE][READOUT]}
        for typ,policy in controls.items():
            rule,arm = policy[suffix]
            by_case[k][typ] = s[rule][arm]
    means = {t:{m:float(np.mean([s[t][m] for s in by_case.values()])) for m in ("F1","弱类召回","浓度误差")} for t in ("现场","固定","论文")}
    gap = means["现场"]["F1"]-max(means[t]["F1"] for t in ("固定","论文"))
    d = {"现场F1":means["现场"]["F1"],"固定F1":means["固定"]["F1"],"论文F1":means["论文"]["F1"],"差值":gap,
         "进入独立":gap>=.01 and means["现场"]["弱类召回"]>=means["固定"]["弱类召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"controls":controls,"scores":scores,"requests":events})
    print(json.dumps({"装机逐瓶询问":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("该数据与规则已有实测")
        run()
