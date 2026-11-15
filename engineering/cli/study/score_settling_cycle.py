"""新增训练证据是同一配方有效周期的先后，固定、现场和论文共享实际装机名单及读数。"""
from hashlib import file_digest,sha256
import json
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score,recall_score

from score_paired_design import DEST as PRIOR,GASES,LocalStore,LocalAcceptance,atomic_json,digest
from score_year_gas_calibration import DATA,NAME,HASH,feature_cycles
from harness.models.calibration_query import ACTIVE
from harness.models.paired_design import ADAPT,GEOMETRY
from harness.models.settling_cycle import VIEWS,PRIMARY,CycleTools,SettlingPolicy,summarize
from harness.models.year_gas_calibration import GasBase

DEST = PRIOR.parent/"settling-cycle"


def data():
    with (DATA/NAME).open("rb") as stream:
        if file_digest(stream,"md5").hexdigest()!=HASH:raise ValueError("第一年原包变化")
    raw = loadmat(DATA/NAME,variable_names=["sensor","target"],simplify_cells=True)
    t = raw["target"]
    valid = np.isfinite(t["states"])&np.isfinite(t["acetone"])&np.isfinite(t["toluene"])
    st = t["states"][valid].astype(int)
    ids = sorted(set(st),key=lambda i:sha256(f"year-gas-v1/{i}".encode()).digest())
    pos = [np.flatnonzero(st==i) for i in ids]
    sites = sorted({key.rsplit("_",1)[0] for key in raw["sensor"]})
    profiles = {}
    for chip in sites:
        x = np.stack([raw["sensor"][chip+f"_sensor{i}"][valid] for i in range(4)],axis=1)
        cycles = feature_cycles(x)
        profiles[chip] = summarize([cycles[idx] for idx in pos])
    y = {gas:np.array([t[gas][valid][i[0]] for i in pos]) for gas in GASES}
    return np.asarray(ids),profiles,y


def fit_bases(cache,gas,exclude,source,y):
    identity = (gas,tuple(sorted(exclude.tolist())))
    if identity not in cache:
        keep = np.ones(120,bool);keep[exclude] = False
        cache[identity] = {v:GasBase().fit(source[v][:120][keep],y[:120][keep]) for v in VIEWS}
    return cache[identity]


def infer(gas,ix,context,source,target,y,cache):
    folds = np.arange(len(ix))%4
    held = np.empty((len(ix),3*len(VIEWS)))
    cut = float(np.median(y[:120]))
    for j in range(4):
        test = folds==j
        bases = fit_bases(cache,gas,ix[test],source,y)
        model = CycleTools().fit(bases,{v:target[v][ix[~test]] for v in VIEWS},y[ix[~test]])
        _,held[test] = model.predict({v:target[v][ix[test]] for v in VIEWS})
    policy = SettlingPolicy().fit(held,context[ix],y[ix],cut)
    bases = fit_bases(cache,gas,np.array([],int),source,y)
    model = CycleTools().fit(bases,{v:target[v][ix] for v in VIEWS},y[ix])
    names,bank = model.predict({v:target[v][120:] for v in VIEWS})
    return model,policy,{**dict(zip(names,bank.T,strict=True)),**policy.predict(bank,context[120:])}


def run():
    ids,profiles,y = data()
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    idx = {uid:i for i,uid in enumerate(ids[:120])}
    src,*chips = sorted(profiles)
    source = profiles[src][0]
    receiver = LocalAcceptance();receiver.output = DEST/"accepted"
    events,cache = {},{}
    for key,prior_key in previous["requests"].items():
        q = json.loads((PRIOR/"accepted"/(prior_key+".json")).read_bytes())["payload"]
        if digest(q)!=prior_key or q["配方"]!=ids[120:].tolist():raise ValueError("原装机或服务记录变更")
        chip,gas,budget = key.split("-")
        target,ctx = profiles[chip]
        pub = previous["controls"]["固定"][gas+"-"+budget][0]
        paper_query = previous["controls"]["论文"][gas+"-"+budget][0]
        queries = set((pub,paper_query,ADAPT))
        readings = {r:dict(cols) for r,cols in q["浓度"].items()}
        for rule in sorted(queries):
            ix = np.array([idx[uid] for uid in q["装机"][rule]])
            if len(ix)!=int(budget) or len(set(ix))!=len(ix):raise ValueError("计费范围不同")
            model,policy,out = infer(gas,ix,ctx,source,target,y[gas],cache)
            if rule==pub:
                weight = DEST/(key+".joblib");joblib.dump((model,policy),weight,compress=3)
                loaded,decider = joblib.load(weight)
                _,values = loaded.predict({v:target[v][120:] for v in VIEWS})
                if not np.allclose(decider.predict(values,ctx[120:])[PRIMARY],out[PRIMARY]):raise ValueError("模型身份复核失败")
            readings[rule].update({k:v.tolist() for k,v in out.items()})
        payload = {"配方":q["配方"],"浓度":readings,"现场询问":pub,"上轮":prior_key,"阈值":q["阈值"],"权重":sha256(weight.read_bytes()).hexdigest()}
        req = digest(payload)
        if not receiver.submit(req,payload)["accepted"]:raise ValueError("未交接")
        events[key] = req
    scores = {}
    for key,req in events.items():
        p = json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
        if digest(p)!=req or p["配方"]!=ids[120:].tolist():raise ValueError("回执发生变化")
        gas = key.split("-")[1];yy = y[gas][120:]>p["阈值"]
        scores[key] = {}
        for rule,cols in p["浓度"].items():
            scores[key][rule]={}
            for arm,v in cols.items():
                labels = np.asarray(v)>p["阈值"]
                scores[key][rule][arm] = {"F1":float(f1_score(yy,labels,average="macro")),
                                          "弱类召回":float(recall_score(yy,labels,average=None).min())}
    controls = {"固定":{},"论文":{}}
    for suffix in previous["controls"]["固定"]:
        keys = [k for k in scores if k.endswith(suffix)]
        for t in controls:
            opts = [(r,a) for r,cols in scores[keys[0]].items() for a in cols if
                    (t=="论文" and ("论文" in a or a=="全曲线回源") or
                     t=="固定" and r not in (ADAPT,GEOMETRY,ACTIVE) and a!=PRIMARY)]
            controls[t][suffix] = max(opts,key=lambda v:np.mean([scores[k][v[0]][v[1]]["F1"] for k in keys]))
    parts = []
    for key,s in scores.items():
        suffix = key.split("-",1)[1]
        selected = previous["controls"]["固定"][suffix][0]
        row = {"现场":s[selected][PRIMARY]}
        for t,policy in controls.items():
            r,a = policy[suffix];row[t] = s[r][a]
        parts.append(row)
    means = {t:{m:float(np.mean([s[t][m] for s in parts])) for m in ("F1","弱类召回")} for t in parts[0]}
    gap = means["现场"]["F1"]-max(means[t]["F1"] for t in controls)
    d = {"现场F1":means["现场"]["F1"],"固定F1":means["固定"]["F1"],"论文F1":means["论文"]["F1"],
         "差值":gap,"进入独立":gap>=.01 and means["现场"]["弱类召回"]>=means["固定"]["弱类召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"controls":controls,"scores":scores,"requests":events})
    print(json.dumps({"单瓶重复周期变化":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同完整周期训练已经交付")
        run()
