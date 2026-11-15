"""从逐轮实际交接核验当批近邻，并把冻结的 PHM08 选择移到新测点。"""
from hashlib import sha256

import numpy as np

from compare_current_neighbors import DEST
from compare_error_slice import BASE, RAW, SUBSETS, read_cmapss, test_answers
from compare_group_objectives import atomic_json, digest, metric, official, read, remaining_cycles, summary
from harness.data.cmapss_input import condition_key
from harness.models.feedback_neighbors import CurrentNeighbors
from harness.models.device_neighbors import nearest
from score_local_defaults import cost


def public_receipts(cohort, pkg, checksum, fit_ids):
    folder = DEST / "live" / cohort
    run = read(folder / "run.json")
    state = read(folder / "regular" / "state.json")
    prior = read(DEST / cohort / "dynamic-public" / "issued.json")["predictions"]
    if run["completed_rounds"] != 10 or run["policy"] != "weight" or run["start"] != 1:
        raise ValueError("入口尚未完成预定的十轮")
    service, base, groups, weights = {}, {}, {}, []
    for record in run["accepted"]:
        receipt = read(folder / "accepted" / (record["request"] + ".json"))
        out = receipt["payload"]
        values = out["predictions"]
        selected = state["completed"][record["batch"]]
        if (receipt["request_id"] != digest([run["identity"], out]) or receipt["request_id"] != record["request"]
                or digest(values) != record["outputs_sha256"] or out["selected"] != selected["model_id"]
                or out["input_digest"] != selected["input_digest"] or values.keys() != selected["predictions"].keys()
                or values.keys() & service.keys() or out["decision"]["weight"] != record["weight"]):
            raise ValueError("服务回执与当时交接不一致")
        service.update(values)
        base.update(selected["predictions"])
        groups.update(dict.fromkeys(values, len(weights)))
        weights.append(record["weight"])
    if base.keys() != prior["current"].keys() or max(abs(v-prior["current"][i]) for i,v in base.items()) > 1e-7:
        raise ValueError("普通维护轨迹与现行默认不同")
    arms = {k:prior[k] for k in ("fixed","current","mature","simple") if k in prior}
    arms.update({"from-five":prior["weight"], "from-one":service})
    # 静态参照使用完全相同的逐轮基础预测，不重新训练，也不读反馈或服务答案。
    lookup = CurrentNeighbors(None,checksum,[r for r in pkg.train if r.sample_id in fit_ids])
    keys = list(base)
    x = np.asarray([state["samples"][i]["values"] for i in keys])
    query = np.clip((x[:,:21]-lookup.mean)/lookup.scale,-10,10)
    _, reference = nearest(lookup.bank,lookup.y,lookup.units,query,np.zeros(len(lookup.y)),np.zeros(len(query)))
    arms["always-from-one"] = {i:float(.5*(base[i]+v)) for i,v in zip(keys,reference,strict=True)}
    if any(abs(service[i]-(base[i]+weights[groups[i]]*(v-base[i])))>1e-8 for i,v in zip(keys,reference,strict=True)):
        raise ValueError("冻结的普通权重与固定参照未复现实际输出")
    sealed = folder/"fixed-half.json"
    atomic_json(sealed,{"regular":digest(base),"bank":lookup.identity,"predictions":arms["always-from-one"]})
    arms["always-from-one"] = read(sealed)["predictions"]
    truth = {r.sample_id:r.label for r in pkg.validation} if cohort == "engine" else remaining_cycles()
    devices = {i:state["samples"][i]["unit_id"] for i in service}
    table = summary("engine",truth,{"parent":base, **arms},devices)
    for arm, values in arms.items():
        table[arm]["rounds"] = [metric("engine",truth,{i:v for i,v in values.items() if groups[i]==r}) for r in range(10)]
    fixed_entry = DEST/"live"/"fixed-entry"/cohort
    fixed_run = read(fixed_entry/"run.json")
    forecasts = {}
    if fixed_run["completed_rounds"]!=10 or fixed_run["policy"]!="always" or fixed_run["start"]!=1:
        raise ValueError("固定策略入口未完成")
    for ref,control in zip(fixed_run["accepted"],run["accepted"],strict=True):
        receipt = read(fixed_entry/"accepted"/(ref["request"]+".json"))
        payload = receipt["payload"]
        original = read(folder/"accepted"/(control["request"]+".json"))["payload"]
        if (receipt["request_id"]!=digest([fixed_run["identity"],payload]) or ref["request"]!=receipt["request_id"]
                or payload["input_digest"]!=original["input_digest"] or payload["bank"]!=original["bank"]
                or payload["selected"]!=original["selected"] or ref["weight"]!=.5
                or digest(payload["predictions"])!=ref["outputs_sha256"] or forecasts.keys() & payload["predictions"].keys()):
            raise ValueError("固定策略的当轮服务身份不一致")
        forecasts.update(payload["predictions"])
    fixed_predictions = arms["always-from-one"]
    if forecasts.keys()!=fixed_predictions.keys() or max(abs(v-fixed_predictions[i]) for i,v in forecasts.items())>1e-8:
        raise ValueError("固定策略实交与独立复算不同")
    atomic_json(folder / "scores.json", {"run_sha256":sha256((folder/"run.json").read_bytes()).hexdigest(),
                "fixed_half_sha256":sha256(sealed.read_bytes()).hexdigest(),
                "fixed_entry_sha256":sha256((fixed_entry/"run.json").read_bytes()).hexdigest(),
                "fixed_entry_seconds":sum(r["reference_seconds"] for r in fixed_run["accepted"]),
                "fixed_entry_cost":cost(fixed_entry/"regular",checksum,fit_ids),
                "arms":table,"weights":weights,"cost":cost(folder/"regular",checksum,fit_ids),
                "reference_seconds":sum(r["reference_seconds"] for r in run["accepted"])})
    print(cohort,{k:round(table[k]["score"],3) for k in arms},flush=True)
    return weights


def transfer(pkg, checksum, fit_ids, early_weights):
    streams = BASE / "live-first"
    fresh = read(streams/"cross-stream-offset16"/"issued.json")["predictions"]
    terminal = read(streams/"transfer"/"engine"/"issued.json")["predictions"]
    gates = read(DEST/"phm08"/"dynamic-public"/"issued.json")["history"]
    lookup = CurrentNeighbors(None,checksum,[r for r in pkg.train if r.sample_id in fit_ids])
    tracks = {s:read_cmapss(RAW,subset=s,partition="test") for s in SUBSETS}
    info, inputs = {}, []
    for subset, records in tracks.items():
        for unit in records:
            for r in range(11):
                cycle = 16+20*r if r<10 else len(unit.values)
                if r<10 and (subset=="FD001" or cycle>=len(unit.values)):
                    continue
                key = f"{subset}/{unit.unit}/{cycle}" if r<10 else unit.unit_id
                row = unit.values[cycle-1]
                inputs.append((*row[3:],*row[:3]))
                info[key] = (unit.unit_id,subset,min(r,9),len(unit.values)-cycle,condition_key(row[:3]))
    if len(info)!=len(inputs) or info.keys() != fresh["fixed"].keys() | terminal["fixed"].keys():
        raise ValueError("测点与本次冻结清单不符")
    x = np.asarray(inputs)
    query = np.clip((x[:,:21]-lookup.mean)/lookup.scale,-10,10)
    _, reference = nearest(lookup.bank,lookup.y,lookup.units,query,np.zeros(len(lookup.y)),np.zeros(len(query)))
    arms = {name:{**fresh[k],**terminal[k]} for name,k in
            (("current","error-first"),("fixed","fixed"),("simple","simple"),("mature","mature"))}
    # 目标服务窗没有反馈。下面三臂显式使用源窗规则；target-local 始终回退到当轮默认。
    arms["target-local"] = arms["current"]
    variants = ("always-from-five","source-weight-from-five","source-condition-from-five","source-weight-from-one")
    arms.update({k:{} for k in variants})
    for (key,(_,_,r,_,condition)), ref in zip(info.items(),reference,strict=True):
        gate = gates[r-4]["gate"] if r>=4 else {}
        factors = (.5 if gate else 0.,gate.get("weight",0.),
                   .5 if condition in gate.get("by-condition",()) else 0.,early_weights[r])
        base = arms["current"][key]
        for arm,w in zip(variants,factors,strict=True):
            arms[arm][key] = float(base+w*(ref-base))
    folder = DEST/"live"/"transfer"
    issued = folder/"issued.json"
    atomic_json(issued,{"inputs":digest([list(info),inputs]),"bank":lookup.identity,"early_weights":early_weights,
                        "source_policy":sha256((DEST/"phm08"/"dynamic-public"/"issued.json").read_bytes()).hexdigest(),
                        "predictions":arms})
    arms = read(issued)["predictions"]
    end = {i:y for subset, records in tracks.items() for i,y in test_answers(RAW,subset,records).items()}
    truth = {k:end[uid]+gap for k,(uid,_,_,gap,_) in info.items()}
    scores = {}
    for suite, keys in (("fresh",fresh["fixed"]),("terminal",terminal["fixed"])):
        selection = {arm:{k:pred[k] for k in keys} for arm,pred in arms.items()}
        table = summary("engine",truth,{"parent":selection["current"],**selection},{k:info[k][0] for k in keys})
        for arm,pred in selection.items():
            table[arm]["subsets"] = {s:metric("engine",truth,{k:v for k,v in pred.items() if info[k][1]==s})
                                     for s in sorted({info[k][1] for k in keys})}
        scores[suite] = table
        print(suite,len(keys),{k:round(table[k]["score"],3) for k in selection},flush=True)
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(issued.read_bytes()).hexdigest(),"suites":scores})


if __name__ == "__main__":
    package, checksum, batches = official("engine")
    fit_ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    public_receipts("engine",package,checksum,fit_ids)
    weights = public_receipts("phm08",package,checksum,fit_ids)
    transfer(package,checksum,fit_ids,weights)
