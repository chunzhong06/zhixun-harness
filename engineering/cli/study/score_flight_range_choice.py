"""相同源机队与预测成本的航班分段比较，独立第六组只读取冻结产物。"""
from argparse import ArgumentParser
import json

import h5py
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

from score_flight_load_neighbors import (DEST as FIRST, SITES, dataset_path, signals, read_flights, file_hash,
                                        read, deliver, score, infer_fleet, digest, LocalAcceptance, LocalStore, atomic_json)
from harness.models.flight_range_choice import FlightRangeChoice,range_views,train_heads,CHOICES
from harness.models.flight_load_neighbors import RULES

DEST=FIRST.parent/"flight-range-choice"


def evaluate_forward(cohort):
    scores = cohort["scores"]
    primary = scores["事前主方案"]
    paper_name = "Darrah双向序列本地适配"
    dynamic = {*CHOICES, *RULES, "旧按覆盖两森林", "现场按分歧",
               "系统实际交付", "分段实际交付", "事前主方案"}
    fixed = [key for key in scores if key not in dynamic and key != paper_name]
    best = min(fixed, key=lambda name: scores[name]["rmse"])
    fixed_gain = 1 - primary["rmse"] / scores[best]["rmse"]
    paper_gain = 1 - primary["rmse"] / scores[paper_name]["rmse"]
    worst_change = primary["worst"] / scores["整组选择"]["worst"] - 1
    return {"最强固定": best, "固定对照列数（含等价读法）": len(fixed),
            "主方案误差": primary["rmse"], "最强固定误差": scores[best]["rmse"],
            "论文适配误差": scores[paper_name]["rmse"], "上版误差": scores["整组选择"]["rmse"],
            "相对最强固定的误差降幅": fixed_gain, "相对论文适配的误差降幅": paper_gain,
            "上版最差台": scores["整组选择"]["worst"], "主方案最差台": primary["worst"],
            "最差台误差增幅": worst_change, "通过机队验收": bool(min(fixed_gain, paper_gain) >= .01 and worst_change <= .01)}


def matrices(flights,prep):
    records=[range_views(prep,flights[k][1]) for k in sorted(flights)]
    return [np.array(col) for col in zip(*records,strict=True)]


def train():
    previous=joblib.load(FIRST/"flight.joblib")
    prep=previous.preparation
    values,alternate,targets,units,sites=[],[],[],[],[]
    with h5py.File(dataset_path("DS02")) as data:
        rows=signals(data,"dev")
        early=[rows[k][1] for uid in sorted({u for u,_ in rows})[:3] for k in sorted(k for k in rows if k[0]==uid)[:8]]
        healthy=np.concatenate(early)
        scale=StandardScaler().fit(healthy[:,14:])
        expected=make_pipeline(PolynomialFeatures(2,include_bias=False),Ridge(alpha=1.)).fit(
            scale.transform(healthy[:,14:]),healthy[:,:14])
        alt_prep=(scale,expected,None,None)
    for name in SITES:
        with h5py.File(dataset_path(name)) as data:
            flights=signals(data,"dev")
            keys=sorted(flights)
            truths=read_flights(data,"dev",answers=True)
            if set(keys)!=set(truths):
                raise ValueError("训练标签与飞行错位")
            values.append(matrices(flights,prep))
            alternate.append(matrices(flights,alt_prep))
            targets.extend(truths[k] for k in keys)
            sites.extend([name]*len(keys))
            units.extend(u for u,_ in keys)
    x=[np.vstack(col) for col in zip(*values,strict=True)]
    alt=[np.vstack(col) for col in zip(*alternate,strict=True)]
    y,site,unit=np.array(targets),np.array(sites),np.array(units)
    predictions=np.empty((len(y),5))
    for name in SITES:
        held=site==name
        # DS01被留出时，误差估计所使用的健康参照也只能来自其它机队。
        bank=alt if name=="DS01" else x
        heads=train_heads([m[~held] for m in bank],y[~held])
        predictions[held]=np.column_stack([m.predict(v[held]) for m,v in zip(heads,bank[:5],strict=True)])
    return FlightRangeChoice(prep,x,y,predictions,x[5],site,unit)


def publish(name,model,receiver,original,blind):
    with h5py.File(dataset_path(name)) as data:
        flights=signals(data,"test")
        keys=sorted(flights)
        if original["ids"]!=[list(k) for k in keys]:
            raise ValueError("与先前真实提交的测量不一致")
        out=model.predictions(matrices(flights,model.prep))
        out={**original["predictions"],**{k:v.tolist() for k,v in out.items()}}
        if blind:
            primary=read(DEST/"selection.json")["primary"]
            controls=read(DEST/"fixed-controls.json")
            head_outputs=np.column_stack([out[k] for k in controls["顺序"]])
            for label,weights in controls["固定权重"].items():
                out[label]=(head_outputs@weights).tolist()
            aux=data["A_test"][:,:2].astype(int)
            bounds=np.r_[0,np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))+1,len(aux)]
            issued={}
            for lo,hi in zip(bounds[:-1],bounds[1:],strict=True):
                request={"inputs":{"sensors":data["X_s_test"][lo:hi],"settings":data["W_test"][lo:hi]}}
                value,=infer_fleet(model,[request])
                issued[tuple(map(int,aux[lo]))]=value["outputs"]["value"]
            out["分段实际交付"]=[issued[k] for k in keys]
            np.testing.assert_allclose(out["分段实际交付"],out[model.rule],rtol=0,atol=1e-9)
            out["事前主方案"]=out[primary]
        payload={"ids":original["ids"],"predictions":out}
        key=digest([name,payload])
        ack=receiver.submit(key,payload)
        if not ack["accepted"] or ack["request_id"]!=key:
            raise RuntimeError("分段预测未交接")
        return key


def main(blind):
    directory=DEST/"forward" if blind else DEST
    with LocalStore(directory/"lock").exclusive():
        if (directory/"scores.json").exists():
            raise ValueError("已有该组实测")
        receiver=LocalAcceptance()
        receiver.output=directory/"accepted"
        old_scores=read(FIRST/"scores.json")
        selected=read(FIRST/"training.json")
        if blind:
            frozen=read(DEST/"selection.json")
            if file_hash(DEST/"range.joblib")!=frozen["artifact"]:
                raise ValueError("分段估计产物身份改变")
            model=joblib.load(DEST/"range.joblib")
        else:
            model=train()
        pending,coverage={},{}
        for name in (("DS06",) if blind else SITES):
            if blind:
                known=read(FIRST/"overlap.json")
                controls=LocalAcceptance()
                controls.output=directory/"matched-controls"
                digest_old,coverage[name]=deliver(directory,name,joblib.load(FIRST/"flight.joblib"),
                    joblib.load(FIRST/"five-source.joblib"),True,controls,set(known["measurements"]),set(known["operating"]))
                source=controls.output
            else:
                digest_old=old_scores[name]["request"]
                source=FIRST/"accepted"
            original=read(source/(digest_old+".json"))["payload"]
            if digest([name,original])!=digest_old:
                raise ValueError("固定对照预测身份改变")
            pending[name]=publish(name,model,receiver,original,blind)
        result={}
        for name,key in pending.items():
            with h5py.File(dataset_path(name)) as data:
                result[name]={**score(name,receiver,key,data),**coverage.get(name,{})}
            if blind:
                verdict=evaluate_forward(result[name])
                result[name]["验收"]=verdict
                print(json.dumps({"独立机队验收":verdict},ensure_ascii=False),flush=True)
        if not blind:
            means={k:float(np.mean([a["scores"][k]["rmse"] for a in result.values()]))
                   for k in (*CHOICES,selected["rule"])}
            model.rule=min(CHOICES,key=means.get)
            primary="分段实际交付" if means[model.rule]<means[selected["rule"]] else "系统实际交付"
            joblib.dump(model,DEST/"range.joblib")
            atomic_json(DEST/"selection.json",{"primary":primary,"rule":model.rule,"development_means":means,
                                               "artifact":file_hash(DEST/"range.joblib")})
            print(json.dumps({"现场候选开发均值":means},ensure_ascii=False),flush=True)
        atomic_json(directory/"scores.json",result)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--blind",action="store_true")
    main(parser.parse_args().blind)
