"""依次向独立机队发出冻结的DS01工具预测，两组交接完成后才读取寿命答案。"""
from argparse import ArgumentParser
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
from catboost import CatBoostRegressor

from score_ncmapss_reference import ROOT, DEST as OLD, RAW, MEMBER_SHA, file_hash, row_groups, flights, measures
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

RAW2 = RAW.with_name("N-CMAPSS_DS02-006.h5")
DEST = OLD.parent/"ncmapss-ds02-forward"


def read(path):
    return json.loads(path.read_bytes())


def read_flights(raw,part,*,answers):
    aux = raw["A_"+part]
    sums,counts,labels = {},{},{}
    for start in range(0,len(aux),65536):
        chunk = slice(start,start+65536)
        keys = aux[chunk,:2]
        if not np.array_equal(keys,keys.astype(int)):
            raise ValueError("飞行或设备编号非整数")
        unique,inverse = np.unique(keys.astype(int),axis=0,return_inverse=True)
        if answers:
            values = raw["Y_"+part][chunk].ravel()
        else:
            values = np.column_stack((raw["X_s_"+part][chunk],raw["W_"+part][chunk]))
        if not np.isfinite(values).all():
            raise ValueError("实际飞行数据存在非有限值")
        for j,pair in enumerate(unique):
            key = tuple(map(int,pair))
            subset = values[inverse==j]
            if answers:
                if not np.all(subset==subset[0]) or labels.setdefault(key,float(subset[0]))!=subset[0]:
                    raise ValueError("同一飞行的真实终点答案不一致")
            else:
                sums[key] = sums.get(key,0)+subset.sum(0)
                counts[key] = counts.get(key,0)+len(subset)
    if answers:
        for uid in {k[0] for k in labels}:
            cycles = sorted(k[1] for k in labels if k[0]==uid)
            if cycles!=list(range(cycles[0],cycles[-1]+1)) or any(labels[uid,c]!=cycles[-1]-c for c in cycles):
                raise ValueError("设备未提供连续到失效的完整寿命")
        return labels
    return {k:sums[k]/v for k,v in counts.items()}


def main(subset="DS02"):
    raw_path, = RAW.parent.glob(f"N-CMAPSS_{subset}-*.h5")
    destination = OLD.parent/f"ncmapss-{subset.lower()}-forward"
    with LocalStore(destination/"lock").exclusive():
        meta = read(raw_path.with_suffix(".source.json"))
        if (destination/"scores.json").exists() or file_hash(raw_path)!=meta["sha256"] or file_hash(RAW)!=MEMBER_SHA:
            raise ValueError("数据身份变更或协议已有结果")
        selection = None
        if subset=="DS03":
            frozen = OLD.parent/"engine-source-tool-choice/source.json"
            decision = read(frozen)["flight"]
            selection = {"source":file_hash(frozen),"choice":decision["choice"],"source_scores":decision["score"]}
            if decision["choice"]!=min(decision["score"],key=decision["score"].get):
                raise ValueError("源端选择与原始留出误差不符")
        old = read(OLD/"scores.json")
        request = old["requests"]["service"]
        earlier = read(OLD/"accepted"/(request+".json"))["payload"]
        if digest(["service",earlier])!=request:
            raise ValueError("冻结的DS01预测回执不符")
        models = {}
        for key,identity in earlier["models"].items():
            p = OLD/("preparation.joblib" if key=="preparation" else key+".cbm")
            if file_hash(p)!=identity:
                raise ValueError("已发放模型发生变化")
            if key!="preparation":
                models[key] = CatBoostRegressor().load_model(str(p))
        op_scale,cluster,scales,global_scale,simple = joblib.load(OLD/"preparation.joblib")

        def predict(x):
            group = row_groups(x,op_scale,cluster)
            inputs = {"raw":x,"operating":x[:,14:],"global":global_scale.transform(x),
                      **{k:np.column_stack((s.transform(group)[:,:14],x[:,14:])) for k,s in scales.items()}}
            out = {k:m.predict(inputs[k],thread_count=1) for k,m in models.items()}
            out.update({k:m.predict(x) for k,m in simple.items()})
            out["half"] = (out["forest"]+out["ridge"])/2
            return out

        with h5py.File(RAW) as base,h5py.File(raw_path) as data:
            for key in ("X_s_var","W_var","A_var"):
                if not np.array_equal(base[key][...],data[key][...]):
                    raise ValueError("两套机队的公开字段不能直接对应")
            seen = set()
            if subset=="DS03":
                with h5py.File(RAW2) as second:
                    for part in ("dev","test"):
                        for row in read_flights(second,part,answers=False).values():
                            seen.add(np.asarray(row,dtype="<f8").tobytes())
            for part in ("dev","test"):
                original,_ = flights(base,part)
                for rows in original.values():
                    seen.update(np.asarray(r[1],dtype="<f8").tobytes() for r in rows)
                if part=="test":
                    by_id = {(u,c):x for u,seq in original.items() for c,x,_ in seq}
                    x = np.asarray([by_id[tuple(k)] for k in earlier["ids"]])
                    for k,p in predict(x).items():
                        np.testing.assert_allclose(p,earlier["predictions"][k],rtol=0,atol=1e-8)
            receiver = LocalAcceptance()
            receiver.output = destination/"accepted"
            requests = {}
            seen_units = set()
            for part in ("test","dev"):
                raw = read_flights(data,part,answers=False)
                ids = sorted(raw)
                units = {k[0] for k in ids}
                if subset=="DS02" and units!=({"test":{11,14,15},"dev":{2,5,10,16,18,20}}[part]):
                    raise ValueError("DS02设备与文献公布的拆分不符")
                if not units or seen_units & units:
                    raise ValueError("同一机队的两个作者拆分出现空集或设备交叉")
                seen_units.update(units)
                x = np.array([raw[k] for k in ids])
                fingerprints = {row.astype("<f8").tobytes() for row in x}
                if len(fingerprints)!=len(x) or seen.intersection(fingerprints):
                    raise ValueError("飞行均值记录存在完全重复")
                seen.update(fingerprints)
                payload = {"ids":ids,"features":digest(x.tolist()),"models":earlier["models"],
                           "predictions":{k:v.tolist() for k,v in predict(x).items()}}
                if selection:
                    payload["selection"] = selection
                    payload["predictions"]["selected"] = payload["predictions"][selection["choice"]]
                key = digest([part,payload])
                ack = receiver.submit(key,payload)
                if not ack["accepted"] or ack["request_id"]!=key:
                    raise RuntimeError("飞行预测未交接")
                requests[part] = key
            scores = {}
            for part,key in requests.items():
                targets = read_flights(data,part,answers=True)
                payload = read(receiver.output/(key+".json"))["payload"]
                keys = [tuple(i) for i in payload["ids"]]
                if set(keys)!=set(targets):
                    raise ValueError("答案与交付测点不同")
                y = np.array([targets[k] for k in keys])
                devices = np.array([k[0] for k in keys])
                scores[part] = {k:measures(y,np.asarray(v),devices) for k,v in payload["predictions"].items()}
                print(json.dumps({part:{k:{"误差":v["rmse"],"最差设备":v["worst"]} for k,v in scores[part].items()}},ensure_ascii=False),flush=True)
            result = {"requests":requests,"scores":scores,"raw_sha256":meta["sha256"],"source_run":request}
            if selection:
                result["selection"] = selection
            atomic_json(destination/"scores.json",result)


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--subset",choices=("DS02","DS03"),default="DS02")
    main(parser.parse_args().subset)
