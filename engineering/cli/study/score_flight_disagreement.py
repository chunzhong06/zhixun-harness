"""各工具用完全相同的发布产物，按飞行分歧调权与始终用同一组权重作直接对照。"""
from argparse import ArgumentParser
import json

import h5py
import joblib
import numpy as np

from score_engine_fleet_training import DEST as PRIOR, SEEN, source_rows, dataset_path, arrays, signature
from score_engine_flight_load import DEST as FIRST, RAW, signals, old_arms, features, score
from score_ncmapss_field import file_hash
from harness.methods.fleet_training import fleet_trials
from harness.methods.flight_disagreement import fit_disagreement, combine
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from engineering.cli.workflow.flight_input_predict import infer_fleet

DEST = FIRST.parent / "flight-disagreement"


def train():
    prep, history, fold_features, _, _ = source_rows()
    selected, new, trace, held = fleet_trials(history, fold_features, collect_predictions=True)
    prior = json.loads((PRIOR / "training.json").read_bytes())
    if trace != prior["cross_validation"] or selected != prior["selected"]:
        raise ValueError("本次训练的留组结果与同数据先导结果不同")
    order = list(new)
    p = np.vstack([np.column_stack([held[n][k] for k in order]) for n in SEEN])
    labels = np.concatenate([history[n][1] for n in SEEN])
    units = np.concatenate([history[n][2] for n in SEEN])
    fleets = np.concatenate([[n] * len(history[n][1]) for n in SEEN])
    params = fit_disagreement(p,labels,fleets,units)
    joblib.dump({"tools":new,"weights":params,"preparation":prep},DEST/"mixture.joblib")
    atomic_json(DEST/"training.json",{"order":order,"weights":params,"artifact":file_hash(DEST/"mixture.joblib"),
                                      "prior":file_hash(PRIOR/"fleet.joblib"),"selected":selected})


def run(blind):
    directory = DEST/"forward" if blind else DEST
    with LocalStore(directory/"lock").exclusive():
        if (directory/"scores.json").exists():
            raise ValueError("本次飞行组合已有评分")
        if not blind:
            train()
        snapshot = json.loads((DEST/"training.json").read_bytes())
        if snapshot["artifact"] != file_hash(DEST/"mixture.joblib") or snapshot["prior"] != file_hash(PRIOR/"fleet.joblib"):
            raise ValueError("独立测量发放前模型版本改变")
        bundle = joblib.load(DEST/"mixture.joblib")
        old = joblib.load(FIRST/"tools.joblib")
        state = json.loads((PRIOR/"training.json").read_bytes())
        known = set(state["measured_flights"])
        weather = set(state["operating_profiles"])
        receiver = LocalAcceptance()
        receiver.output = directory/"accepted"
        pending, metadata = {}, {}
        for name in (("DS05",) if blind else SEEN):
            path = dataset_path(name)
            with h5py.File(path) as data, h5py.File(RAW) as src:
                if any(not np.array_equal(src[k][...],data[k][...]) for k in ("W_var","X_s_var","A_var")):
                    raise ValueError("飞行字段不同")
                flights = signals(data,"test")
                keys = sorted(flights)
                overlap = sum(signature(r[1]) in known for r in flights.values())
                if blind and overlap:
                    raise ValueError("新机队与已见测量重复")
                x = arrays(flights,bundle["preparation"])
                out = old_arms(x["原始均值"])
                a,b,c = features(flights,old[0])
                f,s = old[1][0].predict(a),old[2][0].predict(b)
                out.update({"旧全程森林":f,"旧可比工况森林":s,"旧按覆盖两森林":c*f+(1-c)*s})
                candidates = {k:model.predict(x[k.split("/")[0]]) for k,model in bundle["tools"].items()}
                out.update(candidates)
                matrix = np.column_stack([candidates[k] for k in snapshot["order"]])
                weights = snapshot["weights"]
                out.update({"全局固定组合":matrix@weights["global"],"等权组合":matrix.mean(axis=1),
                            "整组选择":candidates[snapshot["selected"]]})
                for i,w in enumerate(weights["local"],1):
                    out[f"固定分歧组{i}"] = matrix@w
                aux = data["A_test"][:,:2].astype(int)
                bounds = np.r_[0,np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))+1,len(aux)]
                direct = {}
                for start,end in zip(bounds[:-1],bounds[1:],strict=True):
                    request = {"inputs":{"sensors":data["X_s_test"][start:end],"settings":data["W_test"][start:end]}}
                    result, = infer_fleet(bundle,[request])
                    direct[tuple(map(int,aux[start]))] = result["outputs"]["value"]
                pred = np.array([direct[k] for k in keys])
                np.testing.assert_allclose(pred,combine(matrix,weights),atol=1e-8,rtol=0)
                out["现场按分歧"] = pred
                payload = {"ids":[list(k) for k in keys],"artifact":snapshot["artifact"],
                           "predictions":{k:v.tolist() for k,v in out.items()}}
                key = digest([name,payload])
                ack = receiver.submit(key,payload)
                if not ack["accepted"] or ack["request_id"] != key:
                    raise ValueError("独立预测交接失败")
                pending[name] = key
                metadata[name] = {"rows":len(keys),"units":len({k[0] for k in keys}),"overlap":overlap,
                                  "reused_operating_sequences":sum(signature(r[1][:,14:]) in weather for r in flights.values())}
        results = {}
        for name,key in pending.items():
            with h5py.File(dataset_path(name)) as f:
                result = score(name,receiver,key,f)
                result.update(metadata[name])
                results[name] = result
        atomic_json(directory/"scores.json",results)


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--blind",action="store_true")
    run(parser.parse_args().blind)
