"""复用逐飞行数据，只把源侧实测工况分组，部署只读取一次飞行的原始可测字段。"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from score_ncmapss_field import RAW, MEMBER_SHA, file_hash, flights, rows_as_arrays, forest, measures
from harness.contracts import digest
from harness.models.condition_processing import ConditionStandardScaler
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = ROOT/"results/experiments/local-defaults/whole-maintenance/ncmapss-early-reference"


def row_groups(x,op_scale,cluster):
    return np.column_stack((x[:,:14],cluster.predict(op_scale.transform(x[:,14:]))))


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists() or file_hash(RAW)!=MEMBER_SHA:
            raise ValueError("协议已运行或源文件身份改变")
        with h5py.File(RAW) as dataset:
            values = [v.decode().strip() for v in np.ravel(dataset["W_var"][...])]
            if values!=["alt","Mach","TRA","T2"]:
                raise ValueError("工况字段与冻结协议不符")
            dev,_ = flights(dataset,"dev")
            test,_ = flights(dataset,"test")
        source = (4,5,6)
        if set(dev)!={1,2,3,4,5,6} or set(test)!={7,8,9,10}:
            raise ValueError("真实设备与冻结协议不符")
        sx,sy = rows_as_arrays([r for unit in source for r in dev[unit]])
        samples = {name:[] for name in ("early","uniform","late")}
        for unit in source:
            rows = dev[unit]
            samples["uniform"].extend(rows[i] for i in np.linspace(0,len(rows)-1,8).round().astype(int))
            samples["early"].extend(rows[:8])
            samples["late"].extend(rows[-8:])
        op = rows_as_arrays(samples["uniform"])[0][:,14:]
        op_scale = StandardScaler().fit(op)
        cluster = KMeans(n_clusters=3,n_init=10,random_state=17).fit(op_scale.transform(op))
        group_x = row_groups(sx,op_scale,cluster)
        scales = {}
        for name,ref in samples.items():
            scales[name] = ConditionStandardScaler((14,),(0,)).fit(row_groups(rows_as_arrays(ref)[0],op_scale,cluster))
        global_scale = StandardScaler().fit(sx)

        def features(x):
            group = row_groups(x,op_scale,cluster)
            return {"raw":x,"operating":x[:,14:],"global":global_scale.transform(x),
                    **{k:np.column_stack((scale.transform(group)[:,:14],x[:,14:])) for k,scale in scales.items()}}

        bank = {}
        checksums = {}
        for k,z in features(sx).items():
            tree = CatBoostRegressor(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1,thread_count=1,
                                     random_seed=17,verbose=False,allow_writing_files=False)
            tree.fit(z,sy)
            file = DEST/(k+".cbm")
            tree.save_model(str(file))
            bank[k] = CatBoostRegressor().load_model(str(file))
            checksums[k] = file_hash(file)
        import joblib
        simple = {"forest":forest().fit(sx,sy),"ridge":make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(sx,sy)}
        file = DEST/"preparation.joblib"
        joblib.dump((op_scale,cluster,scales,global_scale,simple),file)
        op_scale,cluster,scales,global_scale,simple = joblib.load(file)
        checksums["preparation"] = file_hash(file)
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        issued = {}
        truths = {}
        owners = {}
        for site,units,series in (("service",sorted(test),test),("reference",[1,2,3],dev)):
            records = [(u,r) for u in units for r in series[u]]
            x,y = rows_as_arrays([r for _,r in records])
            z = features(x)
            pred = {k:m.predict(z[k],thread_count=1) for k,m in bank.items()}
            pred.update({k:m.predict(x) for k,m in simple.items()})
            pred["half"] = .5*(pred["forest"]+pred["ridge"])
            payload = {"ids":[[u,r[0]] for u,r in records],"models":checksums,
                       "predictions":{k:v.tolist() for k,v in pred.items()}}
            req = digest([site,payload])
            ack = receiver.submit(req,payload)
            if not ack["accepted"] or ack["request_id"]!=req:
                raise RuntimeError("飞行预测交接失败")
            issued[site],truths[site],owners[site] = req,y,np.array([u for u,_ in records])
        result = {}
        for site,req in issued.items():
            p = json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
            result[site] = {k:measures(truths[site],np.asarray(v),owners[site]) for k,v in p["predictions"].items()}
            print(json.dumps({site:{k:{"误差":v["rmse"],"最差设备":v["worst"]} for k,v in result[site].items()}},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",{"sites":result,"requests":issued,"source_devices":list(source),
                    "fitted_groups":{k:len(v.groups_) for k,v in scales.items()},"h5py":h5py.__version__})


if __name__=="__main__":
    main()
