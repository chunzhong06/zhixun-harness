"""五组历史机队准备候选并记录已见成绩，第六组的单次发放用冻结训练产物。"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import h5py
import joblib
import numpy as np

from score_engine_fleet_training import dataset_path, arrays
from score_engine_flight_load import DEST as FIRST, RAW, signals, score, old_arms, features
from score_engine_flight_forward import signature
from score_ds02_forward import read_flights
from score_ncmapss_field import file_hash
from score_flight_disagreement import DEST as PREVIOUS
from harness.contracts import digest
from harness.methods.fleet_training import train_recipe
from harness.methods.flight_disagreement import combine
from harness.models.flight_load_neighbors import FlightLoadNeighbors, load_profile, RULES
from harness.runtime.storage import LocalStore, atomic_json
from harness.runtime.handoff import LocalAcceptance
from engineering.cli.workflow.flight_input_predict import infer_fleet

DEST = FIRST.parent / "flight-load-neighbors"
SITES = ("DS01","DS02","DS03","DS04","DS05")
ROOT = RAW.parents[3]


def read(path):
    return json.loads(path.read_bytes())


def train():
    prep = joblib.load(FIRST/"tools.joblib")[0]
    feats, profiles, targets, fleets, units = [], [], [], [], []
    known, weather = set(), set()
    for site in SITES:
        with h5py.File(dataset_path(site)) as data:
            source = signals(data,"dev")
            labels = read_flights(data,"dev",answers=True)
            keys = sorted(source)
            if set(keys) != set(labels):
                raise ValueError("作者源训练飞行与寿命不对齐")
            feats.append(arrays(source,prep))
            profiles.extend(load_profile(prep,source[k][1]) for k in keys)
            targets.extend(labels[k] for k in keys)
            fleets.extend([site]*len(keys))
            units.extend(u for u,_ in keys)
            for part in (source,signals(data,"test")):
                for _,rows in part.values():
                    known.add(signature(rows))
                    weather.add(signature(rows[:,14:]))
    y = np.array(targets)
    load = FlightLoadNeighbors(prep,np.array(profiles),y,np.array(fleets),np.array(units))
    fixed = {}
    for view in feats[0]:
        x = np.vstack([matrix[view] for matrix in feats])
        for learner in ("岭","森林"):
            name = view+"/"+learner
            fixed[name] = train_recipe(name,x,y)
    joblib.dump(fixed,DEST/"five-source.joblib")
    atomic_json(DEST/"overlap.json",{"measurements":sorted(known),"operating":sorted(weather)})
    return load, fixed


def predict_all(load, fixed, flights):
    keys = sorted(flights)
    x = arrays(flights,load.preparation)
    profiles = np.array([load_profile(load.preparation,flights[k][1]) for k in keys])
    out = load.predictions(profiles)
    out.update({"五组"+k:tool.predict(x[k.split("/")[0]]) for k,tool in fixed.items()})
    old = joblib.load(FIRST/"tools.joblib")
    a,b,c = features(flights,old[0])
    full,local = old[1][0].predict(a),old[2][0].predict(b)
    out.update(old_arms(x["原始均值"]))
    out.update({"旧全程森林":full,"旧可比工况森林":local,"旧按覆盖两森林":c*full+(1-c)*local})
    previous = joblib.load(PREVIOUS/"mixture.joblib")
    snap = read(PREVIOUS/"training.json")
    p = {k:tool.predict(x[k.split("/")[0]]) for k,tool in previous["tools"].items()}
    out.update(p)
    matrix = np.column_stack([p[k] for k in snap["order"]])
    w = snap["weights"]
    out.update({"全局固定组合":matrix@w["global"],"等权组合":matrix.mean(1),
                "整组选择":p[snap["selected"]],"现场按分歧":combine(matrix,w)})
    for i,v in enumerate(w["local"],1):
        out[f"固定分歧组{i}"] = matrix@v
    return out


def batch_sequence(rows):
    # 输入为当前完整航班的共用采样读数，固定长度序列保留时间顺序，不调入其他航班。
    return np.asarray(rows[np.linspace(0,len(rows)-1,100).round().astype(int)],dtype=np.float32)


def deliver(directory, name, load, fixed, blind, receiver, known, weather):
    path = dataset_path(name)
    with h5py.File(path) as data,h5py.File(RAW) as src:
        if any(not np.array_equal(data[k][...],src[k][...]) for k in ("W_var","X_s_var","A_var")):
            raise ValueError("新机队传感器或工况字段不同")
        flights = signals(data,"test")
        keys = sorted(flights)
        sigs = [signature(flights[k][1]) for k in keys]
        overlap = sum(s in known for s in sigs)
        if len(set(sigs))!=len(keys) or blind and overlap:
            raise ValueError("发现重复的评价测量")
        out = predict_all(load,fixed,flights)
        if blind:
            # 从原始全航班测量独立调用交付入口，逐条复核候选及新增论文适配。
            aux = data["A_test"][:,:2].astype(int)
            ends = np.r_[0,np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))+1,len(aux)]
            served = {}
            for lo,hi in zip(ends[:-1],ends[1:],strict=True):
                req = {"inputs":{"sensors":data["X_s_test"][lo:hi],"settings":data["W_test"][lo:hi]}}
                answer, = infer_fleet(load,[req])
                served[tuple(map(int,aux[lo]))] = answer["outputs"]["value"]
            out["系统实际交付"] = np.array([served[k] for k in keys])
            np.testing.assert_allclose(out["系统实际交付"],out[load.rule],rtol=0,atol=1e-9)
            paper = DEST/"paper"
            with TemporaryDirectory(dir=directory) as tmp:
                inp,output = Path(tmp)/"inputs.npy",Path(tmp)/"outputs.npy"
                np.save(inp,np.stack([batch_sequence(flights[k][1]) for k in keys]))
                call_paper(["predict",str(inp),str(output),str(paper/"paper.pt")])
                out["Darrah双向序列本地适配"] = np.load(output)
        payload = {"ids":[list(k) for k in keys],"predictions":{k:v.tolist() for k,v in out.items()},
                   "selected":load.rule if blind else None}
        key = digest([name,payload])
        ack = receiver.submit(key,payload)
        if not ack["accepted"] or ack["request_id"]!=key:
            raise RuntimeError("航班交接失败")
        return key, {"units":len({u for u,_ in keys}),"overlap":overlap,
                     "reused_weather":sum(signature(flights[k][1][:,14:]) in weather for k in keys)}


def call_paper(args):
    # 主环境无需依赖 PyTorch；可显式选用已安装 CPU 版 PyTorch 的解释器。
    python = os.environ.get("FLIGHT_PAPER_PYTHON", sys.executable)
    process = subprocess.run([python,"-B",
                              str(ROOT/"engineering/cli/study/flight_paper_cpu.py"),*args],
                             cwd=ROOT,check=True)
    return process.returncode


def main(blind):
    directory = DEST/"forward" if blind else DEST
    with LocalStore(directory/"lock").exclusive():
        if (directory/"scores.json").exists():
            raise ValueError("此机队已有得分")
        if blind:
            meta = read(DEST/"training.json")
            if file_hash(DEST/"flight.joblib")!=meta["artifact"] or file_hash(DEST/"five-source.joblib")!=meta["fixed"]:
                raise ValueError("训练产物与冻结身份不符")
            load = joblib.load(DEST/"flight.joblib")
            fixed = joblib.load(DEST/"five-source.joblib")
        else:
            load,fixed = train()
        known = read(DEST/"overlap.json")
        receiver = LocalAcceptance()
        receiver.output = directory/"accepted"
        pending, result = {}, {}
        for site in (("DS06",) if blind else SITES):
            pending[site] = deliver(directory,site,load,fixed,blind,receiver,set(known["measurements"]),set(known["operating"]))
        for site,(req,counts) in pending.items():
            with h5py.File(dataset_path(site)) as data:
                result[site] = {**score(site,receiver,req,data),**counts}
        if not blind:
            means = {rule:float(np.mean([s["scores"][rule]["rmse"] for s in result.values()])) for rule in RULES}
            load.rule = min(means,key=means.get)
            joblib.dump(load,DEST/"flight.joblib")
            atomic_json(DEST/"training.json",{"rule":load.rule,"means":means,"artifact":file_hash(DEST/"flight.joblib"),
                                              "fixed":file_hash(DEST/"five-source.joblib")})
            print(json.dumps({"开发候选均值":means},ensure_ascii=False),flush=True)
        atomic_json(directory/"scores.json",result)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--blind",action="store_true")
    main(parser.parse_args().blind)
