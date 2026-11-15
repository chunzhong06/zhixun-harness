"""机队级留出研究和新机队盲测共用工具；两阶段互相独立，盲测只加载发布产物。"""
from argparse import ArgumentParser
import json

import h5py
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from score_engine_flight_load import DEST as FIRST, OLD, RAW, signals, score, old_arms
from score_engine_flight_forward import signature
from score_ds02_forward import read_flights
from score_ncmapss_field import file_hash
from harness.contracts import digest
from harness.methods.fleet_training import fleet_trials
from harness.models.fleet_flight import flight_views
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = OLD.parent / "engine-fleet-training"
SEEN = ("DS01", "DS02", "DS03", "DS04")


def dataset_path(site):
    path, = RAW.parent.glob(f"N-CMAPSS_{site}*.h5")
    return path


def arrays(flights, prep):
    rows = [flight_views(prep, *flights[k]) for k in sorted(flights)]
    return {v: np.array([r[v] for r in rows]) for v in rows[0]}


def source_rows():
    prep = joblib.load(FIRST / "tools.joblib")[0]
    history, measured, operating, source_signals = {}, set(), set(), {}
    for site in SEEN:
        with h5py.File(dataset_path(site)) as f:
            flights = signals(f, "dev")
            source_signals[site] = flights
            keys = sorted(flights)
            answers = read_flights(f, "dev", answers=True)
            if set(keys) != set(answers):
                raise ValueError("源训练飞行的标签和输入错位")
            history[site] = (arrays(flights, prep), np.array([answers[k] for k in keys]), np.array([k[0] for k in keys]))
            for part in (flights, signals(f, "test")):
                for _, records in part.values():
                    measured.add(signature(records))
                    operating.add(signature(records[:, 14:]))
    original = {n: h[0] for n,h in history.items()}
    fold_features = {held: original for held in SEEN}
    # 留出DS01时连健康刻度也须从另一机队重估；另外三折均可用原DS01训练侧刻度。
    alternative = source_signals["DS02"]
    early = [alternative[k][1] for u in sorted({k[0] for k in alternative})[:3]
             for k in sorted(k for k in alternative if k[0]==u)[:8]]
    ref = np.concatenate(early)
    scale = StandardScaler().fit(ref[:,14:])
    healthy = make_pipeline(PolynomialFeatures(2,include_bias=False), Ridge(alpha=1.)).fit(
        scale.transform(ref[:,14:]),ref[:,:14])
    fold_features["DS01"] = {n:arrays(v,(scale,healthy,None,None)) for n,v in source_signals.items()}
    return prep, history, fold_features, measured, operating


def fit_source():
    prep, history, fold_features, measured, operating = source_rows()
    selected, tools, trace = fleet_trials(history,fold_features)
    joblib.dump((selected, tools, prep), DEST / "fleet.joblib")
    atomic_json(DEST / "training.json", {"selected": selected, "cross_validation": trace,
                                         "artifact": file_hash(DEST / "fleet.joblib"),
                                         "source_rows": {k: len(v[1]) for k, v in history.items()},
                                         "measured_flights": sorted(measured), "operating_profiles": sorted(operating)})
    print(json.dumps({"源端留整组": trace}, ensure_ascii=False), flush=True)


def evaluate(sites, directory, blind):
    from engineering.cli.workflow.flight_input_predict import infer_fleet
    path = DEST / "fleet.joblib"
    state = json.loads((DEST / "training.json").read_bytes())
    if file_hash(path) != state["artifact"]:
        raise ValueError("冻结的训练模型身份发生变化")
    bundle = joblib.load(path)
    selected, tools, prep = bundle
    if selected != state["selected"]:
        raise ValueError("实际发放与源端选择不符")
    measured, operating = set(state["measured_flights"]), set(state["operating_profiles"])
    receiver = LocalAcceptance()
    receiver.output = directory / "accepted"
    pending, overlap = {}, {}
    older = joblib.load(FIRST / "tools.joblib")
    for site in sites:
        with h5py.File(dataset_path(site)) as f, h5py.File(RAW) as reference:
            if any(not np.array_equal(f[k][...], reference[k][...]) for k in ("W_var","X_s_var","A_var")):
                raise ValueError("字段顺序变化")
            flights = signals(f, "test")
            keys = sorted(flights)
            x = arrays(flights, prep)
            shared = sum(signature(v[1]) in measured for v in flights.values())
            if blind and shared:
                raise ValueError("待盲测的飞行与已见传感器记录重叠")
            overlap[site] = {"rows": len(keys), "measured_overlap": shared,
                             "operating_reuse": sum(signature(v[1][:,14:]) in operating for v in flights.values())}
            out = old_arms(x["原始均值"])
            from score_engine_flight_load import features
            a,b,coverage = features(flights,older[0])
            full, comparable = older[1][0].predict(a), older[2][0].predict(b)
            out.update({"旧全程森林":full, "旧可比工况森林":comparable,
                        "旧按覆盖两森林":coverage*full+(1-coverage)*comparable})
            out.update({k: m.predict(x[k.split("/")[0]]) for k,m in tools.items()})
            aux = f["A_test"][:,:2].astype(int)
            borders = np.r_[0,np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))+1,len(aux)]
            primary = {}
            for start,end in zip(borders[:-1],borders[1:],strict=True):
                request = {"inputs":{"sensors":f["X_s_test"][start:end],"settings":f["W_test"][start:end]}}
                response, = infer_fleet(bundle,[request])
                primary[tuple(map(int,aux[start]))] = response["outputs"]["value"]
            out["系统选用"] = np.array([primary[k] for k in keys])
            np.testing.assert_allclose(out["系统选用"],out[selected],atol=1e-8,rtol=0)
            # 避免误把分别发布的选中模型与研究侧批量推理细微舍入差异视为提分。
            payload = {"ids":[list(k) for k in keys], "selection":selected, "artifact":state["artifact"],
                       "predictions":{k:v.tolist() for k,v in out.items()}}
            key = digest([site,payload])
            ack = receiver.submit(key,payload)
            if not ack["accepted"] or ack["request_id"] != key:
                raise RuntimeError("航班交付未获接收")
            pending[site] = key
    results = {}
    for site,key in pending.items():
        with h5py.File(dataset_path(site)) as f:
            results[site] = score(site,receiver,key,f)
            results[site]["overlap"] = overlap[site]
    atomic_json(directory/"scores.json",{"selection":selected,"sites":results})


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--blind", action="store_true")
    args = parser.parse_args()
    directory = DEST / "forward" if args.blind else DEST
    with LocalStore(directory/"lock").exclusive():
        if (directory/"scores.json").exists():
            raise ValueError("已有机队得分")
        if not args.blind:
            fit_source()
        evaluate(("DS05",) if args.blind else SEEN,directory,args.blind)
