"""把按失效回标维护模型的方法迁移到 N-CMAPSS 新发动机，每次飞行落地后发一条预测。"""
from collections import Counter, defaultdict
from hashlib import sha256
from time import process_time
import json
import math

import h5py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness.models.feedback_choice import HistoricalToolChoice
from harness.runtime.storage import atomic_json
from read_ncmapss_member import DEST as RAW
from read_ncmapss_member import MEMBER_SHA, ROOT


PROTOCOL = ROOT/"engineering/configs/experiments/ncmapss-first-field.json"
OUT = ROOT/"results/experiments/local-defaults/whole-maintenance/ncmapss-ds01"
TOOLS = ("forest", "ridge", "median", "half")


def file_hash(path):
    with path.open("rb") as stream:
        import hashlib
        return hashlib.file_digest(stream, "sha256").hexdigest()


def flights(raw, split):
    data = raw["X_s_"+split], raw["W_"+split], raw["A_"+split], raw["Y_"+split]
    count = len(data[0])
    if any(len(v) != count for v in data):
        raise ValueError("四组传感器、工况、辅助编号和答案不对齐")
    sums = {}
    sizes = Counter()
    answers = {}
    for start in range(0, count, 65536):
        sensor, settings, aux, target = [d[start:start+65536] for d in data]
        values = np.column_stack([sensor, settings])
        group = np.asarray(aux[:, :2], dtype=int)
        if not np.array_equal(group, aux[:, :2]) or not np.isfinite(values).all():
            raise ValueError("观测缺失或设备／飞行身份无效")
        unique, inverse = np.unique(group,axis=0,return_inverse=True)
        for index, item in enumerate(unique):
            key = tuple(map(int, item))
            mask = inverse == index
            block = values[mask]
            labels = np.ravel(target[mask])
            if (not np.isfinite(labels).all() or not np.all(labels == labels[0])
                    or answers.setdefault(key, float(labels[0])) != labels[0]):
                raise ValueError("同一飞行的剩余循环标签不一致")
            sums[key] = sums.get(key, 0) + block.sum(axis=0)
            sizes[key] += len(block)
    table = defaultdict(list)
    for (unit, cycle), total in sorted(sums.items()):
        table[unit].append((cycle, total/sizes[unit, cycle], answers[unit, cycle]))
    for unit, rows in table.items():
        last = rows[-1][0]
        if ([r[0] for r in rows] != list(range(rows[0][0], last+1)) or
                any(last-c != y for c, _, y in rows)):
            raise ValueError(f"设备 {unit} 的记录没有明确的连续失效终点")
    return table, count


def rows_as_arrays(rows):
    return np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def forest():
    return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=17, n_jobs=1)


def measures(y, p, owner):
    squared = (y-p)**2
    unit_scores = {str(u): float(np.sqrt(squared[owner == u].mean())) for u in np.unique(owner)}
    return {"rmse":float(np.sqrt(squared.mean())), "per_unit":unit_scores,
            "worst": max(unit_scores.values())}


def run():
    if file_hash(RAW) != MEMBER_SHA:
        raise ValueError("冻结的 NASA HDF5 原件身份变化")
    with h5py.File(RAW, "r") as raw:
        fields = {k: [v.decode().strip() for v in np.ravel(raw[k][...])] for k in ("A_var", "X_s_var", "W_var")}
        if fields["A_var"][:2] != ["unit", "cycle"]:
            raise ValueError(f"设备列和循环列不符：{fields['A_var']}")
        dev, n_dev = flights(raw,"dev")
        test, n_test = flights(raw,"test")
    ordered = sorted(dev, key=lambda u: sha256(f"nc-field-v1/{u}".encode("ascii")).digest())
    n_source = math.ceil(len(ordered)/2)
    source_units, refs = ordered[:n_source], ordered[n_source:]
    if not refs or not test or set(dev) & set(test):
        raise ValueError("设备划分不符")
    sx, sy = rows_as_arrays([row for u in source_units for row in dev[u]])
    base = forest().fit(sx,sy)
    ridge = make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(sx,sy)
    def fixed(x):
        f, r = base.predict(x), ridge.predict(x)
        return {"forest": f, "ridge": r, "median": np.full(len(x), np.median(sy)), "half": (f+r)/2}

    def build(rows):
        tx, ty = rows_as_arrays(rows)
        weights = np.r_[np.ones(len(sy)), np.full(len(ty), len(sy)/len(ty))]
        return forest().fit(np.vstack((sx,tx)),np.r_[sy,ty],sample_weight=weights)

    service = defaultdict(list)
    truth = {}
    owners = {}
    for u, track in test.items():
        for flight, values, target in track:
            k = f"{u}/{flight}"
            service[flight-track[0][0]+1].append((k,values))
            truth[k], owners[k] = target, u
    ends = sorted((len(dev[u]),u) for u in refs)
    policy = HistoricalToolChoice(TOOLS,"forest")
    recent = "forest"
    all_rows, fit_rows, confirm_sets = [], [], []
    all_model = paired = active = base
    pos = 0
    counts = Counter()
    fit_time = Counter()
    transitions = []
    files = []
    for when, batch in sorted(service.items()):
        while pos < len(ends) and ends[pos][0] < when:
            time_end, u = ends[pos]
            track = dev[u]
            ix = np.linspace(0,len(track)-1,min(5,len(track))).round().astype(int)
            revealed = [track[i] for i in ix]
            rx, ry = rows_as_arrays(revealed)
            p = fixed(rx)
            losses = {a:float(np.mean((p[a]-ry)**2)) for a in TOOLS}
            policy.update([losses])
            recent = min(TOOLS,key=losses.get)
            all_rows.extend(revealed)
            start = process_time()
            all_model = build(all_rows)
            fit_time["all"] += process_time()-start
            counts["all"] += 1
            if pos % 2 == 0:
                fit_rows.extend(revealed)
                start = process_time()
                paired = build(fit_rows)
                fit_time["paired"] += process_time()-start
                counts["paired"] += 1
            else:
                confirm_sets.append((rx,ry))
            accepted = False
            if confirm_sets and paired is not active:
                def mean_mse(model):
                    return float(np.mean([np.mean((model.predict(xx)-yy)**2) for xx,yy in confirm_sets]))
                if mean_mse(paired) < mean_mse(active):
                    active = paired
                    accepted = True
            transitions.append({"time":when,"unit":u,"failure":time_end,"returned":[track[i][0] for i in ix],
                                "role":"fit" if pos%2 == 0 else "confirm","loss":losses,"chosen":policy.active,
                                "paper_recent":recent,"paired_adopted":accepted})
            pos += 1
        ids, values = zip(*batch,strict=True)
        x = np.asarray(values)
        p = fixed(x)
        p.update({"project":p[policy.active], "once":p[policy.first], "mature-recent":p[recent],
                  "all-refit":all_model.predict(x), "paired-always":paired.predict(x), "paired-gate":active.predict(x)})
        file = OUT / f"issued-{when:04}.json"
        atomic_json(file,{"cycle_since_start":when,"keys":list(ids),"finished_reference_units":pos,
                          "choice":policy.active,"predictions":{a:v.tolist() for a,v in p.items()}})
        files.append(file)
    saved = [json.loads(f.read_bytes()) for f in files]
    keys = [k for row in saved for k in row["keys"]]
    if len(keys) != len(set(keys)) or set(keys) != truth.keys():
        raise ValueError("已交付的飞行重复或缺失")
    owner = np.asarray([owners[k] for k in keys])
    true = np.asarray([truth[k] for k in keys])
    pred = {a:np.concatenate([r["predictions"][a] for r in saved]) for a in saved[0]["predictions"]}
    table = {a:measures(true,p,owner) for a,p in pred.items()}
    controls = (*TOOLS,"all-refit","paired-always","mature-recent","once")
    strong = min(controls,key=lambda a:table[a]["rmse"])
    decision = {a: all(table[a]["rmse"] <= .99*table[c]["rmse"] and table[a]["worst"] <= 1.01*table[c]["worst"]
                       for c in controls)
                for a in ("project","paired-gate")}
    result = {"protocol_sha256":file_hash(PROTOCOL), "raw_sha256":MEMBER_SHA, "source_units":source_units,
              "reference_units":refs,"service_units":sorted(test),"source_flights":len(sy),"rows":len(keys),
              "raw_observations":[n_dev,n_test],"fields":fields,"labels":len(all_rows),"transitions":transitions,
              "fits_excluding_source":dict(counts),"fit_cpu_seconds":dict(fit_time),"scores":table,
              "project_changes_from_source":int(np.sum(pred["project"] != pred["forest"])),
              "project_changes_from_once":int(np.sum(pred["project"] != pred["once"])),
              "before_first_feedback":sum(len(row["keys"]) for row in saved if row["finished_reference_units"] == 0),
              "best_control":strong,"meets_gate":decision,"issued_sha256":[file_hash(f) for f in files]}
    atomic_json(OUT/"scores.json",result)
    print(source_units,refs,sorted(test),len(keys),result["before_first_feedback"],len(all_rows),
          {a:round(v["rmse"],4) for a,v in table.items()},flush=True)


if __name__ == "__main__":
    run()
