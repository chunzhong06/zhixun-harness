"""用源端仿真的真实健康参数划分故障模型，实际发放端禁止访问健康参数及机队编号。"""
import json
from pathlib import Path

import h5py
import joblib
import numpy as np

from score_flight_segment_training import BASE, PREP, SITES, load_site, dataset_path, receipt
from score_flight_complement import digest, LocalAcceptance, LocalStore, atomic_json, measures, file_hash
from harness.models.flight_fault_experts import ANCHOR, FAULTS, PRIMARY, FaultExperts
from harness.methods.flight_complement import fixed_mix

DEST = BASE / "flight-fault-experts"
PRIOR = BASE / "flight-segment-precision"
PAPER = "整趟序列"
MODES = {("HPT_eff_mod",): 0, ("HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"): 1,
         ("fan_eff_mod", "fan_flow_mod"): 2, ("HPC_eff_mod", "HPC_flow_mod"): 3}


def source_conditions(site):
    with h5py.File(dataset_path(site)) as f:
        cols = [k.decode().strip() for k in f["T_var"][:].ravel()]
        ids = f["A_dev"][:, :2].astype(int)
        starts = np.r_[0, np.flatnonzero(np.any(np.diff(ids, axis=0), axis=1)) + 1]
        ends = np.r_[starts[1:] - 1, len(ids) - 1]
        first = f["T_dev"][starts]
        last = f["T_dev"][ends]
    if len(np.unique(ids[starts], axis=0)) != len(starts):
        raise ValueError("训练航班ID重复")
    mode = {}
    for unit in np.unique(ids[:, 0]):
        where = np.flatnonzero(ids[starts, 0] == unit)
        change = last[where[-1]] - first[where[0]]
        active = tuple(k for k, val in zip(cols, change, strict=True) if abs(val) > 1e-6)
        mode[int(unit)] = MODES[active]
    return {tuple(key): (mode[int(key[0])], val) for key, val in zip(ids[starts], (first + last) / 2, strict=True)}


def make_bank():
    prep = joblib.load(PREP).prep
    previous = json.loads((PRIOR / "scores.json").read_bytes())
    bank, records = {}, {}
    for site in SITES:
        keys, x, truth = load_site(site, "dev", prep)
        source = source_conditions(site)
        if set(keys) != set(source):
            raise ValueError("源端健康标志与训练测量无法对齐")
        bank[site + "_source"] = x
        bank[site + "_life"] = truth
        bank[site + "_modes"] = np.array([source[k][0] for k in keys])
        bank[site + "_health"] = np.stack([source[k][1] for k in keys])
        bank[site + "_ids"] = np.asarray(keys)
    for site, v in previous["sites"].items():
        ids, x, y = load_site(site, "test", prep)
        old = receipt(PRIOR, site, v)
        if old["ids"] != [list(k) for k in ids]:
            raise ValueError("开发航班与旧回执错位")
        bank[site + "_target"] = x
        records[site] = (ids, y, old["predictions"])
    np.savez_compressed(DEST / "features.npz", **bank)
    return bank, records


def issue(folder, site, ids, pred, truth):
    receiver = LocalAcceptance()
    receiver.output = folder / "accepted"
    payload = {"ids": [list(k) for k in ids], "predictions": {k: np.asarray(v).tolist() for k,v in pred.items()}}
    key = digest([site, payload])
    if not receiver.submit(key, payload)["accepted"]:
        raise ValueError("航班预测未交接")
    sent = receipt(folder, site, {"request": key})["predictions"]
    owner = np.asarray(ids)[:, 0]
    return {"request": key, "rows": len(owner), "scores": {k: measures(truth, np.asarray(v), owner) for k,v in sent.items()}}


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("故障配对已完成")
        data, records = make_bank()
        x, y, groups = [np.concatenate([data[s + suffix] for s in SITES]) for suffix in ("_source", "_life", "_modes")]
        model = FaultExperts().fit(x, y, groups)
        weights = DEST / "experts.joblib"
        joblib.dump(model, weights)
        tools = [ANCHOR, PAPER, *(k + "专长" for k in FAULTS)]
        matrices = {}
        for site, (_, _, pred) in records.items():
            new, _ = model.outputs(data[site + "_target"], np.array(pred[ANCHOR]))
            pred.update(new)
            matrices[site] = np.column_stack([pred[k] for k in tools])
        scores = {}
        for site, (ids, truth, pred) in records.items():
            others = [s for s in records if s != site]
            train = np.vstack([matrices[s] for s in others])
            labels = np.concatenate([records[s][1] for s in others])
            sites = np.concatenate([np.repeat(s, len(records[s][1])) for s in others])
            mix = fixed_mix(train, labels, sites)
            pred["故障专家与统一及论文固定组合"] = matrices[site] @ mix
            scores[site] = issue(DEST, site, ids, pred, truth)
        means = {k: float(np.mean([site["scores"][k]["rmse"] for site in scores.values()])) for k in pred}
        controls = [ANCHOR, "三段和论文固定最优", "故障专家与统一及论文固定组合", "专长固定均值", "专长按源比例",
                    "统一和源比例各半", *(k + "专长" for k in FAULTS)]
        # 其余旧固定排名已逐组在同回执保存，旧最强为ANCHOR。
        fixed = min(controls, key=means.get)
        gain = 1 - means[PRIMARY] / means[fixed]
        decision = {"现场": PRIMARY, "固定": fixed, "降幅": gain,
                    "进独立": gain >= .01 and means[PRIMARY] < .99 * means["三段和论文固定最优"]}
        atomic_json(DEST / "scores.json", {"sites": scores, "means": means, "decision": decision,
                    "字段缓存SHA256": file_hash(DEST / "features.npz"), "权重SHA256": file_hash(weights)})
        print(json.dumps({"开发": {k: means[k] for k in [PRIMARY, "按当前故障概率", *controls]}, "判定": decision}, ensure_ascii=False))


if __name__ == "__main__":
    run()
