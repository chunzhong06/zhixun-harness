"""只用当时已经出现的运行记录或故障终点，给参考设备首条预测的剩余寿命划界。"""
from collections import defaultdict
from hashlib import sha256
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from score_completed_choice import femto, forecast, xjtu
from score_completed_residual_transfer import LOCAL, ROOT, SourceCombinations, _trainer, atomic_json
from score_completed_residual_transfer import engines, frozen, official, read, summary, tree_values
from compare_public_selection import match_bearing_window
from unseen_sources import bearing_data, track_rows, RESULT


OUT = ROOT / "results/experiments/local-defaults/whole-maintenance/survival-dominance"
PROTOCOL = ROOT / "engineering/configs/experiments/life-survival-dominance.json"


def reference_tracks(name, package):
    """返回观察端事件；终点和测量在评分模拟器中分两类按时发放。"""
    groups = defaultdict(list)
    if name == "phm08":
        roles = read(RESULT / "issued/engine/study.json")["roles"]
        for unit_id, track in track_rows(ROOT / "data/public/phm08/train.txt"):
            u = f"PHM08/{unit_id}"
            if u in roles and roles[u] != "serve":
                # 首次预测只用首行；后续只交付发生过的采样时刻。
                groups[u] = (list(range(1, len(track)+1)), [*track[0,3:], *track[0,:3]])
        return groups
    if name == "femto":
        units, _, _ = bearing_data()
        for u, records in units.items():
            rows = sorted(records, key=lambda r:r.time_end)
            groups[u] = ([r.time_end-1 for r in rows], match_bearing_window(rows[0]).values)
    else:
        for r in package.validation:
            groups[r.unit_id].append(r)
        for u, records in list(groups.items()):
            rows = sorted(records, key=lambda r:r.time_end)
            groups[u] = ([r.time_end-1 for r in rows], rows[0].values)
    return groups


def decide(active, estimates, visible):
    if not visible:
        return active
    units = list(visible)
    current = estimates[active][units]
    target = np.asarray([visible[u][0] for u in units])
    open_end = np.asarray([not visible[u][1] for u in units])
    chosen, best_gain = active, 0.
    for arm, values in estimates.items():
        if arm == active:
            continue
        challenger = values[units]
        if np.any(open_end & (challenger < current)):
            continue
        # 差值是实际剩余时间的一次函数；在允许的下界处取最小值。
        gain = float(np.mean((challenger-current) * (2*target-current-challenger)))
        if gain > best_gain:
            chosen, best_gain = arm, gain
    return chosen


def issue(site, stream, refs, predict, order):
    ends, _, service, truth, _ = stream
    units = sorted(refs)
    if set(units) != ends.keys() or set(units) & {u for _,u,_,_ in service}:
        raise ValueError("参考设备与服务划分不同")
    observables = []
    for i, u in enumerate(units):
        clocks, _ = refs[u]
        if any(t > ends[u] for t in clocks) or clocks != sorted(set(clocks)):
            raise ValueError("参考观测时钟或原始终点不符")
        observables.extend((t, i, t-clocks[0], False) for t in clocks)
        observables.append((ends[u], i, ends[u]-clocks[0], True))
    observables.sort()
    reference = predict([refs[u][1] for u in units])
    estimates = {k: np.asarray(reference[k]) for k in order}
    service.sort(key=lambda r: (r[2], r[0]))
    forecasts = predict([r[3] for r in service])
    owner = {k:u for k,u,_,_ in service}
    if len(owner) != len(service):
        raise ValueError("服务观测重复")
    pos = 0
    visible, completed = {}, {}
    after_lower = exact_only = order[0]
    changed = []
    estimates_out = {a:{} for a in (*order, "survival", "exact-only")}
    first_failure = min(ends.values())
    early_rows = early_diff = differing = 0
    for when, records in groupby(enumerate(service), key=lambda pair: pair[1][2]):
        start = pos
        while pos < len(observables) and observables[pos][0] < when:
            _, i, bound, exact = observables[pos]
            if (i in completed or (i in visible and visible[i][0] > bound)):
                raise ValueError("现场存活时间倒退或故障后仍有观测")
            visible[i] = (bound, exact)
            if exact:
                completed[i] = (bound, True)
            pos += 1
        if pos != start:
            next_lower = decide(after_lower, estimates, visible)
            next_exact = decide(exact_only, estimates, completed)
            if next_lower != after_lower or next_exact != exact_only:
                changed.append({"time": float(when), "survival": next_lower, "exact_only": next_exact,
                                "observed": len(visible), "completed": len(completed)})
            after_lower, exact_only = next_lower, next_exact
        for i, (key, _, _, _) in records:
            for a in order:
                estimates_out[a][key] = float(forecasts[a][i])
            s, e = estimates_out[after_lower][key], estimates_out[exact_only][key]
            estimates_out["survival"][key], estimates_out["exact-only"][key] = s, e
            differing += int(s != e)
            if when <= first_failure:
                early_rows += 1
                early_diff += int(s != e)
    file = OUT/site/"issued.json"
    atomic_json(file, {"protocol": sha256(PROTOCOL.read_bytes()).hexdigest(), "order": order, "choices": changed,
                       "reference_first": {u: refs[u][0][0] for u in units}, "predictions": estimates_out})
    emitted = read(file)["predictions"]
    metric = "engine" if site == "phm08" else "bearing"
    table = summary(metric, truth(), {"parent":emitted[order[0]], **emitted}, owner)
    del table["parent"]
    result = {"scores": table, "issued_sha256": sha256(file.read_bytes()).hexdigest(), "changed_events": changed,
              "service_rows": len(service), "devices":len(set(owner.values())), "diff_vs_exact": differing,
              "rows_before_first_label":early_rows, "early_differences":early_diff,
              "exact_labels":len(completed), "observation_events":pos-len(completed), "target_fits":0}
    atomic_json(file.with_name("scores.json"), result)
    print(site,{a:round(v["score"], 3) for a,v in table.items()},early_rows,early_diff,differing,flush=True)
    return result


def main():
    results = {}
    for task in ("bearing", "engine"):
        pkg, _, batches = official(task)
        train = [r for r in batches[0].samples if r.role == "fit"]
        model_bank, _ = frozen(task)
        tree = model_bank["catboost"]
        archive, = (ROOT/"data/official"/task).glob("*.zip")
        with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
            root = Path(tmp)/"root.npz"
            root.write_bytes(pkg.checkpoint)
            original = trainer.Model.load(root)
            if task == "bearing":
                median = float(np.median([r.label for r in train]))
                def predict(x):
                    return forecast(x, tree, median, original)
                sites = {"femto":femto(54), "xjtu":xjtu(pkg,LOCAL/"fifth-offset6.json")}
                order = ("phase", "median", "tree", "original")
            else:
                repair = SourceCombinations([r.values for r in train])
                def predict(x):
                    x = np.asarray(x)
                    restored = repair.transform(x)
                    return {"restored":original.predict(restored), "tree":tree_values(tree, "engine", restored),
                            "original":original.predict(x)}
                sites = {"phm08":engines()}
                order = ("restored", "tree", "original")
            for site, stream in sites.items():
                results[site] = issue(site, stream, reference_tracks(site,pkg), predict, order)
    atomic_json(OUT/"scores.json", results)


if __name__ == "__main__":
    main()
