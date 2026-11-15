"""冻结已交接权重和预测，单独检验十轮预算结束后的读数到达间隔。"""
from collections import defaultdict
from hashlib import sha256
from math import isclose

import numpy as np

from check_prior_device_prediction import OUT as PRIOR, issue
from compare_group_objectives import ROOT, atomic_json, delivered, digest, official, read, reference_folders, summary
from compare_proposal_frequency import BASE, TASKS, remaining_cycles, remaining_minutes

DEST = BASE/"service-cadence"
SEEDS = (17, 47, 89)


def clock_batches(observations, width, limit):
    groups = defaultdict(list)
    for sid, unit, time in observations:
        window = (time-1)//width
        if limit is not None:
            window = min(window, limit-1)
        groups[window].append((sid, unit, time))
    return [groups[k] for k in sorted(groups)]


def lags(batches):
    last, gaps = {}, {}
    for rows in batches:
        for sid, unit, time in rows:
            if unit in last:
                if last[unit] >= time:
                    raise ValueError("过去批次时间没有早于当前设备时钟")
                gaps[sid] = time-last[unit]
        for _, unit, time in rows:
            last[unit] = max(last.get(unit, 0), time)
    return {"n":len(gaps), "quantiles_0_50_90_100":np.quantile(list(gaps.values()), [0,.5,.9,1]).tolist()}


def score_parts(task, truth, streams, devices, tail):
    result = {}
    for name, subset in (("whole",set(devices)), ("first_nine",set(devices)-tail), ("tail",tail)):
        part = {arm:{i:p for i,p in preds.items() if i in subset} for arm,preds in streams.items()}
        metrics = summary(task, truth, part, devices)
        prior = metrics["parent"]["group_scores"]
        for arm, values in metrics.items():
            current = values["group_scores"]
            values["better_devices"] = sum(current[k] < v-1e-12 for k,v in prior.items())
            values["worse_devices"] = sum(current[k] > v+1e-12 for k,v in prior.items())
        result[name] = metrics
    return result


def train_gap_coverage(rows, width):
    by_unit = defaultdict(list)
    for row in rows:
        by_unit[row.unit_id].append(row.time_end)
    pairs = units = 0
    gaps = []
    for times in by_unit.values():
        times.sort()
        gaps.extend(np.diff(times))
        # 最早两点足够近即可配成一对，过远时最早点也无法与更晚的点配对。
        j = count = 0
        while j+1 < len(times):
            if times[j+1]-times[j] <= width:
                count += 1
                j += 2
            else:
                j += 1
        pairs += count
        units += int(count>0)
    return {"rows":len(rows), "devices":len(by_unit), "eligible_devices":units,
            "max_disjoint_short_pairs":pairs, "median_adjacent_gap":float(np.median(gaps))}


def peer_assignments(cohort, previous, samples):
    """只在已关闭窗口的同时刻、同系列观测间置换。全组单环保留供体数量且没有自配。"""
    groups = defaultdict(list)
    for unit, (sid, time) in previous.items():
        if cohort == "pronostia":
            series = unit.rsplit("_", 1)[0]
        elif cohort == "engine":
            series = unit.split(":", 1)[0]
        else:
            series = cohort
        condition = str(samples[sid]["group"]) if TASKS[cohort] == "engine" else ""
        groups[(time, series, condition)].append(unit)
    maps = {seed:{} for seed in SEEDS}
    for key, units in groups.items():
        if len(units) < 2:
            continue
        for seed in SEEDS:
            ring = sorted(units, key=lambda u: digest([seed, cohort, key, u]))
            maps[seed].update(zip(ring, ring[1:] + ring[:1], strict=True))
    return maps


def identity_predictions(cohort, regular, samples, raw):
    out = dict(parent={}, own_average={}, own_clock={})
    for seed in SEEDS:
        out[f"peer_average_{seed}"] = {}
        out[f"peer_clock_{seed}"] = {}
    prior = {}
    eligible = []
    for batch in regular:
        peers = peer_assignments(cohort, prior, samples)
        latest = {}
        for sid, unit, t in batch:
            if unit in peers[SEEDS[0]]:
                prev, prev_t = prior[unit]
                if prev_t >= t:
                    raise ValueError("置换供体必须早于当前观测")
                assignments = []
                out["parent"][sid] = raw[sid]
                out["own_average"][sid] = .5 * (raw[sid] + raw[prev])
                out["own_clock"][sid] = .5 * (raw[sid] + raw[prev] - (t-prev_t))
                for seed in SEEDS:
                    other = peers[seed][unit]
                    donor, donor_t = prior[other]
                    if donor_t != prev_t or other == unit:
                        raise ValueError("供体与本机时间未匹配或同一设备自配")
                    out[f"peer_average_{seed}"][sid] = .5 * (raw[sid] + raw[donor])
                    out[f"peer_clock_{seed}"][sid] = .5 * (raw[sid] + raw[donor] - (t-prev_t))
                    assignments.append(donor)
                eligible.append([sid, prev, *assignments])
            if unit not in latest or t > latest[unit][1]:
                latest[unit] = (sid, t)
        prior = latest
    return out, digest(eligible)


def compare(cohort):
    task = TASKS[cohort]
    spec = read(ROOT/"engineering/configs/experiments/official-rul-series.json")
    width = 5 if cohort == "pronostia" else spec["service_window_width"][task]
    fixed, paper = reference_folders(cohort, "mature")
    project = BASE/"selected"/task if task == cohort else fixed.parent/"project"
    samples = read(fixed/"state.json")["samples"]
    service = [(sid,r["unit_id"],r["time_end"]) for sid,r in samples.items() if r["role"]=="serve"]
    devices = {sid:unit for sid,unit,_ in service}
    if any(samples[sid]["label"] is not None or samples[sid]["available_at"] != min((time-1)//width,9)
           for sid,_,time in service):
        raise ValueError("服务批次使用了时钟宽度之外的分段，或携带答案")
    role_units = {role:{r["unit_id"] for r in samples.values() if r["role"]==role}
                  for role in ("explore","adopt","serve")}
    if any(a & b for a,b in ((role_units["serve"],role_units["adopt"]),
                              (role_units["serve"],role_units["explore"]),(role_units["adopt"],role_units["explore"]))):
        raise ValueError("探索、确认与服务的设备身份重合")
    recorded = clock_batches(service, width, 10)
    regular = clock_batches(service, width, None)
    if len(recorded)!=10 or regular[:9]!=recorded[:9]:
        raise ValueError("前九窗输入发生变化")
    tail = {i for i,_,_ in recorded[-1]}
    prior = PRIOR/cohort/"issued.json"
    if sha256(prior.read_bytes()).hexdigest() != read(prior.with_name("scores.json"))["issued_sha256"]:
        raise ValueError("上一轮封存内容摘要不符")
    original = read(prior)["outputs"]
    output = {}
    model_ids = {}
    for arm, folder in (("project",project), ("paper",paper)):
        stream = delivered(folder)
        if stream != original[arm]["parent"] or stream.keys() != devices.keys():
            raise ValueError("每条原预测必须与原交接一致")
        rounds = read(folder/"run.json")["rounds"]
        last_receipt = read(folder/rounds[-1]["submitted"])["payload"]
        if last_receipt["predictions"].keys() != tail or last_receipt["model_id"] != rounds[-1]["selected"]:
            raise ValueError("分开的末批并非来自同次权重交接")
        model_ids[arm] = last_receipt["model_id"]
        output[arm] = {}
        for schedule, batches in (("recorded",recorded), ("regular",regular)):
            proposals, _ = issue(batches, stream)
            if schedule=="recorded" and proposals != original[arm]:
                raise ValueError("未复现旧回放每一条输出")
            if any(proposals[k][i] != original[arm][k][i] for k in proposals for i in set(devices)-tail):
                raise ValueError("时序改变波及第十轮以前的预测")
            output[arm][schedule] = proposals
    # 两个日程的逐条预测先封存，独立评价才读取答案。
    dest = DEST/cohort
    path = dest/"issued.json"
    payload = {"prior_sha256":sha256(prior.read_bytes()).hexdigest(), "clock_width":width,
               "clock_input_sha256":digest(service), "last_model_ids":model_ids, "outputs":output}
    if path.exists() and read(path) != payload:
        raise ValueError("重跑与前次封存的逐条预测不同")
    atomic_json(path, payload)
    peer_path = dest/"identity-issued.json"
    identity, eligibility = {}, {}
    for arm in output:
        identity[arm], eligibility[arm] = identity_predictions(cohort, regular, samples, original[arm]["parent"])
        own = output[arm]["regular"]
        for key, baseline in (("own_average","average"),("own_clock","clock")):
            if any(not isclose(value,own[baseline][sid],rel_tol=1e-12,abs_tol=1e-10)
                   for sid,value in identity[arm][key].items()):
                raise ValueError("子集上的同机估计未复现连续时钟的结果")
    if eligibility["project"] != eligibility["paper"]:
        raise ValueError("两个系统匹配了不同的供体")
    peer_record = {"parent_sha256":sha256(path.read_bytes()).hexdigest(), "seeds":list(SEEDS),
                   "assignments_sha256":eligibility["project"], "outputs":identity}
    if peer_path.exists() and read(peer_path) != peer_record:
        raise ValueError("置换预测未复现")
    atomic_json(peer_path,peer_record)
    if cohort=="phm08":
        truth = remaining_cycles()
    elif cohort=="pronostia":
        truth = remaining_minutes("full-test-manifest.json")
    else:
        truth = {r.sample_id:r.label for r in official(task)[0].validation}
    tables = {}
    for arm, schedules in output.items():
        tables[arm] = {}
        for key, streams in schedules.items():
            clipped = {arm:{i:max(0.,v) for i,v in p.items()} for arm,p in streams.items()}
            raw = score_parts(task, truth, streams, devices, tail)
            tables[arm][key] = {"raw":raw, "nonnegative":score_parts(task,truth,clipped,devices,tail)}
            if key=="recorded":
                prior_scores = read(prior.with_name("scores.json"))["scores"][arm]
                if any(not isclose(m["score"],prior_scores[k]["score"],rel_tol=1e-12) for k,m in raw["whole"].items()):
                    raise ValueError("旧主分不符")
    atomic_json(dest/"scores.json", {"issued_sha256":sha256(path.read_bytes()).hexdigest(), "task":task,
        "source":cohort, "regular_windows":len(regular), "window_width":width, "rows":len(service),
        "tail_rows":len(tail), "device_counts":{k:len(v) for k,v in role_units.items()},
        "last_model_ids":model_ids, "gap_stats":{"recorded":lags(recorded),"regular":lags(regular)}, "scores":tables})
    eligible = set(identity["project"]["parent"])
    peer_scores = {}
    if eligible:
        for arm, streams in identity.items():
            clipped = {name:{i:max(0.,v) for i,v in p.items()} for name,p in streams.items()}
            peer_scores[arm] = {k:summary(task,truth,s,devices) for k,s in (("raw",streams),("nonnegative",clipped))}
    atomic_json(dest/"identity-scores.json", {"issued_sha256":sha256(peer_path.read_bytes()).hexdigest(),
        "eligible":len(eligible), "service_rows":len(service), "devices":len({devices[i] for i in eligible}),
        "arms":peer_scores})
    package, checksum, training_batches = official(task)
    fit_ids = {r.sample_id for r in training_batches[0].samples if r.role=="fit"}
    fitted = [r for r in package.train if r.sample_id in fit_ids]
    if len(fitted)!=len(fit_ids):
        raise ValueError("固定训练清单未完全包含在官方训练档案中")
    atomic_json(dest/"train-gaps.json", {"package_sha256":checksum,"clock_width":width,
        "fit_ids_sha256":digest(sorted(fit_ids)), "fit":train_gap_coverage(fitted,width),
        "full_official_train":train_gap_coverage(package.train,width)})
    print(cohort, len(tail), len(service), len(regular),
          {a:{s:[round(v["score"],3) for v in z["raw"]["whole"].values()] for s,z in d.items()} for a,d in tables.items()})


if __name__ == "__main__":
    for cohort in ("bearing","pronostia","engine","phm08"):
        compare(cohort)
