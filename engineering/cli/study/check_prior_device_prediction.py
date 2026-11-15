"""只读两套既有交接，以之前已经到达的同机测量做配对诊断，不训练、不读取未来批长。"""
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from compare_group_objectives import (
    BASE, LocalStore, atomic_json, delivered, digest, official, read,
    reference_folders, remaining_cycles, remaining_minutes, summary,
)
from compare_proposal_frequency import TASKS

OUT = BASE / "prior-device"


def issue(batches, raw):
    # 快照在整批输出发出后更新；同一批无可用的先后顺序。
    prior = {}
    blended, propagated = {}, {}
    affected = 0
    for rows in batches:
        updates = {}
        for i, unit, time in rows:
            value = raw[i]
            blended[i] = propagated[i] = value
            if unit in prior:
                prev_time, prev = prior[unit]
                if prev_time < time:
                    blended[i] = .5 * (prev + value)
                    propagated[i] = .5 * (prev - (time - prev_time) + value)
                    affected += 1
            recent = updates.get(unit, prior.get(unit))
            if recent is None or recent[0] < time:
                updates[unit] = (time, value)
        prior.update(updates)
    if blended.keys() != raw.keys():
        raise ValueError("已交接预测与服务分片不对应")
    return {"parent": raw, "average": blended, "clock": propagated}, affected


def audit(cohort):
    task = TASKS[cohort]
    fixed, paper = reference_folders(cohort, "mature")
    old = BASE/"selected"/task if cohort == task else fixed.parent/"project"
    samples = read(fixed/"state.json")["samples"]
    batches = [[] for _ in range(10)]
    devices = {}
    for i, row in samples.items():
        if row["role"] == "serve":
            if row["label"] is not None or not row["unit_id"] or row["time_end"] < 1:
                raise ValueError("批次、时钟或无答案交接契约失效")
            devices[i] = row["unit_id"]
            batches[row["available_at"]].append((i, devices[i], row["time_end"]))
    dest = OUT/cohort
    with LocalStore(dest).exclusive():
        issued, counts, elapsed = {}, {}, {}
        for arm, folder in (("project", old), ("paper", paper)):
            report = read(folder/"run.json")["rounds"]
            for rows, event in zip(batches, report, strict=True):
                receipt = read(folder/event["submitted"])["payload"]
                if receipt["predictions"].keys() != {i for i, _, _ in rows}:
                    raise ValueError("参与对照的历史批次输入不同")
            raw = delivered(folder)
            start = perf_counter()
            issued[arm], counts[arm] = issue(batches, raw)
            elapsed[arm] = perf_counter() - start
        record = {"batch_input_digest": digest(batches), "outputs": issued}
        path = dest/"issued.json"
        if path.exists() and read(path) != record:
            raise ValueError("重跑与封存结果不同")
        atomic_json(path, record)
        if cohort == "pronostia":
            truth = remaining_minutes("full-test-manifest.json")
        elif cohort == "phm08":
            truth = remaining_cycles()
        else:
            package, _, _ = official(task)
            truth = {r.sample_id: r.label for r in package.validation}
        table = {k: summary(task, truth, predictions, devices) for k, predictions in issued.items()}
        clipped = {k: summary(task, truth, {
            method: {i: max(0., value) for i, value in stream.items()}
            for method, stream in predictions.items()}, devices) for k, predictions in issued.items()}
        negative = {k: {method: sum(v < 0 for v in stream.values()) for method, stream in predictions.items()}
                    for k, predictions in issued.items()}
        atomic_json(dest/"scores.json", {"scores": table, "nonnegative_scores": clipped,
                                         "negative_predictions": negative, "with_prior": counts, "seconds_for_both": elapsed,
                                         "issued_sha256": sha256(path.read_bytes()).hexdigest(),
                                         "program_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
        print(cohort, "paired", counts["project"], "of", len(devices),
              {k: [round(row["score"], 3) for row in arm.values()] for k, arm in table.items()})


if __name__ == "__main__":
    for source in ("bearing", "engine", "pronostia", "phm08"):
        audit(source)
