"""恢复同一套三任务交付，核对覆盖查询、模型、每条服务预测和其他两项成绩。"""
import json

from run_gas_served_scale_integrated import (BASE, ROOT, execute, load_issued, read, official, metric,
                                             metrics, infer, read_official_gas, ARCHIVE, prediction_label,
                                             prediction_request, atomic_json)
from harness.models.coverage_scale_gas import MODE

STUDY = BASE / "whole-maintenance/gas-feedback-coverage"
DEST = STUDY / "integrated"
PRIOR = BASE / "whole-maintenance/gas-served-scale/integrated"


def main():
    if (DEST / "scores.json").exists():
        raise ValueError("已完成此入口的验收")
    execute("all", 4, destination=DEST, mode=MODE)
    tasks = ("gas", "bearing", "engine")
    first = {t: load_issued(DEST, t)[0] for t in tasks}
    execute("all", 10, destination=DEST, mode=MODE)
    sent, records, gas = load_issued(DEST, "gas")
    execute("gas", 10, with_options=False, destination=DEST, mode=MODE)
    if load_issued(DEST, "gas")[0] != sent:
        raise ValueError("省略选项后改变已发气体预测")
    result = {}
    for t in tasks:
        files, _, rows = load_issued(DEST, t)
        if len(files) != 10 or files[:4] != first[t]:
            raise ValueError("三任务四轮至十轮接续错误")
        if t != "gas":
            if rows != load_issued(PRIOR, t)[2]:
                raise ValueError("另两项已发预测受影响")
            answers = {r.sample_id: r.label for r in official(t)[0].validation}
            result[t] = {"rows": len(rows), "score": metric(t, answers, rows), "identical": True}
    source = {r.sample_id: r for r in read_official_gas(ARCHIVE).validation}
    folder = STUDY / "seed-17/固定均匀覆盖"
    frozen = read(folder / "run.json")
    run = read(DEST / "gas/run.json")
    for i, (record, key) in enumerate(zip(records, frozen["requests"], strict=True)):
        issued = record["payload"]
        trial = read(folder / "accepted" / (key + ".json"))["payload"]
        if issued["predictions"] != dict(zip(trial["ids"], trial["predictions"], strict=True)):
            raise ValueError("集成气体预测未复现候选筛选")
        expected = frozen["feedback"][i]["ids"] if i < 9 else []
        if list(run["accepted"][i]["returned"]) != expected:
            raise ValueError("实际查询顺序不同")
        ids = sorted(issued["predictions"])
        model = DEST / "gas/snapshots" / (issued["snapshot"] + ".json")
        outputs = infer(model, ROOT / "data/official", (prediction_request(list(source[k].values)) for k in ids))
        if [prediction_label(v) for v in outputs] != [gas[k] for k in ids]:
            raise ValueError("实际单条推理与交付不一致")
    before = load_issued(PRIOR, "gas")[2]
    if set(gas) != set(before) or len(gas) != 414:
        raise ValueError("改前改后服务池不同")
    fixed = sum(gas[k] == source[k].label and before[k] != source[k].label for k in gas)
    spoiled = sum(gas[k] != source[k].label and before[k] == source[k].label for k in gas)
    result["gas"] = {**metrics([source[k].label for k in gas], list(gas.values())),
                     "改对": int(fixed), "改错": int(spoiled), "单条复核": len(gas),
                     "回标": sum(len(e["returned"]) for e in run["accepted"])}
    atomic_json(DEST / "scores.json", result)
    print(json.dumps({"三任务完整交付": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
