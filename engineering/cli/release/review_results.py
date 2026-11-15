"""用三个官方包和已保存的运行记录复核模型决定、后续预测与费用，不执行模型。"""
import argparse
import json
import math
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.audit_gas_policy import audit as audit_gas, read
from engineering.cli.study.evaluate_rul_baseline import PACKAGES, describe
from engineering.cli.study.run_rul_maintenance import compare, partition
from harness.contracts import digest
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.methods.planning.rul_continuation import device_biases
from harness.models.model_artifacts import ModelArtifacts


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same_score(left, right):
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-11)


def truth_score(expected, predictions, ids):
    require(set(predictions) == set(ids) and len(ids) == len(set(ids)), "气体评价样本不对应")
    return macro_f1((expected[s], predictions[s]) for s in ids)


class Evidence:
    """仅打开已存在的权重、作业和已提交状态；预测须绑定具体模型及原始测量值。"""

    def __init__(self, folder, state_path, package_sha, rows):
        require((folder / "models").is_dir(), f"缺少模型目录 {folder}")
        self.models = ModelArtifacts(folder / "models")
        self.state = read(state_path)
        require(self.state["evidence_kind"] == "real_model" and self.state["model_experiment"] is True,
                "复核需要真实模型运行")
        self.folder, self.package_sha = folder, package_sha
        self.jobs = {}
        self.outputs = {}
        self.failures = []
        values = {r.sample_id: r.values for r in rows}
        for path in sorted((folder / "jobs").glob("*/state.json")):
            job, packet = read(path), read(path.with_name("input.json"))
            request = packet["request"]
            require(path.parent.name == job["request_id"] == digest(request) and job["nonce"] == packet["nonce"]
                    and request["package_sha256"] == package_sha, "作业回执与输入不同源")
            require(job["evidence_kind"] == "real_model" and job["status"] in ("completed", "failed")
                    and math.isfinite(job["cpu_seconds"]) and job["cpu_seconds"] >= 0, "作业状态或费用无效")
            self.jobs[job["request_id"]] = job
            if job["status"] == "failed":
                self.failures.append(job["request_id"])
            elif request["kind"] == "predict":
                inputs = packet["rows"]
                ids = frozenset(row["sample_id"] for row in inputs)
                require(digest(inputs) == request["inputs_digest"] and len(inputs) == len(ids)
                        and set(job["value"]) == ids, "预测身份或输入摘要不符")
                require(all(set(row) == {"sample_id", "values"}
                            and tuple(row["values"]) == values[row["sample_id"]] for row in inputs),
                        "模型输入与官方测量值不符")
                returned = read(path.with_name("result.json"))
                require(returned["nonce"] == job["nonce"] and returned["status"] == "completed"
                        and returned["value"] == job["value"], "工作进程实际输出与回执不符")
                key = (digest(request["artifact"]), ids)
                previous = self.outputs.setdefault(key, job["value"])
                require(previous == job["value"], "同一输入和模型的两份回执不一致")
        self.cpu = math.fsum(j["cpu_seconds"] for j in self.jobs.values())

    def predictions(self, reference, ids):
        return self.outputs[(digest(reference), frozenset(ids))]

    def weights(self, ref, parent=None):
        manifest, _ = self.models.read(ref)
        request = manifest["request"]
        require(request["package_sha256"] == self.package_sha, "权重来自其他官方包")
        if parent is not None:
            parent_manifest, _ = self.models.read(parent)
            job_path = self.folder / "jobs" / ref["key"]
            trained = read(job_path / "training.json")
            require(request["parent"] == parent and trained["parent_checkpoint_sha256"] == "sha256:" + parent_manifest["sha256"]
                    and trained["checkpoint_sha256"] == "sha256:" + manifest["sha256"]
                    and self.jobs[ref["key"]]["artifact"] == ref and self.jobs[ref["key"]]["status"] == "completed",
                    "候选的父权重、模型文件或训练回执不一致")
        return manifest


def review_gas(root, data, *, public_folder=None):
    folder = public_folder if public_folder is not None else root / "results/experiments/official-gas/pilot-11"
    archive = data / "gas/cpu-a-guide-20261110-v1.zip"
    frozen = read(folder / "protocol.json")
    summary = audit_gas(folder, archive, persist=False)
    require(summary == read(folder / "comparison.json"), "气体逐轮配对与保存的对照结论不一致")
    package = read_official_gas(archive)
    answers = {r.sample_id: r.label for r in package.validation}
    held = frozen["holdout"]
    static_folder = folder / "static" if public_folder is not None else folder.parent / "pilot-1/static"
    original = read(static_folder / "result.json")["holdout"]
    static = Evidence(static_folder, static_folder / "state.json", frozen["package_sha256"], package.validation)
    static_ref = static.state["models"][static.state["current"]]["artifact"]
    require("sha256:" + static.weights(static_ref)["sha256"] == package.checkpoint_sha256
            and static.predictions(static_ref, held) == original["predictions"], "静态留出未对应官方原版的实际输出")
    parent_score = truth_score(answers, original["predictions"], held)
    require(same_score(parent_score, original["macro_f1"]), "气体静态起点评价不一致")
    arms = {}
    for arm in ("sequence-first-window", "sequence-protected"):
        run = folder / arm
        evidence = Evidence(run, run / "state.json", frozen["package_sha256"], package.validation)
        state, result = evidence.state, read(run / "result.json")
        require(result["protocol_digest"] == digest(frozen), "气体回放协议不匹配")
        previous = next(m["model_id"] for m in state["models"].values() if m["parent_id"] is None)
        ref = lambda mid: state["models"][mid]["artifact"]
        require("sha256:" + evidence.weights(ref(previous))["sha256"] == package.checkpoint_sha256, "气体起点权重不符")
        changes = []
        require(len(result["rounds"]) == len(state["completed"]) == frozen["protocol"]["rounds"], "气体回放缺轮")
        for index, row in enumerate(result["rounds"], 1):
            bid = row["round"]
            done = state["completed"][bid]
            decision = done["decision"]
            require(bid == f"round-{index}" and row["incumbent"] == decision["current_model"] == previous
                    and row["selected"] == decision["selected_model"] == done["model_id"]
                    and decision == state["decisions"][bid], "气体模型的轮间接续不一致")
            if row["candidate"] is not None:
                candidate = row["candidate"]["model"]
                weights = evidence.weights(ref(candidate), ref(previous))
                require(row["candidate"]["checkpoint_sha256"] == "sha256:" + weights["sha256"]
                        and row["candidate"]["gas_pair"] == weights["request"]["gas_pair"], "气体报告中的修正或权重不一致")
                require(state["models"][candidate]["parent_id"] == previous, "气体候选父版本错误")
            applied = row["selected"]
            if applied != previous:
                require(row["candidate"]["model"] == applied, "采用版本没有对应候选")
                ids = frozen["rounds"][bid]["adopt"]
                scores = [truth_score(answers, evidence.predictions(ref(m), ids), ids) for m in (previous, applied)]
                require(all(same_score(a,b) for a,b in zip(scores, row["confirmation"], strict=True))
                        and scores[1] - scores[0] > state["config"]["min_improvement"], "采用缺少确认增益")
                changes.append({"round": index, "repair": row["candidate"]["gas_pair"], "confirmation": scores})
            previous = applied
        predictions = evidence.predictions(ref(previous), held)
        require(state["current"] == previous and predictions == result["holdout"]["predictions"], "末版预测与实际产物不符")
        arms[arm] = {"rounds": len(result["rounds"]), "adoptions": changes, "before_updates": next(
            a["before_pooled"] for a in summary["arms"] if a["arm"] == arm),
            "final": truth_score(answers, predictions, held), "worker_cpu_seconds": evidence.cpu}
    return {"metric": "Macro-F1", "original": parent_score, "development_rows": len(held), "arms": arms,
            "different_final_predictions": summary["final_different_predictions"], "before_changes": len(summary["before_changes"])}


def review_life(root, data, name):
    archive_name, checksum, reader = PACKAGES[name]
    archive = data / name / archive_name
    folder = root / f"results/experiments/official-{name}/maintenance-1"
    frozen, result, selection = (read(folder / f) for f in ("protocol.json", "result.json", "selected.json"))
    expect = read(root / "engineering/configs/experiments/official-rul-device-maintenance.json")
    require(frozen["protocol"] == expect and sha256(archive.read_bytes()).hexdigest() == frozen["archive_sha256"] == checksum,
            "寿命官方原包或协议不匹配")
    package = reader(archive)
    rows, devices = partition(package, expect)
    require(devices == frozen["devices"] and frozen["roles"] == {
        k: {"count":len(v), "sample_ids_sha256": digest([r.sample_id for r in v])} for k,v in rows.items()},
        "探索、确认或服务设备清单不一致")
    baseline_path = folder.parent / "baseline" / f"{name}-baseline.json"
    require(sha256(baseline_path.read_bytes()).hexdigest() == expect["baseline_sha"][name], "原版预测文件发生变化")
    baseline = read(baseline_path)
    original = baseline["predictions"]
    evidence = Evidence(folder, folder / "workflow/state.json", checksum, package.validation)
    initial, candidate = result["parent"], result["candidate"]
    require(evidence.weights(initial)["sha256"] == baseline["checkpoint_sha256"] == package.checkpoint_sha, "寿命起点权重不符")
    evidence.weights(candidate, initial)
    done = evidence.state["completed"]["maintenance-1"]
    decision = done["decision"]
    require(result["decision"] == decision == evidence.state["decisions"]["maintenance-1"]
            and done["model_id"] == evidence.state["current"] == decision["selected_model"]
            and result["protocol_digest"] == selection["protocol_digest"] == digest(frozen), "寿命决策或协议不一致")
    require(evidence.state["models"][decision["current_model"]]["artifact"] == initial, "寿命维护起点不是原版")
    ref = evidence.state["models"][done["model_id"]]["artifact"]
    require(ref in (initial, candidate) and result["plan"] == done["trials"], "寿命已生效版本或尝试记录不一致")
    confirmation = rows["adopt"]
    confirmed = compare(package.task, confirmation, original, evidence.predictions(candidate, [r.sample_id for r in confirmation]))
    trials = [t for t in done["trials"] if t.get("model_id")]
    discovery = rows["explore"]
    diagnostics = device_biases(discovery, original)
    issue = next((d for d in diagnostics if d["matching"] * 4 >= 3 * d["recent_count"]), None)
    snapshot = next(e["context"] for e in evidence.state["events"] if e["key"] == "maintenance-1:plan_context:0")
    fact = next(x["value"] for x in snapshot["facts"]["evidence"] if x["kind"] == "performance")
    require(fact["device_bias_issue"] == issue and len(diagnostics) == fact["eligible_devices"]
            and bool(trials) == bool(issue), "已揭晓的设备误差与实际是否试改不一致")
    should_adopt = False
    if trials:
        require(len(trials) == 1 and evidence.state["models"][trials[0]["model_id"]]["artifact"] == candidate,
                "寿命试改与共享候选不同")
        check = compare(package.task, discovery, original, evidence.predictions(candidate, [r.sample_id for r in discovery]))
        should_adopt = (check["candidate"]["rmse"] < check["original"]["rmse"]
                        and confirmed["original"]["rmse"] - confirmed["candidate"]["rmse"] > 1e-12)
    require((ref == candidate) == should_adopt, "寿命换版决定与探索及确认成绩不一致")
    expected_policies = {"static":"original", "fixed":"candidate", "harness":"candidate" if ref == candidate else "original",
                         "fixed_confirmed":"candidate" if confirmed["original"]["rmse"]-confirmed["candidate"]["rmse"] > 1e-12 else "original"}
    require(confirmed == selection["confirmation"] and selection["policies"] == result["policies"] == expected_policies,
            "无法由冻结的确认数据复核策略选择")
    served = rows["serve"]
    outputs = evidence.predictions(candidate, [r.sample_id for r in served])
    paired = compare(package.task, served, original, outputs)
    chosen = original if ref == initial else outputs
    require(outputs == result["candidate_service_predictions"] and paired == result["service_comparison"]
            and describe(package.task, served, chosen) == result["harness_service"]
            and set(done["predictions"]) == {r.sample_id for r in served if r.time_end == 1}
            and all(same_score(v, chosen[sid]) for sid, v in done["predictions"].items()), "寿命服务预测或设备评分不一致")
    trained = len([t for t in done["trials"] if t.get("model_id")])
    require(result["physical_fits"] == len(list((folder / "jobs").glob("*/training.json"))) == 1
            and result["logical_fit_counts"] == {"static":0,"fixed":1,"fixed_confirmed":1,"harness":trained}
            and same_score(result["shared_jobs_cpu_seconds"], evidence.cpu)
            and sorted(evidence.failures) == sorted(j["request_id"] for j in result.get("prior_failed_jobs", [])),
            "寿命候选次数或作业费用不符")
    return {"metric": "RMSE", "unit": package.task.target_unit, "rounds":len(evidence.state["completed"]),
            "harness_candidates": trained, "selected":expected_policies["harness"], "fixed_confirmed":expected_policies["fixed_confirmed"],
            "devices":len(paired["devices"]), "rows":len(served), "original":paired["original"]["rmse"],
            "issue": issue, "confirmation": [confirmed["original"]["rmse"], confirmed["candidate"]["rmse"]],
            "candidate":paired["candidate"]["rmse"], "worker_cpu_seconds":evidence.cpu, "failed_jobs":len(evidence.failures)}


def offline_record(root, name):
    source = ("official-gas/pilot-11/sequence-protected" if name == "gas"
              else f"official-{name}/maintenance-1/workflow")
    state = read(root / "results/experiments" / source / "state.json")
    folder = root / "results/engineering/contest-reproduction/swanlab" / name
    status, proof, records = (read(folder / f) for f in ("recording-status.json", "verification.json", "records.json"))
    require(status["run_fingerprint"] == state["fingerprint"] and status["event_digest"] == digest(state["events"])
            and proof["verified"] is True and proof["model_experiment"] is True and proof["mode"] == "offline"
            and not proof["cloud_verified"] and proof["network_attempts"] == proof["model_import_attempts"] == 0
            and proof["events"] == len(records) and proof["files"], "离线记录的对象或核验回执不一致")
    ids = [r["event_id"] for r in records if r["kind"] not in {"training_epoch", "comparison"}]
    require(ids == [e["key"] for e in state["events"]], "SwanLab 记录与运行事件不对应")
    for f in proof["files"]:
        path = (folder / f["path"].replace("\\", "/")).resolve()
        require(path.is_relative_to(folder.resolve()) and sha256(path.read_bytes()).hexdigest() == f["sha256"],
                "SwanLab 离线文件与核验回执不一致")
    return {"mode":"offline", "sdk":proof["sdk_version"], "cloud_verified":False}


def review(root, data):
    results = {"gas": review_gas(root, data), **{task: review_life(root, data, task) for task in PACKAGES}}
    for name, case in results.items():
        case["swanlab"] = offline_record(root, name)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/official", help="含 gas、bearing、engine 三个官方包的目录")
    parser.add_argument("--json", action="store_true", help="输出完整核对摘要")
    parser.add_argument("--gas-run", type=Path, help="复核公开运行入口的气体任务目录，包含原版、固定和自主三组")
    args = parser.parse_args()
    try:
        if args.gas_run:
            report = review_gas(ROOT, args.data_root, public_folder=args.gas_run)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        report = review(ROOT, args.data_root)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"复核中止：{error}\n")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    gas = report["gas"]
    arm = gas["arms"]["sequence-protected"]
    rounds = "、".join(str(r["round"]) for r in arm["adoptions"])
    print(f"气体　{arm['rounds']} 轮，采用轮次 {rounds}。开发留出的 Macro-F1 从 {gas['original']:.5f} 到 {arm['final']:.5f}。")
    labels = ("乙醇", "乙烯", "氨", "乙醛", "丙酮", "甲苯")
    for step in arm["adoptions"]:
        a, b = step["confirmation"]
        target = labels[step["repair"]["target"] - 1]
        print(f"　　　第 {step['round']} 轮修正{target}漏判，共同确认分数从 {a:.4f} 提高到 {b:.4f}。")
    print(f"　　　同工具固定顺序与逐轮选择的末版预测有 {gas['different_final_predictions']} 条不同。")
    for name, title in (("bearing","轴承"), ("engine","发动机")):
        r = report[name]
        action = "跳过试改" if not r["harness_candidates"] else "试改后确认"
        chosen = "原版" if r["selected"] == "original" else "候选"
        print(f"{title}　{action}，采用{chosen}。{r['devices']} 台服务设备上的 RMSE：原版 {r['original']:.2f}，候选 {r['candidate']:.2f}。")
        if r["issue"]:
            issue = r["issue"]
            a, b = r["confirmation"]
            print(f"　　　{issue['unit']} 的近期偏差触发试改。另一组设备的确认误差从 {a:.2f} 变为 {b:.2f}。")
        else:
            print("　　　探索设备的近期误差没有达到预定的同向比例。")
    print("三个任务均已关联 SwanLab 离线记录。气体十轮为公开开发回放；两项寿命记录各覆盖一次维护决定。")


if __name__ == "__main__":
    main()
