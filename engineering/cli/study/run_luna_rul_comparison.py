"""在逐轮公开寿命回放中，分开发布盲测输入、驱动回复、实际预测和最后揭晓的分数。"""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import process_time, sleep
from zipfile import ZipFile

for env_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BASELINE_CPU_THREADS"):
    os.environ[env_var] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.run_rul_maintenance import PACKAGES, PROTOCOL, partition
from engineering.cli.study.run_rul_series import SERIES, windows
from harness.contracts import digest
from harness.methods.planning.rul_continuation import device_biases
from harness.models.engine_manifests import fit_lists
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import LocalStore, atomic_json

DEFAULT = ROOT / "results/experiments/luna-comparison/rul"
DRIVEN = ("project", "mature")
ARMS = (*DRIVEN, "static", "fixed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"冻结记录不一致 {path}")
    else:
        atomic_json(path, value)


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


class Replay:
    """每个进程独占一个任务，四种策略仅通过共同的公开输入相连。"""

    driven_arms = DRIVEN
    record_invalid_actions = False

    def __init__(self, task, root, stack):
        self.task, self.root = task, root
        self.study = root / "issued" / task
        name, checksum, reader = PACKAGES[task]
        self.archive = ROOT / "data/official" / task / name
        if sha256(self.archive.read_bytes()).hexdigest() != checksum:
            raise ValueError("官方包的身份与预定来源不符")
        self.package = reader(self.archive)
        series = read(SERIES)
        roles, assignment = partition(self.package, read(PROTOCOL))
        self.fit = roles["fit"]
        self.feature_names = ["time", *self.package.task.features]
        self.train_columns = self.columns(self.fit)
        self.batches = windows(self.package, assignment, self.fit, series, task)
        self.folder = Path(stack.enter_context(TemporaryDirectory(prefix="rul-luna-")))
        self.zip = stack.enter_context(ZipFile(self.archive))
        self.profile = "B" if task == "bearing" else "C"
        self.trainer = stack.enter_context(_trainer(self.zip, self.folder, self.profile))
        self.study.mkdir(parents=True, exist_ok=True)
        self.original = self.package.checkpoint_sha
        self.model_path(self.original).parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path(self.original).exists():
            self.model_path(self.original).write_bytes(self.package.checkpoint)
        self.extra = fit_lists(self.fit) if task == "engine" else {}
        self.lines = {}
        frozen(root / "protocol" / (task + ".json"), {
            "archive_sha256": checksum, "checkpoint_sha256": self.original, "tasks": list(ARMS),
            "series": series["name"], "series_sha256": sha256(SERIES.read_bytes()).hexdigest(),
            "assignment": assignment, "fit_sha256": digest([r.sample_id for r in self.fit]),
            "rounds": 10, "max_trials": 2, "epochs_per_trial": 1, "max_rows_per_trial": len(self.fit),
            "fixed": "one device-balanced official epoch per round, same nomination and confirmation gates",
            "project": "Luna, existing balanced official continuation and registered engine manifests",
            "mature": "Luna, Self-Healing ML diagnostic subset removal/retention on fixed public fit pool",
            "selection": "one exploration RMSE winner strictly improving the parent; confirm strictly lower RMSE on disjoint devices, then predict service; tie keeps parent",
            "source": "public previously examined trajectories; no cloud; current service targets released only at scoring",
        })

    def model_path(self, key):
        return self.study / "models" / (key + ".npz")

    def state_path(self, arm, number):
        return self.study / arm / f"{number:02d}.json"

    def request_path(self, kind, arm, number):
        return self.root / kind / self.task / arm / f"{number:02d}.json"

    def roles(self, number):
        by_role = defaultdict(list)
        for r in self.batches[number - 1].samples:
            by_role[r.role].append(r)
        return by_role

    def model(self, key):
        path = self.model_path(key)
        if sha256(path.read_bytes()).hexdigest() != key:
            raise ValueError("已发布的父检查点被改写")
        return self.trainer.Model.load(path)

    def predict(self, key, rows):
        output = self.model(key).predict(np.asarray([r.values for r in rows], dtype=float))
        if output.shape != (len(rows),) or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("官方预测输出无效")
        return output

    def errors(self, key, rows):
        return self.predict(key, rows) - np.array([r.label for r in rows])

    def columns(self, rows):
        return np.array([(r.time_end, *r.values) for r in rows], dtype=float).T

    def diagnostics(self, rows, err):
        # 只把当前探索档案与固定训练池按同一阈值切片。排序帮助驱动在输入预算内查到偏差最大的条件。
        fields = []
        for name, fit, visible in zip(self.feature_names, self.train_columns, self.columns(rows), strict=True):
            splits = []
            for cutoff in sorted(set(float(x) for x in np.quantile(fit, [.25, .5, .75]))):
                left, fit_left = visible <= cutoff, fit <= cutoff
                if fit_left.all() or not fit_left.any():
                    continue
                splits.append({"threshold": cutoff, "fit_le": int(fit_left.sum()), "explore_le": int(left.sum()),
                               "bias_le": round(float(err[left].mean()), 2) if left.any() else None,
                               "bias_gt": round(float(err[~left].mean()), 2) if (~left).any() else None})
            if splits:
                contrast = max((abs(s["bias_le"] - s["bias_gt"]) for s in splits
                                if s["bias_le"] is not None and s["bias_gt"] is not None
                                and min(s["explore_le"], len(rows)-s["explore_le"]) >= 3), default=0)
                fields.append((name, contrast, splits))
        fields.sort(key=lambda f: (f[0] != "time", -f[1], f[0]))
        return {name: thresholds for name, _, thresholds in fields[:4]}

    def request(self, arm, number, parent, prior):
        rows = self.roles(number)["explore"]
        err = self.errors(parent, rows)
        preds = dict(zip((r.sample_id for r in rows), (float(r.label + e) for r, e in zip(rows, err)), strict=True))
        units = device_biases(rows, preds)
        kinds = ["balanced", *self.extra] if arm == "project" else ["balanced", "include", "exclude"]
        info = {"task": self.task, "arm": arm, "round": number, "parent_sha256": parent,
                "target_unit": self.package.task.target_unit, "explore_rows": len(rows),
                "devices": len(units), "rmse": round(rmse(err), 4), "mean_signed_error": round(float(err.mean()), 3),
                "bias_definition": "predicted minus true; positive means overestimation",
                "most_biased_devices": [{k: (round(v, 2) if isinstance(v, float) else v) for k,v in u.items()}
                                        for u in units[:3]], "feature_splits": self.diagnostics(rows, err),
                "previous": prior, "kinds": kinds, "budget": {"max_actions": 2, "epochs": 1, "max_rows": len(self.fit)},
                "action_format": {"observation_key": "copy input key", "actions": [{"kind": "one of kinds"}],
                                  "reason": "short diagnosis with evidence; [] actions means keep"},
                "rule": "All candidates continue the current parent. Lowest exploration RMSE gets one disjoint confirmation. Strict improvement required on both. No service scores are revealed."}
        if arm == "project" and self.extra:
            info["engine_manifests"] = "balanced=8 points/device; device-balanced=7 fixed points/device; cycle-window=7 points/device with targeted substitution in cycles 81-161."
        if arm == "mature":
            info["subset_rule"] = "include/exclude require field from feature_splits, op=le or gt, threshold copied exactly from that field. include trains matching rows; exclude trains the remainder. fit_le gives count <= threshold. balanced uses full available fit pool."
        info["observation_key"] = digest(info)
        frozen(self.request_path("requests", arm, number), info)
        return info

    def training_rows(self, action, request):
        kind = action["kind"]
        if kind not in request["kinds"]:
            raise ValueError(f"未提供的训练动作 {kind}")
        if kind == "balanced":
            return self.fit
        if kind in self.extra:
            return self.extra[kind]
        field, op, threshold = action["field"], action["op"], action["threshold"]
        if (op not in ("le", "gt") or type(threshold) not in (float, int)
                or threshold not in [s["threshold"] for s in request["feature_splits"][field]]):
            raise ValueError("训练条件须取自本轮可见条件")
        values = self.train_columns[self.feature_names.index(field)]
        selected = values <= threshold if op == "le" else values > threshold
        if kind == "exclude":
            selected = ~selected
        return [row for row, hit in zip(self.fit, selected, strict=True) if hit]

    def file_lines(self, split, rows, path):
        # 读取一次原官方文件，保留被选行的字节与原顺序，所有训练清单均来自同一公开训练池。
        if split not in self.lines:
            allowed = ({r.sample_id for r in self.fit} if split == "train" else
                       {r.sample_id for b in self.batches for r in b.samples if r.role == "explore"})
            found = {}
            with self.zip.open(f"public/{split}.jsonl") as source:
                for line in source:
                    sid = json.loads(line)["id"]
                    if sid in allowed:
                        found[sid] = line
            if set(found) != allowed:
                raise ValueError("官方原始行不齐")
            self.lines[split] = found
        selected = {r.sample_id for r in rows}
        if len(selected) != len(rows) or not selected.issubset(self.lines[split]):
            raise ValueError("请求使用了冻结范围以外的行")
        path.write_bytes(b"".join(raw for sid, raw in self.lines[split].items() if sid in selected))

    def train(self, arm, number, index, action, rows, parent, explore):
        job = self.study / arm / "jobs" / f"{number:02d}-{index}"
        identity = {"arm": arm, "round": number, "index": index, "action": action, "parent": parent,
                    "fit_sha256": digest(sorted(r.sample_id for r in rows)), "rows": len(rows),
                    "explore_sha256": digest(sorted(r.sample_id for r in explore))}
        receipt = job / "receipt.json"
        if receipt.exists():
            saved = read(receipt)
            if saved["identity"] != identity or sha256(self.model_path(saved["candidate"]).read_bytes()).hexdigest() != saved["candidate"]:
                raise ValueError("重试训练与已完成候选的身份不匹配")
            return saved
        with TemporaryDirectory(dir=self.folder) as tmp:
            work = Path(tmp)
            train, validation = work / "train.jsonl", work / "validation.jsonl"
            self.file_lines("train", rows, train)
            self.file_lines("validation", explore, validation)
            start = process_time()
            record = self.trainer.train_run(str(self.folder / "baselines" / "configs" / f"{self.profile}.json"),
                                            train, validation, work / "out", resume=self.model_path(parent),
                                            epochs=1, stage_id=f"round-{number}")
            cpu = process_time() - start
            contents = (work / "out/checkpoint.npz").read_bytes()
            candidate = sha256(contents).hexdigest()
            if (record["parent_checkpoint_sha256"] != "sha256:" + parent or
                    record["checkpoint_sha256"] != "sha256:" + candidate or
                    record["final_epochs"] != record["initial_epochs"] + 1 or record["train_rows"] != len(rows)):
                raise ValueError("官方训练报告与请求不符")
            checkpoint = self.model_path(candidate)
            if not checkpoint.exists():
                # 模型在输出所在盘上原子发布，两台设备的目录互不共享。
                staged = checkpoint.with_suffix(f".{os.getpid()}.writing")
                staged.write_bytes(contents)
                os.replace(staged, checkpoint)
            saved = {"identity": identity, "candidate": candidate, "training_cpu_seconds": cpu,
                     "official": record}
            atomic_json(receipt, saved)
            return saved

    def step(self, arm, *, publish_only=False):
        number = next((n for n in range(1, 11) if not self.state_path(arm, n).exists()), None)
        if number is None:
            return False
        previous = read(self.state_path(arm, number - 1)) if number > 1 else None
        parent = previous["selected"] if previous else self.original
        roles = self.roles(number)
        request = {"kinds": ["balanced"]}
        decision = {"actions": [{"kind": "balanced"}] if arm == "fixed" else [], "reason": "precommitted control"}
        if arm in self.driven_arms:
            brief = {k: previous[k] for k in ("round", "trials", "adopted", "nomination", "confirmation")} if previous else None
            request = self.request(arm, number, parent, brief)
            response = self.request_path("responses", arm, number)
            if publish_only or not response.exists():
                return False
            decision = read(response)
            if decision.get("observation_key") != request["observation_key"]:
                raise ValueError(f"{response} 中的 observation_key 无效")
        elif publish_only:
            return False
        actions = decision["actions"]
        if not isinstance(actions, list) or len(actions) > 2:
            raise ValueError("本轮最多两个候选")
        if len(actions) == 1 and actions[0].get("kind") == "keep":
            actions = []
        # 驱动可能提交逻辑互补却选中同一批训练行。只执行一次，并在回执明确标识等价动作。
        unique = {}
        invalid = []
        for action in actions:
            try:
                rows = self.training_rows(action, request)
                ids = tuple(sorted(r.sample_id for r in rows))
                if not ids or len(ids) > len(self.fit):
                    raise ValueError("训练子集为空或超过共同样本上限")
            except (KeyError, ValueError, TypeError) as error:
                if not self.record_invalid_actions:
                    raise
                invalid.append({"action":action,"error":str(error)})
                continue
            unique.setdefault(ids, (action, rows))
        trials = []
        elapsed = process_time()
        parent_explore = rmse(self.errors(parent, roles["explore"]))
        for index, (action, rows) in enumerate(unique.values(), 1):
            saved = self.train(arm, number, index, action, rows, parent, roles["explore"])
            score = rmse(self.errors(saved["candidate"], roles["explore"]))
            trials.append({"action": action, "candidate": saved["candidate"], "rows": len(rows),
                           "explore_rmse": score, "training_cpu_seconds": saved["training_cpu_seconds"]})
        best = min(trials, key=lambda t:t["explore_rmse"], default=None)
        nominated = best["candidate"] if best and best["explore_rmse"] < parent_explore - 1e-12 else None
        selected, confirmation = parent, None
        if nominated:
            old, new = (rmse(self.errors(k, roles["adopt"])) for k in (parent, nominated))
            confirmation = {"parent_rmse": old, "candidate_rmse": new}
            if new < old - 1e-12:
                selected = nominated
        service = roles["serve"]
        if not service or any(r.label is not None for r in service):
            raise ValueError("服务输入应来自本轮且隐藏答案")
        predictions = self.predict(selected, service)
        issued = dict(zip((r.sample_id for r in service), predictions.tolist(), strict=True))
        snapshot = {"task": self.task, "arm": arm, "round": number, "parent": parent, "selected": selected,
                    "adopted": selected != parent, "request_key": request.get("observation_key"),
                    "decision": decision, "parent_explore_rmse": parent_explore, "trials": trials,
                    "nomination": nominated, "confirmation": confirmation, "predictions": issued,
                    "round_cpu_seconds_including_training": process_time() - elapsed}
        if invalid:
            snapshot["invalid_actions"] = invalid
        frozen(self.state_path(arm, number), snapshot)
        print(self.task, arm, number, len(trials), "adopt" if selected != parent else "keep", flush=True)
        return True

    def complete(self):
        return all(self.state_path(arm, 10).exists() for arm in ARMS)

    def score(self):
        """四种策略全部封存后才拿出服务答案；同一设备的十轮测量连续计入其整条轨迹。"""
        if not self.complete():
            raise ValueError("仍有未提交的预测")
        rounds = {arm: [read(self.state_path(arm, n)) for n in range(1, 11)] for arm in ARMS}
        targets = {r.sample_id:r for r in self.package.validation}
        results, devices = {}, {}
        for arm, states in rounds.items():
            residuals = defaultdict(list)
            scores = []
            prior = self.original
            seen = set()
            for st in states:
                rows = self.roles(st["round"])["serve"]
                issued = st["predictions"]
                if set(issued) != {r.sample_id for r in rows} or seen & issued.keys() or st["parent"] != prior:
                    raise ValueError("逐轮评估有跳版或预测编号错误")
                prior = st["selected"]
                seen.update(issued)
                current = []
                for sid, pred in issued.items():
                    row = targets[sid]
                    error = pred - row.label
                    current.append(error)
                    residuals[row.unit_id].append(error)
                scores.append(rmse(current))
            per_device = {u:rmse(err) for u, err in residuals.items()}
            devices[arm] = per_device
            errors = [e for err in residuals.values() for e in err]
            results[arm] = {"rmse": rmse(errors), "mae": float(np.mean(np.abs(errors))),
                            "service_rows": len(seen), "devices": len(per_device),
                            "device_equal_mean_rmse": float(np.mean(list(per_device.values()))),
                            "worst_device_rmse": max(per_device.values()), "round_rmse": scores,
                            "worst_round_rmse": max(scores), "final_checkpoint": prior,
                            "logical_proposals": 10 if arm in DRIVEN else 0,
                            "candidate_fits": sum(len(s["trials"]) for s in states),
                            "fit_rows_total": sum(t["rows"] for s in states for t in s["trials"]),
                            "adoptions": sum(s["adopted"] for s in states),
                            "training_cpu_seconds": sum(t["training_cpu_seconds"] for s in states for t in s["trials"]),
                            "round_cpu_seconds_including_training": sum(s["round_cpu_seconds_including_training"] for s in states)}
        paired = {u:{a:devices[a][u] for a in ARMS} for u in devices["static"]}
        for a in ARMS:
            gains = [by["static"] - by[a] for by in paired.values()]
            results[a]["devices_vs_static"] = {"better":sum(x>1e-9 for x in gains), "tied":sum(abs(x)<=1e-9 for x in gains),
                                               "worse":sum(x< -1e-9 for x in gains)}
        score = self.root / "scores" / (self.task + ".json")
        frozen(score, {"task":self.task, "unit":self.package.task.target_unit, "arms":results,
                       "per_device":paired, "source":"previously used public trajectories"})
        if self.task == "engine":
            self.endpoints({a:x["final_checkpoint"] for a,x in results.items()})
        return score

    def endpoints(self, checkpoints):
        from engineering.cli.study.engine_endpoint_check import RAW, PREFLIGHT, SUBSETS, read_cmapss, test_answers
        from harness.data.cmapss_input import REGISTERED_SHA256
        preflight = read(PREFLIGHT)
        if (sha256(RAW.read_bytes()).hexdigest() != REGISTERED_SHA256 or preflight["raw_archive"] != REGISTERED_SHA256
                or preflight["test_prefixes_matching_train"] or any(preflight["test_endpoints_matching_official_rows_by_subset"].values())):
            raise ValueError("终点轨迹与已核对的外部输入不一致")
        tracks = {s: read_cmapss(RAW, subset=s, partition="test") for s in SUBSETS}
        flat = [t for group in tracks.values() for t in group]
        if len(flat) != 707 or len({t.unit_id for t in flat}) != len(flat):
            raise ValueError("NASA 终点设备清单不符")
        values = np.asarray([(*t.values[-1][3:], *t.values[-1][:3]) for t in flat])
        issued = {arm:dict(zip((t.unit_id for t in flat), self.model(key).predict(values).tolist(), strict=True))
                  for arm,key in checkpoints.items()}
        frozen(self.study / "endpoints.json", {"checkpoints":checkpoints, "predictions":issued})
        answers = {s:test_answers(RAW, s, ts) for s, ts in tracks.items()}
        total = {uid:v for group in answers.values() for uid,v in group.items()}
        result = {arm:{s:rmse([p[uid]-v for uid,v in group.items()]) for s,group in {**answers, "pooled":total}.items()}
                  for arm,p in issued.items()}
        frozen(self.root / "scores" / "engine-nasa.json", {"devices":len(flat), "rmse":result,
               "source":"previously inspected NASA public device endpoints, frozen final models"})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "advance", "score", "watch"))
    parser.add_argument("--task", required=True, choices=tuple(PACKAGES))
    parser.add_argument("--output", default=DEFAULT, type=Path)
    args = parser.parse_args()
    with ExitStack() as stack:
        stack.enter_context(LocalStore(args.output / "issued" / args.task).exclusive())
        replay = Replay(args.task, args.output, stack)
        if args.mode == "score":
            print(replay.score(), flush=True)
            return
        while True:
            moved = False
            for arm in ARMS:
                moved = replay.step(arm, publish_only=args.mode == "prepare") or moved
            if args.mode == "prepare" or (not moved and args.mode == "advance"):
                return
            if replay.complete():
                print(replay.score(), flush=True)
                return
            if not moved:
                sleep(1)


if __name__ == "__main__":
    main()

