"""核对赛方单权重在同一设备上的误差，再用当时已回标的其他轴承决定交付哪份权重。"""
from collections import defaultdict
from hashlib import sha256
from itertools import groupby, permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, tree_values
from compare_group_objectives import ROOT, _trainer, official, read
from harness.models.feedback_choice import HistoricalToolChoice
from harness.models.training_tools import SklearnBackend
from harness.runtime.storage import atomic_json
from run_rul_series import SERIES, windows
from score_completed_choice import femto, xjtu
from score_life_training_weights import cohorts
from score_unseen_xjtu_devices import LOCAL
from score_whole_maintenance import integrated_predictions


DEST = OUT.parent / "whole-maintenance/bearing-native-choices"
PROTOCOL = ROOT / "engineering/configs/experiments/bearing-native-pool.json"
ORDER = ("root", "median", "mean", "readout", "hidden")


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def load_bank(trainer, pkg):
    const = OUT / "constant-weights/bearing"
    readout = OUT / "teacher-readout/bearing"
    hidden = OUT / "tree-hidden/bearing/models"
    constants = read(const / "models.json")
    info = read(readout / "training.json")
    hidden_info = read(hidden / "training.json")
    if {info["parent"], hidden_info["parent"], constants["median"]["source_sha256"],
        constants["mean"]["source_sha256"]} != {pkg.checkpoint_sha}:
        raise ValueError("单权重的原始父版本不一致")
    files = {"root": (readout/"models/original.npz", pkg.checkpoint_sha),
             "readout": (readout/"models/teacher-ridge.npz", info["runs"]["teacher-ridge"]["sha256"]),
             "hidden": (hidden/"whole-teacher.npz", hidden_info["arms"]["whole-teacher"]["sha256"])}
    files.update({k: (const/"models"/(k+".npz"), constants[k]["sha256"]) for k in ("median", "mean")})
    models = {}
    for k, (path, checksum) in files.items():
        if fingerprint(path) != checksum:
            raise ValueError(f"冻结权重发生改变 {k}")
        models[k] = trainer.Model.load(path)
    phase_dir = OUT / "phase/trained/bearing"
    record = read(phase_dir / "receipt.json")
    blob = (phase_dir / "phase.joblib").read_bytes()
    if sha256(blob).hexdigest() != record["sha256"]:
        raise ValueError("阶段树摘要发生改变")
    phase = SklearnBackend(execution_enabled=True).load(blob)

    def forecast(x):
        x = np.asarray(x, dtype=float)
        out = {k: m.predict(x) for k, m in models.items()}
        out["phase"] = np.clip(tree_values(phase, "bearing", x), 0, 1) * record["scale"]
        return out
    return forecast, {**{k: h for k, (_, h) in files.items()}, "phase": record["sha256"]}


def score_output(payload, truth, order=ORDER):
    keys, devices = payload["ids"], np.asarray(payload["devices"])
    if len(keys) != len(set(keys)):
        raise ValueError("同一服务记录重复发放")
    y = np.asarray([truth[k] for k in keys])
    groups = {str(u): np.flatnonzero(devices == u) for u in np.unique(devices)}
    comparator = "current" if "current" in payload["predictions"] else "root"
    parent = np.square(np.asarray(payload["predictions"][comparator]) - y)
    errors = {}
    scores = {}
    for k, pred in payload["predictions"].items():
        delta = np.asarray(pred) - y
        errors[k] = delta**2
        units = {}
        for u, ix in groups.items():
            e = delta[ix]
            units[u] = {"rows": len(ix), "rmse": float(np.sqrt(np.mean(e**2))), "bias": float(np.mean(e)),
                        "within_device_sd": float(np.std(e)), "better": int(np.sum(e**2 < parent[ix]-1e-8)),
                        "worse": int(np.sum(e**2 > parent[ix]+1e-8))}
        scores[k] = {"rmse": float(np.sqrt(errors[k].mean())), "by_device": units,
                     "worst": max(v["rmse"] for v in units.values())}
    best = min(order, key=lambda k: scores[k]["rmse"])
    choices = {u: min(order, key=lambda k: errors[k][ix].mean()) for u, ix in groups.items()}
    oracle = float(np.sqrt(sum(errors[choices[u]][ix].sum() for u, ix in groups.items())/len(y)))
    return {"rows": len(y), "scores": scores, "best_native": best, "oracle_by_device": choices,
            "oracle_rmse": oracle, "scope": "事后每台固定选一份原格式权重，仅用于诊断"}


def publish(folder, payload, answers, order=ORDER):
    path = folder / "issued.json"
    atomic_json(path, payload)
    result = score_output(read(path), answers(), order)
    result["issued_sha256"] = fingerprint(path)
    atomic_json(folder / "scores.json", result)
    return result


def audit(pkg, forecast, weights):
    reports = {}
    for name, keys, x, groups, answers, _ in cohorts("bearing", pkg):
        if name not in ("bearing", "roles", "new-xjtu", "pronostia", "kaist", "ims"):
            continue
        root = OUT / "tree-hidden/bearing" / name
        path = root / "issued.json"
        if fingerprint(path) != read(root / "scores.json")["issued_sha256"]:
            raise ValueError("已有封存发放文件变更")
        old = read(path)["predictions"]
        p = forecast(x)
        for k, archived in (("readout", "teacher-ridge"), ("hidden", "whole-teacher")):
            if not np.allclose(p[k], [old[archived][i] for i in keys], rtol=0, atol=1e-7):
                raise ValueError(f"{name} 未复现具名候选 {archived}")
        p.update({k: np.array([old[k][i] for i in keys]) for k in ("current", "paper") if k in old})
        if name == "bearing":
            actual = integrated_predictions("bearing")
            if set(keys) != actual.keys() or not np.allclose(p["current"], [actual[i] for i in keys], atol=1e-8, rtol=0):
                raise ValueError("原服务设备没有复现推荐版")
        payload = {"ids": keys, "devices": [groups[i] for i in keys], "weights": weights,
                   "predictions": {k: np.asarray(v).tolist() for k, v in p.items()}}
        reports[name] = publish(DEST / "pool" / name, payload, answers)
    return reports


class Select:
    def __init__(self, order):
        self.order = order
        self.historical = HistoricalToolChoice(order, "root")
        self.regret = "root"
        self.rows = defaultdict(list)
        self.first = None

    def update(self, records, forecast):
        for u, values, answers in records:
            p = forecast(values)
            labels = np.asarray(answers)
            self.rows[u].append({k: np.square(p[k] - labels).tolist() for k in self.order})
        losses = [{k: float(np.mean([v for chunk in chunks for v in chunk[k]])) for k in self.order}
                  for chunks in self.rows.values()]
        if losses:
            # 官方协议会再开放同一台的下一段。更新全台的误差，每台在选择时仍只占一份。
            self.historical = HistoricalToolChoice(self.order, "root")
            self.historical.update(losses)
            max_regret = {k: max((r[k]-min(r.values()))/max(r["root"], 1.) for r in losses) for k in self.order}
            best = min(self.order, key=max_regret.get)
            if max_regret[best] < max_regret[self.regret]:
                self.regret = best
            if self.first is None:
                self.first = self.historical.active
        return {"history": self.historical.active, "once": self.first or "root", "regret": self.regret}


def issue(name, rounds, forecast, weights, answers, historical=None, *, order=ORDER, folder=DEST, protocol=PROTOCOL):
    selector = Select(order)
    predictions = defaultdict(list)
    ids, devices, events = [], [], []
    for moment, refs, rows in rounds:
        selected = selector.update([(u, [x for _, x, _ in batch], [y for _, _, y in batch]) for u, batch in refs], forecast)
        keys, owners, x = zip(*rows, strict=True)
        if set(owners) & set(selector.rows):
            raise ValueError("现场反馈和服务设备重叠")
        p = forecast(x)
        for policy, k in selected.items():
            p[policy] = p[k]
        for k, v in p.items():
            predictions[k].extend(v.tolist())
        events.append({"time": moment, "decisions": selected, "feedback": {u: [i for i, _, _ in group] for u, group in refs},
                       "first_id": keys[0], "rows": len(keys)})
        ids.extend(keys)
        devices.extend(owners)
    if historical:
        for k, source in historical.items():
            predictions[k] = [source[i] for i in ids]
    payload = {"ids": ids, "devices": devices, "predictions": dict(predictions), "events": events, "weights": weights,
               "protocol": fingerprint(protocol)}
    return publish(folder / "service" / name, payload, answers, order), payload


def run(pkg, batches, forecast, weights, *, order=ORDER, folder=DEST, protocol=PROTOCOL):
    fit = tuple(r for r in batches[0].samples if r.role == "fit")
    units = sorted({r.unit_id for r in pkg.validation})
    truth = {r.sample_id: r.label for r in pkg.validation}
    archived = read(OUT / "routing/bearing-rotations/issued.json")["streams"]
    expected = {r.role: r.unit_id for b in batches for r in b.samples if r.role != "fit"}
    results, combined, combined_truth = {}, {"ids": [], "devices": [], "predictions": defaultdict(list)}, {}
    for assignment in permutations(units):
        roles = dict(zip(("explore", "adopt", "serve"), assignment, strict=True))
        folds = windows(pkg, {u: role for role, u in roles.items()}, fit, read(SERIES), "bearing")
        rounds = []
        for b in folds:
            service = [(r.sample_id,r.unit_id,r.values) for r in b.samples if r.role == "serve"]
            if service:
                refs = [(u, [(r.sample_id,r.values,r.label) for r in b.samples if r.role == role]) for role, u in roles.items() if role != "serve"]
                rounds.append((b.ordinal, refs, service))
        key = "/".join(assignment)
        historical = {k: archived[key]["predictions"][k] for k in ("current", "fixed")}
        if roles == expected:
            public = read(OUT.parent / "whole-maintenance/bearing/public/scores.json")
            old = OUT.parent / "whole-maintenance/bearing/public"
            paper = {}
            for i, h in enumerate(public["round_sha256"], 1):
                path = old / f"round-{i:02}.json"
                if fingerprint(path) != h:
                    raise ValueError("论文适配历史发放改变")
                paper.update(read(path)["predictions"]["paper-adapted"])
            historical["paper"] = paper
        result, payload = issue("rotation-" + "-".join(assignment), rounds, forecast, weights, lambda: truth, historical,
                                order=order, folder=folder, protocol=protocol)
        if roles == expected:
            results["public"] = result
            if dict(zip(payload["ids"],payload["predictions"]["current"],strict=True)) != integrated_predictions("bearing"):
                raise ValueError("原排列与推荐交付不同")
        for i, unit in zip(payload["ids"], payload["devices"], strict=True):
            combined["ids"].append(key + "/" + i)
            combined["devices"].append(unit)
            combined_truth[key + "/" + i] = truth[i]
        for k, v in payload["predictions"].items():
            if k != "paper":
                combined["predictions"][k].extend(v)
    results["roles"] = publish(folder / "service" / "roles", combined, lambda: combined_truth, order)
    for site, stream in (("xjtu", xjtu(pkg, LOCAL/"fifth-offset6.json")), ("femto", femto(48))):
        failures, reference, service, answers, _ = stream
        rounds = []
        released = set()
        for t, group in groupby(sorted(service, key=lambda r: (r[2],r[0])), key=lambda r: r[2]):
            ready = sorted(u for u, end in failures.items() if end < t and u not in released)
            refs = [(u, [(i,x,y) for who,i,x,y in reference if who == u]) for u in ready]
            if any(len(rows) != 5 for _, rows in refs):
                raise ValueError("独立失效回标的数量不符")
            released.update(ready)
            rounds.append((t, refs, [(i,u,x) for i,u,_,x in group]))
        results[site], _ = issue(site, rounds, forecast, weights, answers, order=order, folder=folder, protocol=protocol)
    return results


if __name__ == "__main__":
    package, _, batches = official("bearing")
    archive, = (ROOT / "data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as work, ZipFile(archive) as z, _trainer(z, Path(work), "B") as trainer:
        bank, identities = load_bank(trainer, package)
        pool = audit(package, bank, identities)
        live = run(package, batches, bank, identities)
    atomic_json(DEST / "scores.json", {"protocol": fingerprint(PROTOCOL), "pool": pool, "service": live})
    for k, r in live.items():
        print(k, {a: round(v["rmse"],2) for a,v in r["scores"].items()}, flush=True)
