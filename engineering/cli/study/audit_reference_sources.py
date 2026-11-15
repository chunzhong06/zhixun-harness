"""用既有公开预测重放“至少两个来源支持才改判”，并连接当时的真实换版回执。"""
from collections import Counter
from hashlib import file_digest
import json
from math import sqrt
from pathlib import Path
from statistics import fmean
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.data.gas_input import read_gas_batches
from harness.data.official_bearing import read_official_bearing
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.evaluation.round_evidence import round_evidence
from harness.models.gas_views import represent
from harness.runtime.storage import atomic_json

EXP = ROOT / "results/experiments"


def read(path):
    return json.loads(path.read_bytes())


def hash_file(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def indexed(rows, key):
    by_id = {r[key]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("回执出现重复身份")
    return by_id


def replay(rows, answers, regression=False):
    """每行先凭参照的来源数决定是否回到原版，再用同一行答案给三种输出计分。"""
    preds = {name: [] for name in ("original", "candidate", "two_sources")}
    changes, counts = [], Counter()
    for sid, base, candidate, supporting in rows:
        sources = sorted({source for _, source in supporting})
        if candidate != base and not sources:
            raise ValueError(f"改判缺少支持参照：{sid}")
        gated = candidate if len(sources) >= 2 else base
        answer = answers[sid]
        for key, pred in zip(preds, (base, candidate, gated)):
            preds[key].append((answer, pred))
        if base == candidate:
            continue
        delta = (abs(candidate-answer)-abs(base-answer) if regression
                 else int(candidate!=answer)-int(base!=answer))
        outcome = "better" if delta < 0 else "worse" if delta > 0 else "equal_error"
        counts[outcome] += 1
        changes.append({"id": sid, "original": base, "candidate": candidate, "two_sources": gated,
                        "answer": answer, "support": supporting, "sources": sources, "outcome": outcome})
    def metric(pairs):
        return sqrt(fmean((a-p)**2 for a, p in pairs)) if regression else macro_f1(pairs)
    return {"rows": len(rows), "metric": "rmse_minutes" if regression else "macro_f1",
            "scores": {k: metric(v) for k,v in preds.items()}, "candidate_changes": counts,
            "blocked": sum(r["two_sources"] != r["candidate"] for r in changes), "changes": changes}


def gas_audit(inputs):
    path = ROOT / "engineering/configs/experiments/gas-frozen-future.json"
    cfg = read(path)
    official, raw = (ROOT / cfg[k] for k in ("official", "source"))
    if hash_file(official) != cfg["official_sha256"]:
        raise ValueError("气体训练参照来源改变")
    package = read_official_gas(official)
    inputs.extend((path, official, raw))
    train = package.train
    bank = represent([s.values for s in train], "shape", None, None)
    norms = (bank**2).sum(axis=1)
    development = EXP / "official-gas/pilot-8/development.json"
    future = EXP / "official-gas/pilot-7/predictions.jsonl"
    inputs.extend((development, future))
    archived = [json.loads(line) for line in future.read_bytes().splitlines()]
    dev = indexed(read(development), "id")
    result = {}
    for batch in (2, 9, 10):
        details = EXP / "official-gas/pilot-9" / ("development-rows.json" if batch == 2 else f"diagnosis-{batch}-rows.json")
        inputs.append(details)
        old_refs = indexed(read(details), "id")
        if batch == 2:
            expected = dev
            samples = [(s.sample_id, s.values, s.label) for s in package.validation]
        else:
            expected = indexed([r for r in archived if r["batch"] == batch], "line")
            source, = read_gas_batches(raw, expected_sha256=cfg["source_sha256"], batch_number=batch)
            samples = [(s.source_line, s.values, s.label) for s in source.records]
        if set(expected) != {sid for sid, _, _ in samples} or not set(old_refs) <= set(expected):
            raise ValueError("气体预测或参照的身份范围与原批次不符")
        rows, answers = [], {}
        for sid, values, label in samples:
            old = expected[sid]
            base, candidate = old["base"], old["double"]
            supporting = []
            ref = old_refs.get(sid)
            if ref and (ref["whole"]["label"] != candidate or ref["answer"] != label):
                raise ValueError("版本或评价行与已有记录不符")
            # 第9、10批只对部分样本留有五个身份。用原程序的表示和稳定排序还原其余改判。
            if base == 1 and (base != candidate or ref is not None):
                q = represent([values], "shape", None, None)
                order = np.argsort(norms - (2*q @ bank.T)[0], kind="stable")[:5]
                refs = [train[i] for i in order]
                if ref and [s.sample_id for s in refs] != ref["whole"]["neighbors"]:
                    raise ValueError(f"前五参照和历史回执不一致：{sid}")
                supporting = [(s.sample_id,s.group) for s in refs if s.label == candidate]
            rows.append((sid, base, candidate, supporting))
            answers[sid] = label
        result[str(batch)] = replay(rows, answers)
        if batch == 2:
            result["development_holdout"] = replay([r for r in rows if dev[r[0]]["primary"]], answers)
    return result


def bearing_audit(inputs):
    path = ROOT/"engineering/configs/experiments/bearing-prefix-matching.json"
    cfg = read(path)
    archive = next(ROOT/k for k in cfg["inputs"] if k.endswith(".zip"))
    folder = EXP/"official-bearing/prefix-matching"
    stored = folder/"predictions.json"
    inputs.extend((path, archive, stored, folder/"result.json"))
    if hash_file(stored) != read(folder/"result.json")["predictions_sha256"] or hash_file(archive) != cfg["inputs"][archive.relative_to(ROOT).as_posix()]:
        raise ValueError("轴承参照版本已变化")
    pkg = read_official_bearing(archive)
    bank = {s.sample_id:s for s in pkg.train}
    answers = {s.sample_id:s.label for s in pkg.validation}
    scans = {}
    for role, records in read(stored)["replay"].items():
        rows = []
        for r in records:
            p, match = r["predictions"], r["matches"]["trend"]
            supporting = [(k,bank[k].unit_id) for k in match["refs"] if bank[k].label >= cfg["long_minutes"]]
            if len(supporting) != match["long"]:
                raise ValueError("长寿命支持数量与参照身份不一致")
            rows.append((r["id"], p["original"], p["trend"], supporting))
        scans[role] = replay(rows, answers, regression=True)
    return scans


def decisions(inputs):
    summary = {}
    for task, arms, suffix in (("gas", ("sequence-protected","sequence-first-window"), "state.json"),
                               ("bearing", ("harness","fixed_confirmed"), "workflow/state.json")):
        for arm in arms:
            path = EXP/"public-runtime"/task/arm/suffix
            state = read(path)
            inputs.append(path)
            rows = []
            for batch in state["completed"]:
                trace = round_evidence(state, batch)
                info = {"batch": batch, "outcome": trace["outcome"], "confirmation_scores": trace["confirmation"]}
                if trace["confirmation"][1] is not None:
                    info["confirmation_scope"] = trace["confirmation_scope"]
                rows.append(info)
            summary[f"{task}/{arm}"] = rows
    return summary


if __name__ == "__main__":
    inputs = []
    result = {"rule": "在同一次改判中，支持目标输出的训练参照须来自至少两个不同来源，否则保留原预测。",
              "bearing": bearing_audit(inputs), "gas": gas_audit(inputs), "actual_decisions": decisions(inputs)}
    result["inputs_sha256"] = {p.relative_to(ROOT).as_posix():hash_file(p) for p in inputs}
    path = EXP/"reference-sources.json"
    if path.exists() and read(path) != json.loads(json.dumps(result)):
        raise ValueError("现有审计与本次重算不同，请核对输入与规则")
    if not path.exists():
        atomic_json(path, result)
    for task in ("bearing","gas"):
        print(task, {k:(v["rows"],dict(v["candidate_changes"]),v["scores"]) for k,v in result[task].items()})
