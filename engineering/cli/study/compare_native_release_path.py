"""固定训练和回标预算，用已存权重分离发布与训练接续，再按同一交付校准算法计分。"""
from hashlib import sha256
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from compare_group_objectives import _trainer
from gas_complementary_feedback import OUT, ROOT, atomic_json, digest, read
from harness.models.observer_errors import ObserverErrors
from harness.runtime.handoff import LocalAcceptance
from score_feedback_refit import sources
from score_whole_maintenance import integrated_predictions


PROTOCOL = ROOT / "engineering/configs/experiments/gas-native-release-path.json"
DEST = OUT / "native-release-path"
HISTORY = OUT / "native-feedback"
WHOLE = OUT.parent / "whole-maintenance/gas"
POLICIES = ("original", "mixed", "confirmed", "detached", "detached-tie")


def identity(path):
    return sha256(path.read_bytes()).hexdigest()


def predict(model, features):
    return model.predict(features).argmax(axis=1) + 1


def measure(trainer, root_weight, study):
    name, _, _, rows, stream, truth, budget, classes = study
    original = HISTORY / name
    report = read(original / "scores.json")
    root_hash = identity(root_weight)
    if root_hash != report["original_sha256"]:
        raise ValueError("原包起点变更")
    old_suite = WHOLE / ("public" if name == "public" else "external")
    shared = read(old_suite / "scores.json")
    weight_paths = {root_hash: root_weight}
    models = {}

    def model(key):
        if key not in models:
            if identity(weight_paths[key]) != key:
                raise ValueError("模型与当轮交付不符")
            models[key] = trainer.Model.load(weight_paths[key])
        return models[key]

    arms = {k: root_hash for k in POLICIES}
    bank_names = tuple(stream[0][1])
    observers = {k: ObserverErrors(bank_names, classes) for k in (*POLICIES, "recommended")}
    endpoint = LocalAcceptance()
    endpoint.output = DEST / name / "accepted"
    protocol = identity(PROTOCOL)
    steps = []
    pending = []
    for j, (ids, votes, pool, feedback_votes, pool_y) in enumerate(stream):
        old_path = original / f"batch-{j+1:02}.json"
        if identity(old_path) != report["receipts"][j]:
            raise ValueError("历史交付身份不符")
        old = read(old_path)
        if list(ids) != old["ids"]:
            raise ValueError("历史验证样本不一致")
        current_x = np.asarray([rows[k] for k in ids])
        event = {"round": j + 1, "weights_before": arms.copy()}
        predictions = {}
        if j:
            update = old["event"]["updates"]
            confirm_ids = [k for k, _ in pending]
            labels = np.array([y for _, y in pending])
            if confirm_ids != update["confirmed"]["confirm_ids"]:
                raise ValueError("可用确认标签与冻结试验不一致")
            confirm_x = np.asarray([rows[k] for k in confirm_ids])
            for k in ("mixed", "confirmed"):
                entry = update[k]
                if entry["parent_sha256"] != arms[k]:
                    raise ValueError("接续父版本不符")
                weight_paths[entry["candidate_sha256"]] = original / "weights" / f"{k}-{j:02}.npz"
            native = update["confirmed"]
            event["native_decision"] = {k: native[k] for k in ("previous_correct", "candidate_correct", "deployed")}
            prior_weight = arms["confirmed"]
            candidate = native["candidate_sha256"]
            old_hits = int(np.sum(predict(model(prior_weight), confirm_x) == labels))
            new_hits = int(np.sum(predict(model(candidate), confirm_x) == labels))
            if (old_hits, new_hits, new_hits > old_hits) != (native["previous_correct"], native["candidate_correct"], native["deployed"]):
                raise ValueError("以旧模型和合法确认数据未复现采用决定")
            predictions["previous-local"] = predict(model(prior_weight), current_x)
            predictions["candidate-local"] = predict(model(candidate), current_x)
            arms.update(mixed=old["weights"]["mixed"], confirmed=old["weights"]["confirmed"])
            # 同一条训练轨迹可继续改进，发布判断仍只使用最近一次留出的回标。
            next_hash = arms["mixed"]
            candidate_hits = int(np.sum(predict(model(next_hash), confirm_x) == labels))
            for arm in ("detached", "detached-tie"):
                old_hits = int(np.sum(predict(model(arms[arm]), confirm_x) == labels))
                accept = candidate_hits > old_hits or (arm == "detached-tie" and candidate_hits == old_hits)
                if accept:
                    arms[arm] = next_hash
                event[arm] = {"previous_correct": old_hits, "candidate_correct": candidate_hits, "accepted": accept}
        for k, observer in observers.items():
            v = votes if k == "recommended" else {**votes, "current": predict(model(arms[k]), current_x)}
            if k in old["predictions"] and v["current"].tolist() != old["predictions"][k]:
                raise ValueError(f"未复现 {k} 的原模型服务输出")
            predictions["raw/" + k] = v["current"]
            predictions["service/" + k] = observer.predict(v, "uniform")
        # 独立复查共享推荐校准的每一条输出。旧论文适配只作已有参照，不重新生成提案。
        reference = read(old_suite / f"round-{j+1:02}.json")["predictions"]
        if predictions["service/recommended"].tolist() != [reference["project"][k] for k in ids]:
            raise ValueError("推荐校准交付与实测对照不符")
        predictions["service/paper-adapted"] = np.array([reference["paper-adapted"][k] for k in ids])
        predictions["service/old-fixed"] = np.array([reference["fixed"][k] for k in ids])
        ix = sorted(range(len(pool)), key=lambda i: digest([17, pool[i]]))[:budget] if j + 1 < len(stream) else []
        sampled_ids = [pool[i] for i in ix]
        event.update(weights_after=arms.copy(), returned_after=sampled_ids)
        payload = {"ids": list(ids), "decision": event, "predictions": {k: v.tolist() for k, v in predictions.items()}}
        request = digest([protocol, name, payload])
        receipt = endpoint.submit(request, payload)
        if not receipt["accepted"] or receipt["request_id"] != request:
            raise ValueError("接收端未确认全部候选输出")
        steps.append(request)
        if ix:
            feedback_x = np.asarray([rows[k] for k in sampled_ids])
            sampled = {k: np.asarray(v)[ix] for k, v in feedback_votes.items()}
            visible_y = [pool_y[i] for i in ix]
            for k, observer in observers.items():
                v = sampled if k == "recommended" else {**sampled, "current": predict(model(arms[k]), feedback_x)}
                observer.update(visible_y, v)
            pending = list(zip(sampled_ids, visible_y, strict=True))[len(ix)//2:]

    receipts = [read(endpoint.output / (key + ".json"))["payload"] for key in steps]
    keys = [k for r in receipts for k in r["ids"]]
    if len(set(keys)) != len(keys) or any(digest([protocol, name, r]) != key for r, key in zip(receipts, steps, strict=True)):
        raise ValueError("已发预测无法核对")
    y = np.asarray([truth[k] for k in keys])
    blocks = np.split(np.arange(len(y)), np.cumsum([len(r["ids"]) for r in receipts])[:-1])
    table = {}
    vectors = {}
    for arm in receipts[0]["predictions"]:
        vectors[arm] = np.concatenate([r["predictions"][arm] for r in receipts])
        table[arm] = scores(y, vectors[arm], blocks, classes)
    for k in ("original", "mixed", "confirmed"):
        if abs(table["raw/" + k]["pooled"] - report["arms"][k]["pooled"]) > 1e-12:
            raise ValueError(f"{k} 的逐条再评价未复现原分数")
    for arm, key in (("recommended", "project"), ("paper-adapted", "paper-adapted"), ("old-fixed", "fixed")):
        if abs(table["service/" + arm]["pooled"] - shared["arms"][key]["score"]) > 1e-12:
            raise ValueError("共同交付工具的参照分数未复现")
    if name == "public" and dict(zip(keys, vectors["service/recommended"].tolist(), strict=True)) != integrated_predictions("gas"):
        raise ValueError("未复现队友实际总入口对全部官方行的交付")
    rejected = []
    for ix, r in zip(blocks[1:], receipts[1:], strict=True):
        p = r["predictions"]
        event = r["decision"]
        if not event["native_decision"]["deployed"]:
            correct = y[ix]
            before, after = (np.asarray(p[a]) == correct for a in ("previous-local", "candidate-local"))
            rejected.append({"round": event["round"], "validation_delta": event["native_decision"]["candidate_correct"] -
                             event["native_decision"]["previous_correct"], "rows": len(ix),
                             "rescued_on_next": int(np.sum(after & ~before)), "spoiled_on_next": int(np.sum(before & ~after))})
    controls = ["service/" + k for k in ("recommended", "original", "mixed", "old-fixed", "paper-adapted")]
    def passes(arm):
        v = table["service/" + arm]
        return all(v["pooled"] >= table[a]["pooled"] + .01 and min(v["recall"]) >= min(table[a]["recall"]) - .02
                   for a in controls)
    summary = {"protocol_sha256": protocol, "requests": steps, "scores": table, "rejections": rejected,
               "passed": {k: passes(k) for k in ("detached", "detached-tie")}, "labels_used": sum(len(r["decision"]["returned_after"]) for r in receipts),
               "original_training_steps": report["updates"], "additional_training_steps": 0,
               "different_from_recommended": {k: int(np.sum(v != vectors["service/recommended"])) for k, v in vectors.items()}}
    atomic_json(DEST / name / "scores.json", summary)
    print(name, {k: round(v["pooled"], 5) for k, v in table.items()}, rejected, flush=True)


if __name__ == "__main__":
    archive, = (ROOT / "data/official/gas").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "A") as trainer:
        parent = Path(tmp) / "root.npz"
        parent.write_bytes(z.read("checkpoint.npz"))
        for study in islice(sources(), 2):
            measure(trainer, parent, study)
