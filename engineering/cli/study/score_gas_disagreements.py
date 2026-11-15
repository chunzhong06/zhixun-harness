"""冻结分类工具，只在此前同一对判别结果的回标明确支持换法时改判。"""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from itertools import islice

import numpy as np
from scipy.stats import binom

from check_feedback_controls import scores
from gas_complementary_feedback import OUT, ROOT, atomic_json, digest, read
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import majority
from score_observer_errors import datasets


DEST = OUT / "disagreement-transfer"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-disagreement-transfer.json"


class DisagreementChoice:
    """每个计数只来自发放时两个方法确实不同、且至少一个正确的已揭晓样本。"""
    def __init__(self, names, classes):
        self.names = names
        self.classes = classes
        self.counts = Counter()
        self.confusions = {n: np.zeros((classes, classes), dtype=int) for n in ("observer", *names)}
        self.global_choice = "observer"
        self.once = None

    def predict(self, options):
        base = options["observer"]
        answer = base.copy()
        best = np.ones(len(base))
        for name in self.names:
            alt = options[name]
            for a, b in set(zip(base, alt, strict=True)):
                if a == b:
                    continue
                wins = self.counts[name, a, b, True]
                losses = self.counts[name, a, b, False]
                n = wins + losses
                if n < 5:
                    continue
                tail = float(binom.sf(wins - 1, n, .5))
                if tail <= .05 / len(self.names):
                    use = (base == a) & (alt == b) & (tail < best)
                    answer[use], best[use] = b, tail
        return answer

    def update(self, options, truth):
        base = options["observer"]
        for name, pred in options.items():
            np.add.at(self.confusions[name], (truth-1, pred-1), 1)
            if name != "observer":
                for a, b, y in zip(base, pred, truth, strict=True):
                    if a != b and y in (a, b):
                        self.counts[name, a, b, y == b] += 1
        def macro(name):
            c = self.confusions[name]
            d = c.sum(0) + c.sum(1)
            return np.divide(2*c.diagonal(), d, out=np.zeros(len(d)), where=d>0).mean()
        self.global_choice = max(self.confusions, key=macro)
        if self.once is None:
            self.once = self.global_choice


def run(name, stream, truths, budget, classes, provenance):
    fixed_names = tuple(stream[0][1])
    observer = ObserverErrors(fixed_names, classes)
    selector = DisagreementChoice(("vote", *fixed_names), classes)
    evidence = []
    folder = DEST / name
    all_ids = []
    observed = set()
    for step, (ids, votes, pool, feedback_votes, feedback_y) in enumerate(stream, 1):
        def versions(raw):
            return {"observer": observer.predict(raw, "uniform"), "vote": majority(raw, classes), **raw}
        options = versions(votes)
        choices = dict(global_choice=selector.global_choice, once=selector.once or "observer")
        pred = {**options, "pair": selector.predict(options), **{arm: options[v] for arm,v in choices.items()}}
        indices = sorted(range(len(pool)), key=lambda i: digest([17, pool[i]]))[:budget] if step < len(stream) else []
        returned = [pool[i] for i in indices]
        if observed.intersection(returned) or len(set(returned)) != len(returned):
            raise ValueError("回标行重复计费")
        before = {k: np.asarray(v)[indices] for k,v in feedback_votes.items()}
        # 此时回标预测仍使用服务前的计数，封存后才把这组答案传入 update。
        feedback = versions(before) if len(indices) else {}
        path = folder / f"batch-{step:02}.json"
        atomic_json(path, {"ids": list(ids), "choices": choices, "seen_feedback": len(observed),
                           "returned_next": returned, "observer_state": digest(observer.counts.tolist()),
                           "predictions": {k:v.tolist() for k,v in pred.items()},
                           "feedback_predictions": {k:v.tolist() for k,v in feedback.items()}})
        saved = read(path)
        evidence.append(sha256(path.read_bytes()).hexdigest())
        if indices:
            y = np.asarray([feedback_y[i] for i in indices], dtype=int)
            issued = {k: np.asarray(v) for k,v in saved["feedback_predictions"].items()}
            selector.update(issued, y)
            observer.update(y, {k: issued[k] for k in fixed_names})
        observed.update(returned)
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("服务测量不唯一")
    records = [read(folder / f"batch-{n:02}.json") for n in range(1, len(stream)+1)]
    p = {a: np.concatenate([r["predictions"][a] for r in records]) for a in records[0]["predictions"]}
    truth = np.asarray([truths[i] for i in all_ids])
    blocks = np.split(np.arange(len(truth)), np.cumsum([len(r["ids"]) for r in records])[:-1])
    table = {a:scores(truth, v, blocks, classes) for a,v in p.items()}
    for v in table.values():
        v["worst_recall"] = min(x for x in v["recall"] if x is not None)
    fixed = max(("vote", *fixed_names), key=lambda a: table[a]["pooled"])
    changed = p["pair"] != p["observer"]
    strong = max(table[a]["pooled"] for a in (fixed, "observer", "global_choice", "once"))
    accepted = (table["pair"]["pooled"] >= strong + .01 and table["pair"]["worst_recall"] >=
                max(table[a]["worst_recall"] for a in (fixed, "observer")) - .02)
    result = {"protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(), "source": provenance,
              "receipts": evidence, "rows": len(truth), "used_labels": len(observed), "best_fixed": fixed,
              "changes": int(changed.sum()), "rescued": int(np.sum(changed & (p["pair"] == truth))),
              "spoiled": int(np.sum(changed & (p["observer"] == truth))), "adoption_gate": accepted, "arms": table}
    atomic_json(folder / "scores.json", result)
    print(name, result["rows"], result["changes"], result["rescued"], result["spoiled"],
          {k:round(table[k]["pooled"], 5) for k in ("pair","observer",fixed,"global_choice","once")}, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("known", "hardware"))
    args = parser.parse_args()
    if args.source == "known":
        for name, stream, truth, budget, classes in islice(datasets(), 2):
            run(name, stream, truth, budget, classes, {"stream": name, "seed": 17})
    else:
        from sgp40_transfer import streams
        for name, stream, truth, source in streams():
            run(name, stream, truth, 8, 2, source)


if __name__ == "__main__":
    main()
