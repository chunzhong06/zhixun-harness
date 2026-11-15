"""从装机前的同步暴露拟合备用传感器到旧传感器的映射，实际发放只读备用器件。"""
from hashlib import sha256
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from audit_recipe_draws import SEEDS
from engineering.cli.workflow.field_classification import checked, source_experts
from harness.contracts import digest
from harness.models.field_classification import model_scores
from harness.models.site_training import RECIPES, SiteTraining
from harness.runtime.storage import atomic_json
from score_recall_floor import TEMPERATURE, external, read
from score_site_training import OUT, ROOT


PROTOCOL = ROOT / "engineering/configs/experiments/temperature-paired-references.json"
OUTPUT = OUT / "paired"
METHODS = ("identity", "diagonal", "full")
SITES = {
    "a-to-a": ("a-to-a", "a-to-a"),
    "replacement": ("a-to-a", "b-to-b"),
    "b-to-a": ("b-to-b", "a-to-a"),
    "b-to-b": ("b-to-b", "b-to-b"),
}


def references(site):
    panels = []
    for bank in SITES[site]:
        spec = read(TEMPERATURE / bank / "stream.json")
        panels.append(checked(TEMPERATURE / bank / spec["source"], spec["source_sha256"]))
    ids = [row["id"] for row in panels[0]]
    if ids != [row["id"] for row in panels[1]] or len(ids) != len(set(ids)):
        raise ValueError("旧器件和备用器件的参考暴露无法逐一配对")
    dates = np.array([key.split("_")[0] for key in ids])
    if len(np.unique(dates)) != 3 or len(ids) != 300:
        raise ValueError("参考测量和冻结开发日期不符")
    old, spare = (np.array([row["values"] for row in panel]) for panel in panels)
    return old, spare, dates


def fit_map(method, x, y):
    if method == "identity":
        return lambda values: values
    if method == "diagonal":
        dx = x - x.mean(axis=0)
        denom = (dx * dx).sum(axis=0)
        slopes = np.divide((dx * (y - y.mean(axis=0))).sum(axis=0), denom,
                           out=np.zeros_like(denom), where=denom > 0)
        return lambda values: (values - x.mean(axis=0)) * slopes + y.mean(axis=0)
    if method != "full":
        raise ValueError(method)
    incoming = StandardScaler().fit(x)
    original = StandardScaler().fit(y)
    reg = Ridge(alpha=1.).fit(incoming.transform(x), original.transform(y))
    return lambda values: original.inverse_transform(reg.predict(incoming.transform(values)))


def choose_reference(old, spare, dates):
    """整日留出，按恢复旧读数的误差挑选映射；不读取气体标签。"""
    errors = {method: [] for method in METHODS}
    for day in np.unique(dates):
        train = dates != day
        scale = StandardScaler().fit(old[train]).scale_
        for method in METHODS:
            mapping = fit_map(method, spare[train], old[train])
            residual = (mapping(spare[~train]) - old[~train]) / scale
            errors[method].append(float(np.mean(residual ** 2)))
    return min(METHODS, key=lambda key: np.mean(errors[key])), errors


def score_map(site, method, mapping, base, stream, grade, budget, vote, seeds):
    mapped = []
    for keys, values, _, pool, _, _, reveal in stream:
        if keys != pool:
            raise ValueError("此协议要求同一采集批交付后从中返回标签")
        x = mapping(values)
        mapped.append((keys, x, vote(x), reveal))
    scores = []
    for seed in seeds:
        policy = SiteTraining(base.source, base.labels, base.observer.names, base.classes, "recipe-ongoing")
        initial = policy.snapshot()
        issued = []
        for j, (keys, x, votes, reveal) in enumerate(mapped):
            new = policy.predict(x, votes)
            new["source-only"] = model_scores(initial, x, base.classes).argmax(axis=1) + 1
            new["ongoing"] = new[policy.current()]
            ix = sorted(range(len(keys)), key=lambda idx: digest([seed, keys[idx]]))[:budget]
            if j + 1 == len(mapped):
                ix = []
            issued.append({"ids": keys, "recipe": policy.current(), "returned": [keys[i] for i in ix],
                           "predictions": {k: v.tolist() for k, v in new.items()}})
            if ix:
                policy.update(x[ix], {k: v[ix] for k, v in votes.items()}, reveal(ix),
                              {k: new[k][ix] for k in RECIPES})
        path = OUTPUT / site / method / f"seed-{seed}.json"
        atomic_json(path, issued)
        saved = read(path)
        if method == "identity":
            prior = read(OUT / "matched" / site / f"seed-{seed}.json")
            for batch, previous, returned in zip(saved, prior["issued"], prior["returned"], strict=True):
                if (batch["ids"] != previous["ids"] or batch["returned"] != returned or
                    any(batch["predictions"][k] != previous["predictions"][k] for k in (*RECIPES, "ongoing"))):
                    raise ValueError("无映射组未逐条复现原始交付")
        actual = grade()
        ids = [key for batch in saved for key in batch["ids"]]
        if len(ids) != len(set(ids)) or set(ids) != set(actual):
            raise ValueError("采集日评分样本不完整")
        truth = [actual[key] for key in ids]
        table = {}
        for arm in saved[0]["predictions"]:
            p = [v for batch in saved for v in batch["predictions"][arm]]
            recalls = recall_score(truth, p, labels=[1, 2, 3], average=None, zero_division=0)
            table[arm] = {"macro": float(f1_score(truth, p, labels=[1, 2, 3], average="macro", zero_division=0)),
                          "worst": float(min(recalls)), "recall": recalls.tolist()}
        scores.append({"seed": seed, "scores": table, "fits_all_recipes": policy.fits,
                       "receipt_sha256": sha256(path.read_bytes()).hexdigest()})
    means = {a: {k: float(np.mean([r["scores"][a][k] for r in scores])) for k in ("macro", "worst")}
             for a in scores[0]["scores"]}
    return {"mean": means, "runs": scores}


def evaluate(site):
    old, spare, dates = references(site)
    start = perf_counter()
    selected, cv = choose_reference(old, spare, dates)
    mappings = {key: fit_map(key, spare, old) for key in METHODS}
    mappings["sham"] = fit_map("full", spare, np.roll(old, len(old) // 2, axis=0))
    first = dates == min(dates)
    mappings["first-day"] = fit_map("full", spare[first], old[first])
    reference_seconds = perf_counter() - start
    base, stream, grade, budget, _ = external(TEMPERATURE / site)
    if not np.array_equal(base.source, old):
        raise ValueError("固定分类器的源数据不同")
    stream = list(stream)
    vote = source_experts(base.source, base.labels, base.lda)
    results = {}
    for key, mapping in mappings.items():
        seeds = SEEDS if key in ("identity", "full", selected) else (17,)
        results[key] = score_map(site, key, mapping, base, stream, grade, budget, vote, seeds)
    report = {"selected_before_service": selected, "reference_date_errors": cv,
              "all_reference_tools_wall_seconds": reference_seconds, "pairs": len(old),
              "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(), "methods": results}
    atomic_json(OUTPUT / site / "scores.json", report)
    print(site, selected, {k: round(v["mean"]["pooled-source"]["macro"], 5) for k, v in results.items()}, flush=True)
    return report


if __name__ == "__main__":
    atomic_json(OUTPUT / "scores.json", {site: evaluate(site) for site in SITES})
