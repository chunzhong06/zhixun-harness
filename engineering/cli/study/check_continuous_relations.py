"""从源记录自动找到可在保留资料复现的乘除关系，仅在唯一字段可解释违例时修复。"""
from itertools import combinations

import numpy as np

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, atomic_json, digest, official
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from score_life_training_weights import cohorts


def discover(rows, task):
    keys = sorted(set(r.unit_id if task == "bearing" else r.sample_id for r in rows), key=digest)
    split = set(keys[::2])
    x = np.asarray([r.values for r in rows])
    mask = np.array([(r.unit_id if task == "bearing" else r.sample_id) in split for r in rows])
    take = [i for i in range(x.shape[1]) if min(len(np.unique(x[g, i])) for g in (mask, ~mask)) >= 10]
    rules = []
    first, check = x[mask], x[~mask]
    for i, j in combinations(take, 2):
        probe = first[:8, i] * first[:8, j]
        plausible = np.isclose(probe[:, None], first[:8], rtol=1e-5, atol=1e-9).all(axis=0)
        for k in np.flatnonzero(plausible):
            if k not in take or k in (i,j):
                continue
            if all(np.isclose(v[:, i]*v[:, j], v[:, k], rtol=1e-5, atol=1e-9).all() for v in (first, check)):
                rules.append((i, j, int(k)))
    return rules


def consistency(row, rule):
    i, j, k = rule
    return np.isclose(row[i] * row[j], row[k], rtol=1e-5, atol=1e-9)


def correct(values, rules):
    x = np.asarray(values, dtype=float).copy()
    violated = np.zeros(len(x), int)
    for i, j, k in rules:
        violated += ~np.isclose(x[:, i]*x[:, j], x[:, k], rtol=1e-5, atol=1e-9)
    for idx in np.flatnonzero(violated >= 2):
        row = x[idx]
        bad = [r for r in rules if not consistency(row, r)]
        candidates = []
        for col in set.intersection(*(set(r) for r in bad)):
            proposed = []
            for i, j, k in bad:
                if col == k:
                    proposed.append(row[i] * row[j])
                else:
                    denom = row[j if col == i else i]
                    if denom != 0:
                        proposed.append(row[k] / denom)
            if len(proposed) < 2 or not np.allclose(proposed, proposed[0], rtol=1e-5, atol=1e-9):
                continue
            candidate = row.copy()
            candidate[col] = np.median(proposed)
            if all(consistency(candidate, r) for r in rules):
                candidates.append(candidate)
        if len(candidates) == 1:
            x[idx] = candidates[0]
    return x, violated


def main():
    result = {}
    for task in ("bearing", "gas"):
        pkg, _, batches = official(task)
        training = list(pkg.train)
        rules = discover(training, task)
        trees = frozen(task)[0]["catboost"]
        tables = {}
        if task == "bearing":
            datasets = ((n, x) for n, _, x, *_ in cohorts(task, pkg))
        else:
            archive = read_gas_batches(ROOT / "data/public/gas-drift/gas-drift.zip", expected_sha256=REGISTERED_SHA256)
            services = [r.values for b in batches for r in b.samples if r.role=="serve" and r.available_at==b.ordinal]
            datasets = (("public",services), ("late", [r.values for b in archive if b.batch_number>=3 for r in b.records]))
        for name, values in datasets:
            x = np.asarray(values)
            corrected, bad = correct(x, rules)
            changed = np.any(corrected != x, axis=1)
            report = {"rows": len(x), "any_violation": int(np.count_nonzero(bad)), "corrected": int(changed.sum())}
            if np.any(changed):
                ref, new = (tree_values(trees, task, a) for a in (x, corrected))
                report["tree_changes"] = int(np.count_nonzero(np.any(np.abs(ref-new).reshape(len(x),-1)>1e-8,axis=1)))
            tables[name] = report
            if name in ("bearing", "public"):
                stress = {}
                for col in sorted({k for r in rules for k in r}):
                    disturbed = x.copy()
                    disturbed[:, col] *= 3
                    fixed, _ = correct(disturbed, rules)
                    restored = np.all(np.isclose(fixed,x,rtol=2e-5,atol=1e-9),axis=1)
                    changed_input = np.any(disturbed != x, axis=1)
                    edited = np.any(disturbed != fixed, axis=1)
                    stress[col] = {"restored": int(np.sum(restored & changed_input)), "wrong": int(np.sum(~restored & edited))}
                report["artificial_by_column"] = stress
        result[task] = {"rules": rules, "source_rows": len(training), "cohorts": tables}
        print(task, rules, {k: (v["rows"],v["any_violation"],v["corrected"]) for k,v in tables.items()})
    atomic_json(OUT/"continuous-relations/scores.json", result)


if __name__ == "__main__":
    main()
