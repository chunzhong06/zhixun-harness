"""同一个源组合修复器前移到探索和确认；按实际采用的权重重新运行十轮。"""
from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256

import numpy as np

from compare_public_selection import external_batches
from compare_group_objectives import ROOT, atomic_json, digest, read, remaining_cycles, summary
from engineering.cli.workflow.public_maintenance import assemble_public, run_prepared
from harness.models.source_combinations import SourceCombinations
from score_engine_setting_reference import DEST
from score_local_defaults import cost


def main(name):
    root = ROOT / "data/official"
    output = DEST / "automatic/decisions" / name
    harness, batches = assemble_public("engine", root, output, "first-proposal", {"mode": "rule"})
    if name == "phm08":
        batches = external_batches("engine", batches)
    originals = batches
    ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    lookup = SourceCombinations([r.values for r in harness.tools.package.train if r.sample_id in ids])
    next_batches, changes = [], {}
    for b in batches:
        known = [r for r in b.samples if r.role != "fit"]
        v = lookup.transform([r.values for r in known])
        rows = {r.sample_id: replace(r, values=tuple(x.tolist())) for r, x in zip(known, v, strict=True)}
        counts = {role: sum(r.values != rows[r.sample_id].values for r in known if r.role == role)
                  for role in ("explore", "adopt", "serve")}
        changes[b.batch_id] = counts
        next_batches.append(replace(b, samples=tuple(rows.get(r.sample_id, r) for r in b.samples)))
    harness.tools.values = {r.sample_id: r.values for b in next_batches for r in b.samples if r.role != "fit"}
    report = run_prepared("engine", output, "first-proposal", 10, {"mode": "rule"}, harness, next_batches)
    preds, groups, actions = {}, {}, []
    for entry, batch in zip(report["rounds"], originals, strict=True):
        stored = read(output / entry["submitted"])["payload"]
        values = stored["predictions"]
        expected = {r.sample_id: r.unit_id for r in batch.samples if r.role == "serve" and r.available_at == batch.ordinal}
        if stored["model_id"] != entry["selected"] or set(values) != set(expected) or preds.keys() & values.keys():
            raise ValueError("新权重服务回执与原始服务批不符")
        preds.update(values)
        groups.update(expected)
        actions.append([entry["batch"], entry["outcome"], entry["selected"]])
    owner = DEST / "automatic" / name
    frozen = owner / "issued.json"
    if sha256(frozen.read_bytes()).hexdigest() != read(owner / "scores.json")["issued_sha256"]:
        raise ValueError("冻结参照身份不同")
    other = read(frozen)["predictions"]
    if name == "engine" and any(abs(preds[k]-other["current-auto"][k]) > 1e-8 for k in preds):
        raise ValueError("官方原始输入相同，服务结果不一致")
    issued = {"decision": preds, "service": other["current-auto"], "original": other["original-auto"],
              "tree": other["tree-auto"], "paper": other["paper-auto"], "old": other["current-raw"]}
    path = output / "comparison.json"
    atomic_json(path, {"arms": issued, "decisions": actions, "changed": changes, "mapping": lookup.identity})
    served = read(path)["arms"]
    truth = {r.sample_id: r.label for r in harness.tools.package.validation} if name == "engine" else remaining_cycles()
    scores = summary("engine", truth, {"parent": served["old"], **served}, groups)
    decision_delta = np.array([preds[i] - other["current-auto"][i] for i in preds])
    atomic_json(output / "scores.json", {"arms": scores, "different_service_predictions": int(np.count_nonzero(abs(decision_delta)>1e-8)),
                "paired_inputs": digest(sorted(preds)), "issued_sha256": sha256(path.read_bytes()).hexdigest(),
                "cost": cost(output, harness.tools.source["package_sha256"], ids)})
    print(name, {k: (round(a["score"], 2), round(a["worst_group"], 2)) for k, a in scores.items()})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("phm08", "engine"))
    main(parser.parse_args().name)
