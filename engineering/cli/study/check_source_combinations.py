"""逐列检验从源数据学习的纠错是否会改坏干净测量，以及是否能恢复注入的单列错误。"""
import numpy as np

from compare_group_objectives import atomic_json, official
from harness.models.source_combinations import SourceCombinations
from score_engine_setting_reference import DEST


def main():
    report = {}
    for task in ("gas", "bearing", "engine"):
        _, _, batches = official(task)
        source = {r.sample_id: r.values for r in batches[0].samples if r.role == "fit"}
        services = {r.sample_id: r.values for b in batches for r in b.samples if r.role == "serve" and r.available_at == b.ordinal}
        fit = np.array(list(source.values()))
        clean = np.array(list(services.values()))
        lookup = SourceCombinations(fit)
        clean_changes = np.any(lookup.transform(clean) != clean, axis=1)
        stress = {}
        if task == "engine":
            for j in lookup.columns:
                corrupt = clean.copy()
                corrupt[:, j] = fit[:, j].max() + np.ptp(fit[:, j])
                restored = lookup.transform(corrupt)
                correct = np.all(restored == clean, axis=1)
                untouched = np.all(restored == corrupt, axis=1)
                stress[j] = {"correct": int(correct.sum()), "held": int(untouched.sum()),
                             "wrong": int(np.count_nonzero(~correct & ~untouched))}
        report[task] = {"source_rows": len(fit), "clean_rows": len(clean), "columns": lookup.columns,
                        "combinations": len(lookup.counts), "clean_changed": int(clean_changes.sum()),
                        "mapping_sha256": lookup.identity, "out_of_range_stress": stress}
    atomic_json(DEST / "automatic/clean-and-stress.json", report)
    print(report)


if __name__ == "__main__":
    main()
