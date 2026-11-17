"""按UCI气体任务的类别/列契约构造工程场景；全部观测值和预测均为人工预写。"""
from dataclasses import replace

from harness.core.contracts import BatchContext, Sample
from harness.adapters.datasets.gas_input import gas_task
from harness.core.recipe import PipelineRecipe


def gas_order_scenario():
    """演练类别缺失、回放后剩余漏判及后续答案，不读取真实包或训练模型。

    gas1近期拟合缺类别6且类别5偏少。预写回放结果修复6但仍漏判5，
    预写加权结果用于检查下一步选择和采用接续；这些不是模型实测收益。
    """
    task = gas_task()
    batches, truth = [], {}
    served = []
    for ordinal in range(3):
        rows = []
        for label in task.classes:
            count = (0 if label == 6 else 1 if label == 5 else 3) if ordinal == 1 else 2
            for index in range(count):
                sid = f"synthetic-gas:{ordinal}:fit:{label}:{index}"
                values = tuple(float(label + index / 10 + j / 1000) for j in range(128))
                rows.append(Sample(sid, 1, values, label, "fit", f"source-{ordinal}", ordinal))
                truth[sid] = label
            for index in range(2):
                sid = f"synthetic-gas:{ordinal}:explore:{label}:{index}"
                values = tuple(float(label + .3 + index / 10 + j / 1000) for j in range(128))
                rows.append(Sample(sid, 1, values, label, "explore", f"class-{label}", ordinal))
                truth[sid] = label
            sid = f"synthetic-gas:{ordinal}:adopt:{label}"
            rows.append(Sample(sid, 1, tuple(float(label + j / 1000) for j in range(128)),
                               label, "adopt", f"class-{label}", ordinal))
            truth[sid] = label
        if ordinal == 1:
            for label in (5, 6):
                sid = f"synthetic-gas:serve:{label}"
                row = Sample(sid, 1, tuple(float(label + j / 1000) for j in range(128)),
                             None, "serve", f"class-{label}", ordinal)
                rows.append(row)
                served.append(row)
                truth[sid] = label
        if ordinal == 2:
            rows.extend(replace(r, revision=2, label=truth[r.sample_id], role="fit", available_at=2) for r in served)
        batches.append(BatchContext(task.task_id, f"gas{ordinal}", ordinal, tuple(rows)))
    base, replay, weighted = dict(truth), dict(truth), dict(truth)
    for sid, label in truth.items():
        if (":1:explore:" in sid or ":1:adopt:" in sid or ":serve:" in sid):
            if label == 6 or (label == 5 and not sid.endswith(":1")):
                base[sid] = 1
            if label == 5:
                replay[sid] = 1
    weighted_key = PipelineRecipe(class_weight="balanced").key
    fixture = {"predictions": {"base": base, "replay": replay, "weighted": weighted},
               "build_outputs": {b.batch_id: {"recent": "base", "uniform": "replay", "coverage": "replay"} for b in batches},
               "component_build_outputs": {b.batch_id: {weighted_key: "weighted"} for b in batches}}
    return task, tuple(batches), fixture
