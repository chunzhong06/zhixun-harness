"""固定源设备总寿命的中位数尺度，仅学习单条传感器观测在寿命过程中的位置。"""
from dataclasses import replace
from hashlib import sha256
from time import process_time

import numpy as np

from compare_fixed_tree_references import OUT,evaluate,frozen
from compare_group_objectives import atomic_json,digest,official
from harness.contracts import ActionPlan
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.training_tools import SklearnBackend


def run(task):
    pkg,checksum,batches=official(task)
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows=[r for r in pkg.train if r.sample_id in ids]
    final={}
    for r in rows:
        final[r.unit_id]=max(final.get(r.unit_id,0),r.label)
    if any(not any(r.unit_id==u and r.label==0 for r in rows) for u in final) or min(final.values())<=0:
        raise ValueError("训练参照的每条设备轨迹都应包含终点")
    scale=float(np.median(list(final.values())))
    normalized=[replace(r,label=r.label/final[r.unit_id]) for r in rows]
    models,receipts=frozen(task)
    recipe=PipelineRecipe(predictor="catboost",processor="none",max_depth=6)
    backend=SklearnBackend(execution_enabled=True)
    start=process_time()
    m=backend.fit(pkg.task,"source",ActionPlan("balanced",pipeline=recipe),normalized)
    secs=process_time()-start
    blob=backend.dump(m)
    out=OUT/"phase/trained"/task
    out.mkdir(parents=True,exist_ok=True)
    (out/"phase.joblib").write_bytes(blob)
    rec={"sha256":sha256(blob).hexdigest(),"recipe":recipe.key,"fit":digest([r.sample_id for r in rows]),
         "cpu_seconds":secs,"scale":scale,"device_count":len(final),"range":[min(final.values()),max(final.values())]}
    if receipts["catboost"]["fit"]!=rec["fit"] or receipts["catboost"]["recipe"]!=rec["recipe"]:
        raise ValueError("绝对与比例模型的配方或输入不同")
    atomic_json(out/"receipt.json",rec)
    arms={"catboost":models["catboost"],"phase":backend.load(blob)}
    meta={"catboost":receipts["catboost"],"phase":rec}
    for cohort in ((task,"pronostia") if task=="bearing" else (task,"phm08")):
        evaluate(cohort,task,arms,meta,pkg,float(np.median([r.label for r in rows])),phase_scale=scale)


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("bearing","engine"))
    run(parser.parse_args().task)
