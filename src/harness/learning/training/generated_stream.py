"""流式源码模型的公共训练与预测，原历史实验使用同一实现。"""
from hashlib import sha256
import json, subprocess, sys
from time import monotonic
import numpy as np
from harness.learning.models.station_residual import error
from harness.adapters.research.generated_regressor import inspect_source
from harness.adapters.storage.local import atomic_json
def fit_source(source,data,destination,reuse_model=None):
    """代码检查、单线程子进程和超时共同限制一次试训，失败回到驱动观察。"""
    destination.mkdir(parents=True,exist_ok=True)
    started=monotonic()
    try:
        inspect_source(source)
        path=destination/"source.py";path.write_text(source,encoding="utf-8")
        command=[sys.executable,"-B","-m", "harness.cli.fit_generated_regressor",
            "--source",str(path.resolve()),"--data",str(data.resolve()),"--output",str(destination.resolve())]
        if reuse_model is not None:command.extend(["--reuse-model",str(reuse_model.resolve())])
        result=subprocess.run(command,
            cwd=destination,capture_output=True,text=True,encoding="utf-8",timeout=90)
        (destination/"stdout.log").write_text(result.stdout,encoding="utf-8")
        (destination/"stderr.log").write_text(result.stderr,encoding="utf-8")
        if result.returncode:raise ValueError(result.stderr[-1200:])
        with np.load(destination/"base.npz",allow_pickle=False) as z:ids=z["ids"];prediction=z["prediction"]
        return ids,prediction
    finally:
        atomic_json(destination/"cost.json",{"elapsed_seconds":monotonic()-started})
def identity(plan):
    payload = {k: plan[k] for k in ("components", "filters")}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def selected_rows(plan, fit, features, names):
    selected = fit.copy()
    for f in plan["filters"]:
        values = features[:, names.index(f["feature"])]
        selected &= values <= float(f["threshold"]) if f["operator"] == "<=" else values > float(f["threshold"])
    if selected.sum() < 12: raise ValueError(f"筛选前{fit.sum()}条，筛选后{selected.sum()}条，至少12条")
    return selected

def summarize(prediction, known, clock, windows, features, names):
    result = []
    for mask in windows:
        use = mask[clock]; ix = clock[use]; residual = prediction[use] - known[use]
        if not len(ix):
            result.append({"rows":0,"rmse":None,"prediction_minus_answer":None,"median_low_high_error":{}})
            continue
        fields = {}
        for j, field in enumerate(names[:9]):
            values = features[ix,j]; split = float(np.median(values)); lo = values <= split
            if lo.any() and (~lo).any(): fields[field] = [split, float(residual[lo].mean()), float(residual[~lo].mean())]
        result.append({"rows": len(ix), "rmse": error(known[use], prediction[use]),
                       "prediction_minus_answer": float(residual.mean()), "median_low_high_error": fields})
    return result
def predict(model,x,limit=None,fallback=None):
    raw=[];issued=[];failed=[]
    for i,row in enumerate(x):
        try:
            value=np.asarray(model.predict(row[None,:]),dtype=float)
            if value.shape!=(1,) or not np.isfinite(value).all():raise ValueError("无效当前输出")
            number=float(value[0]);raw.append(number)
            if limit is not None and abs(number)>limit:raise ValueError("超限")
            issued.append(max(0.,number))
        except Exception:
            if fallback is None:raise
            if len(raw)==i:raw.append(float("nan"))
            issued.append(float(fallback[i]));failed.append(i)
    return np.asarray(raw),np.asarray(issued),failed
