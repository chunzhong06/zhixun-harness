"""在预留的原始采集批只评价冻结版本；按来源行可重新核对每次改判。"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BASELINE_CPU_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT / "engineering/src"))
import numpy as np
from harness.data.gas_input import read_gas_batches
from harness.models.gas_views import pair_model
from harness.runtime.storage import atomic_json

CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def repair_feedback(original, issued, answers, target):
    """在同一来源和已返回答案的行上复查一项修正；原判乙醇的两种修正有互斥出口。"""
    if not (len(original) == len(issued) == len(answers)):
        raise ValueError("原预测、已发预测和返回答案必须逐行配对")
    opportunities = fixed = broken = still_wrong = 0
    for before, after, truth in zip(original, issued, answers):
        opportunities += int(before == 1 and truth == target)
        if before != 1 or after != target:
            continue
        if after == truth:
            fixed += 1
        elif before == truth:
            broken += 1
        else:
            still_wrong += 1
    return {"opportunities": opportunities, "fixed": fixed, "broken": broken,
            "changed_but_still_wrong": still_wrong, "target_errors_left": opportunities-fixed}


def score(rows):
    """六类固定等权；矩阵以实际类别为行，预测类别为列。"""
    y = np.array([r["answer"] for r in rows])
    p = {k: np.array([r[k] for r in rows]) for k in ("base", "single", "double")}
    out = {"rows":len(rows), "models":{}, "changes":{}}
    for k,v in p.items():
        matrix = np.zeros((6,6), dtype=int)
        np.add.at(matrix, (y-1,v-1), 1)
        den = matrix.sum(0) + matrix.sum(1)
        f = np.divide(2*matrix.diagonal(), den, out=np.zeros(6), where=den!=0)
        out["models"][k] = {"macro_f1":float(f.mean()), "confusion":matrix.tolist()}
    for parent, child in (("base","single"), ("single","double")):
        fixed = (p[parent]!=y) & (p[child]==y)
        broke = (p[parent]==y) & (p[child]!=y)
        prior = (p["base"]!=y) & (p[parent]==y)
        out["changes"][child] = {"fixed":int(fixed.sum()), "broken":int(broke.sum()),
            "fixed_by_class":np.bincount(y[fixed],minlength=7)[1:].tolist(),
            "broken_by_class":np.bincount(y[broke],minlength=7)[1:].tolist(),
            "prior_fixes_lost":int((prior & broke).sum())}
    out["repair_feedback"] = {CLASSES[t-1]: repair_feedback(p["base"], p["double"], y, t) for t in (4, 2)}
    return out


def run(protocol, folder):
    cfg = json.loads(protocol.read_bytes())
    package = ROOT / cfg["official"]
    if sha256(package.read_bytes()).hexdigest() != cfg["official_sha256"]:
        raise ValueError("官方文件与冻结协议不符")
    folder.mkdir(parents=True, exist_ok=True)
    pred_file = folder / "predictions.jsonl"
    if pred_file.exists() or (folder/"result.json").exists():
        raise FileExistsError("保留首次评价记录；复核请另选输出目录")
    sys.path.insert(0, str(package))
    from baselines.model import Model
    from baselines.data import flatten
    Model = pair_model(Model)
    known = set()
    with ZipFile(package) as z:
        models = {}
        for split in ("train","validation"):
            for line in z.read(f"public/{split}.jsonl").splitlines():
                known.add(tuple(json.loads(line)["inputs"]["features"]))
        for name, arm in cfg["arms"].items():
            path = ROOT / arm["path"]
            if "member" in arm and sha256(z.read(arm["member"])).hexdigest() != arm["sha256"]:
                raise ValueError("本地原版必须与官方检查点逐字节相同")
            if sha256(path.read_bytes()).hexdigest() != arm["sha256"]:
                raise ValueError(f"{name} 的权重与冻结协议不符")
            model = Model.load(path)
            if model.meta["classes"] != CLASSES or model.meta["paths"] != flatten({"features":[0.0]*128})[0]:
                raise ValueError("模型类别或特征顺序不同")
            models[name] = model
    audits, answers, seen, times = {}, {}, set(), {}
    # 优先完成所有留出批的预测；标签按来源行独立保存，直至预测记录关闭才用于计分。
    with pred_file.open("x", encoding="utf-8") as stream:
        for n in cfg["source_batches"]:
            batch, = read_gas_batches(ROOT/cfg["source"], expected_sha256=cfg["source_sha256"],batch_number=n)
            rows = [r for r in batch.records if r.values not in known]
            if not rows:
                raise ValueError("排除开发行后该批无可评样本")
            audits[n] = {"member":batch.member,"member_sha256":batch.member_sha256,"source_rows":len(batch.records),
                "excluded_development_lines":[r.source_line for r in batch.records if r.values in known],
                "rows_matching_prior_eval_batch":sum(r.values in seen for r in rows),
                "repeat_feature_rows_within_batch":len(rows)-len({r.values for r in rows})}
            seen.update(r.values for r in rows)
            times[n] = Counter()
            for idx,r in enumerate(rows):
                x = np.asarray([r.values],dtype=np.float64)
                pred = {"batch":n,"line":r.source_line}
                order = list(models)
                # 轮换调用顺序；时间只计已加载模型的单条计算，不包含文件读取与计分。
                start_at = idx % len(order)
                for arm in order[start_at:] + order[:start_at]:
                    start = perf_counter_ns()
                    probs = models[arm].predict(x)[0]
                    times[n][arm] += perf_counter_ns()-start
                    if probs.shape!=(6,) or not np.isfinite(probs).all():
                        raise ValueError("预测分数无效")
                    pred[arm] = int(probs.argmax())+1
                stream.write(json.dumps(pred,separators=(",",":"))+"\n")
                answers[n,r.source_line] = r.label
    scored = [json.loads(line) for line in pred_file.read_bytes().splitlines()]
    for row in scored:
        row["answer"] = answers[row["batch"],row["line"]]
    batches = {}
    for n in cfg["source_batches"]:
        subset = [r for r in scored if r["batch"]==n]
        batches[n] = score(subset)
        batches[n]["mean_predict_us"] = {k:v/len(subset)/1000 for k,v in times[n].items()}
    atomic_json(folder/"result.json", {"protocol_sha256":sha256(protocol.read_bytes()).hexdigest(),
        "predictions_sha256":sha256(pred_file.read_bytes()).hexdigest(),"classes":CLASSES,
        "overlap":audits,"batches":batches,"equal_batch_macro_f1":{
        k:float(np.mean([b["models"][k]["macro_f1"] for b in batches.values()])) for k in models}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    run(args.protocol,args.output)
