"""以原公式的逐项标量实现核对BOA、交付时序和固定组合评分。"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from harness.models.online_aggregation import aggregate
from harness.runtime.storage import atomic_json


def read(path): return json.loads(path.read_bytes())


def same(a, b, message):
    if not np.allclose(a, b, rtol=1e-11, atol=1e-10, equal_nan=True): raise ValueError(message)


def literal(columns, answers, mode, freeze=None):
    """独立翻译原文图2，不调用被核验类或其更新函数。"""
    count = columns.shape[1]; prior = 1. / count
    weights = [prior] * count; rates = [0.] * count; totals = [0.] * count
    squares = [0.] * count; maxima = [0.] * count; output = []; used = []
    for i, (row, answer) in enumerate(zip(columns, answers, strict=True)):
        used.append(weights.copy()); prediction = sum(w*p for w,p in zip(weights, row, strict=True))
        output.append(prediction)
        if not math.isfinite(answer) or (freeze is not None and i >= freeze): continue
        loss_mean = sum(w*(p-answer)**2 for w,p in zip(weights, row, strict=True))
        logs = []
        for j in range(count):
            excess = 2*(prediction-answer)*(row[j]-prediction) if mode == "convex" else (row[j]-answer)**2-loss_mean
            maxima[j] = max(maxima[j], abs(excess))
            k = max(-20, math.ceil(math.log2(max(maxima[j], 2.**-20))))
            extent = 2.**(k+1)
            totals[j] += (excess*(1+rates[j]*excess) + (extent if rates[j]*excess > .5 else 0.)) / 2
            squares[j] += excess**2
            rates[j] = min(1/extent, math.sqrt(-math.log(prior)/squares[j]) if squares[j] else math.inf)
            logs.append(math.log(prior)+math.log(rates[j])-rates[j]*totals[j])
        maximum = max(logs); raw = [math.exp(v-maximum) for v in logs]; total = sum(raw)
        weights = [v/total for v in raw]
    return np.asarray(output), np.asarray(used)


def main():
    ap = ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--archive", type=Path, required=True)
    args = ap.parse_args(); out = args.run; result = read(out / "scores.json"); pool = Path(result["pool"])
    if (out / "audit.json").exists(): raise ValueError("不可覆盖核验回执")
    scene.configure(result["start"]); table = read(out / "fixed-combinations.json"); rows = table["candidates"]
    count = len(table["identities"])
    expected = [(a,a,1.) for a in range(count)] + [(a,b,w) for a in range(count) for b in range(a+1,count) for w in (.25,.5,.75)]
    if [(r["left"],r["right"],r["weight"]) for r in rows] != expected: raise ValueError("固定名单改变")
    scores = {}; issued_count = 0; scored_count = 0; sample_weights = 0
    for site in result["counts"]:
        with np.load(out/site/"warmup.npz") as h, np.load(out/site/"issued.npz") as z, np.load(pool/site/"issued.npz") as old:
            columns = np.vstack((h["experts"],z["experts"])); known = np.r_[h["returned"],z["returned"]]; split = len(h["ids"])
            meta = read(pool/site/"deployment.json")
            for j,option in enumerate(meta["options"]): same(z["experts"][:,j],old["bank-"+option["name"]],"专家交付不同")
            for mode in ("convex","selection"):
                values, weights = literal(columns,known,mode)
                same(values[split:],z["boa-"+mode],"原公式预测不同");same(weights[split:],z[mode+"_weights"],"原公式权重不同")
                frozen, frozen_weights = literal(columns,known,mode,split)
                same(frozen[split:],z["frozen-"+mode],"冻结对照不同");same(frozen_weights[split],z[mode+"_opening"],"开场权重不同")
                same(weights.sum(axis=1),1.,"权重未归一");sample_weights += weights.size
                if np.min(weights)<0: raise ValueError("出现非凸权重")
                # 更改本条及未来所有反馈，不能改变此前及本条已经交出的预测。
                cut = split + 28*24; changed = known.copy(); mask = np.arange(len(known))>=cut
                changed[mask & np.isfinite(changed)] += 137.
                changed_predictions, changed_weights = aggregate(columns,changed,mode)
                same(changed_predictions[:cut+1],values[:cut+1],"未来答案改变了已发预测")
                same(changed_weights[:cut+1],weights[:cut+1],"未来答案改变了已用权重")
            truth = scene.readings(args.archive,site).to_numpy()[z["ids"]]; valid = np.isfinite(truth)
            issued_count += len(truth); scored_count += int(valid.sum())
            same(z["uniform"],z["experts"].mean(axis=1),"等权对照不同")
            for key in result["station_rmse"]:
                if key=="best-fixed-combination": continue
                value = float(np.sqrt(np.mean((z[key][valid]-truth[valid])**2))); scores.setdefault(key,{})[site]=value
                same(value,result["station_rmse"][key][site],"完整评分不同")
            control = z["fixed_controls"];same(control[:,:-1],z["experts"],"固定库专家不同")
            same(control[:,-1],old["previous-best-fixed"],"原最强固定不同")
            # 独立直接形成每个组合的逐条预测并评分，不沿用协方差快捷公式。
            for record in rows:
                a,b,w=record["left"],record["right"],record["weight"]
                prediction=w*control[valid,a]+(1-w)*control[valid,b]
                score=float(np.sqrt(np.mean((prediction-truth[valid])**2)))
                same(score,record["station_rmse"][site],"固定组合直接预测评分不同")
            best=rows[table["best_index"]];a,b,w=best["left"],best["right"],best["weight"]
            with np.load(out/site/"best-fixed.npz") as fixed:
                same(fixed["prediction"],w*control[:,a]+(1-w)*control[:,b],"最强固定交付不同")
            same(best["station_rmse"][site],result["station_rmse"]["best-fixed-combination"][site],"固定对照转录不同")
    for r in rows: same(r["macro_rmse"],np.mean(list(r["station_rmse"].values())),"固定汇总不同")
    if min(range(len(rows)),key=lambda i:rows[i]["macro_rmse"])!=table["best_index"]: raise ValueError("非最强固定")
    for key,values in result["station_rmse"].items():same(np.mean(list(values.values())),result["macro_rmse"][key],"汇总评分不同")
    # 零差异和缺失反馈覆盖算法退化边界，不使用性能结果作为断言。
    equal=np.full((12,3),7.);missing=np.full(12,np.nan)
    for mode in ("convex","selection"):
        p,w=aggregate(equal,missing,mode);same(p,7.,"缺失反馈异常");same(w,1/3,"缺失反馈更新了权重")
        p,w=aggregate(equal,np.full(12,7.),mode);same(p,7.,"全相同预测异常");same(w,1/3,"全零超额异常")
    audit={"issued":issued_count,"scored":scored_count,"scalar_formula_checked":True,"weights_checked":sample_weights,
           "future_answers_do_not_change_past":True,"fixed_combinations_directly_scored":len(rows),
           "controls_and_scores_verified":True,"primary":result["macro_rmse"]["boa-convex"],
           "best_fixed":result["macro_rmse"]["best-fixed-combination"],"new_training":0,"new_luna_calls":0}
    atomic_json(out/"audit.json",audit); print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":main()
