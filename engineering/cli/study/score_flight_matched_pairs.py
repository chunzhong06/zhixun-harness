"""同一飞机、相同起点与航班间隔带内，检测可比工况配对是否修复差分训练。"""
import json
import numpy as np
import joblib

from score_flight_life_pairs import (DEST as PRIOR, BASE, source_features, extract, receipt, measures,
                                     digest, LocalAcceptance, LocalStore, atomic_json, file_hash)
from harness.methods.flight_life_pairs import LifeDifferenceFit, matched_indices
from harness.methods.flight_complement import fixed_mix

DEST = BASE / "flight-matched-pairs"


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已完成同工况配对复核")
        prep, x, y, keys = source_features()
        operating = x[:, -10:]
        models, checks = {}, {}
        for kind in ("近", "随机", "远"):
            left, right, weights = matched_indices(keys, operating, kind)
            models[kind] = LifeDifferenceFit(x, y, keys, pairs=(left, right, weights.copy()))
            distance = (operating[left] - operating[right]) / operating.std(axis=0).clip(1e-6)
            checks[kind] = {"配对数": len(left), "标准工况距离": float(np.sqrt(np.mean(distance ** 2))),
                            "平均飞行间隔": float(np.mean(y[left] - y[right]))}
        if checks["近"]["配对数"] != checks["随机"]["配对数"] or checks["远"]["配对数"] != checks["近"]["配对数"]:
            raise ValueError("三种配对的源端预算不相同")
        joblib.dump(models["近"], DEST / "matched.joblib")
        prior = json.loads((PRIOR / "scores.json").read_bytes())
        records = {}
        for site, meta in prior["sites"].items():
            ids, vals, truth = extract(site, "test", prep)
            old = receipt(PRIOR, site, meta)
            if old["ids"] != [list(k) for k in ids]:
                raise ValueError("留出航班错位")
            pred = {}
            for kind, model in models.items():
                generated = model.predict(vals)
                np.testing.assert_allclose(generated["只学绝对寿命"], old["predictions"]["只学绝对寿命"], rtol=1e-9, atol=1e-9)
                pred[kind + "工况差分"] = generated["同台差分各半训练"].tolist()
            records[site] = (ids, truth, {**old["predictions"], **pred})
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        scores = {}
        # 固定组合只用此前完全冻结、未在该项目开发test上拟合的三个工具。
        # 避免用别的整组留出混合器做堆叠，间接把目标机队的标签带入上层训练。
        names = ("近工况差分", "三段联合森林", "整趟序列")
        for site, (ids, truth, preds) in records.items():
            others = {k: v for k, v in records.items() if k != site}
            mt = np.vstack([np.column_stack([v[2][n] for n in names]) for v in others.values()])
            ytrain = np.concatenate([v[1] for v in others.values()])
            source = np.concatenate([np.repeat(k, len(v[0])) for k, v in others.items()])
            matrix = np.column_stack([preds[n] for n in names])
            pred = dict(preds)
            pred["近工况论文森林固定组合"] = (matrix @ fixed_mix(mt, ytrain, source)).tolist()
            payload = {"ids": [list(k) for k in ids], "predictions": pred}
            key = digest([site, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("工况配对预测未交接")
            observed = receipt(DEST, site, {"request": key})["predictions"]
            owner = np.array([k[0] for k in ids])
            scores[site] = {"request": key, "rows": len(ids),
                            "scores": {k: measures(truth, np.array(v), owner) for k, v in observed.items()}}
        means = {k: float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in pred}
        atomic_json(DEST / "scores.json", {"sites": scores, "means": means, "source": checks,
                                          "model": file_hash(DEST / "matched.joblib")})
        print(json.dumps({"配对核对": checks, "开发均值": means}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
