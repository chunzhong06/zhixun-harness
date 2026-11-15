"""各树对每个航段的独立输出能否给现场提供固定平均没有的读法。"""
import json
import joblib
import numpy as np
from sklearn.linear_model import Ridge

from score_flight_segment_training import (BASE, DEST as PRIOR, PREP, PRIMARY as OLD, LocalAcceptance,
    LocalStore, digest, receipt, measures, atomic_json, load_site, file_hash)
from harness.models.flight_segment_precision import flight_evidence, SegmentPrecision
from harness.methods.flight_complement import fixed_mix

DEST = BASE / "flight-segment-precision"
PRIMARY = "按当趟树的分歧分配航段"
NAMES = ("前段独立估计", "中段独立估计", "末段独立估计")


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("本次开发已开分")
        published = json.loads((PRIOR / "scores.json").read_bytes())
        artifact = PRIOR / "segment.joblib"
        if file_hash(artifact) != published["model"]:
            raise ValueError("航段训练权重已改变")
        learner = joblib.load(artifact)
        prep = joblib.load(PREP).prep
        records = {}
        for site, item in published["sites"].items():
            keys, views, truth = load_site(site, "test", prep)
            previous = receipt(PRIOR, site, item)
            predictions, e = flight_evidence(learner, views)
            old = previous["predictions"]
            np.testing.assert_allclose(predictions, np.column_stack([old[n] for n in NAMES]), atol=1e-8)
            if previous["ids"] != [list(k) for k in keys]:
                raise ValueError("航班未对齐")
            records[site] = (predictions, e, truth, old, previous["ids"])
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        summary = {}
        for site, (p, e, truth, old, keys) in records.items():
            src = {k:v for k,v in records.items() if k != site}
            src_p, src_e, y = [np.concatenate([v[i] for v in src.values()]) for i in range(3)]
            sites = np.concatenate([np.repeat(k, len(v[2])) for k,v in src.items()])
            policy = SegmentPrecision().fit(src_p, src_e, y, sites)
            weights = policy.weights(e)
            sw = np.array([len(y) / (len(src) * len(src[k][2])) for k in sites])
            public_weights = np.average(policy.weights(src_e), axis=0, weights=sw)
            paper_src = np.concatenate([v[3]["整趟序列"] for v in src.values()])
            fixed = fixed_mix(np.column_stack((src_p, paper_src)), y, sites)
            affine = Ridge(alpha=len(y) * .25).fit(src_p, y, sample_weight=sw)
            pred = dict(old)
            pred.update({PRIMARY: np.sum(p * weights, axis=1).tolist(),
                         "只给段号固定分量": (p @ policy.fixed).tolist(),
                         "树分歧权重冻结为均值": (p @ public_weights).tolist(),
                         "三段固定线性校准": affine.predict(p).clip(0).tolist(),
                         "三段和论文固定最优": (np.column_stack((p, old["整趟序列"])) @ fixed).tolist()})
            # 固定端也允许直接按树内离散程度加权，检验学习条件的增量。
            precision = np.exp(-2 * e[:, :, 0])
            pred["直接用树方差倒数"] = np.sum(p * precision / precision.sum(1)[:, None], axis=1).tolist()
            payload = {"ids": keys, "predictions": pred}
            key = digest([site, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("航段权重未交接")
            issued = receipt(DEST, site, {"request": key})["predictions"]
            owners = np.asarray(keys)[:, 0]
            summary[site] = {"request": key, "rows": len(keys),
                             "scores": {k: measures(truth, v, owners) for k,v in issued.items()}}
        means = {k: float(np.mean([x["scores"][k]["rmse"] for x in summary.values()])) for k in pred}
        dyn = {PRIMARY, OLD, "现场与十一工具各半", "可比工况共同误差", "差分现场估错", "差分现场与固定各半",
               "窗口估错调权", "窗口差异校正", "前次现场规则", "现场与六工具固定各半", "航班互补调权"}
        fixed = min((k for k in means if k not in dyn), key=means.get)
        paper = min(("整趟序列", "论文与三段固定各半", "三段和论文固定最优"), key=means.get)
        gain = 1 - means[PRIMARY] / min(means[fixed], means[paper])
        decision = {"主候选": PRIMARY, "固定": fixed, "同工具论文": paper, "最小降幅": gain, "进入独立": gain >= .01}
        atomic_json(DEST / "scores.json", {"sites": summary, "means": means, "decision": decision})
        print(json.dumps({"分歧调权": means[PRIMARY], "固定": means[fixed], "同工具论文": means[paper], "判定": decision}, ensure_ascii=False))


if __name__ == "__main__":
    run()
