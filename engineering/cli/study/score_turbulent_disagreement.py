"""先在源端逐次留一得到真实错例，再决定陌生测量与固定主模型意见不合时听谁。"""
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.metrics import f1_score, recall_score

from score_turbulent_contrast import DATA, DEST as PRIOR, DYNAMIC, PAPER, LocalAcceptance, LocalStore
from score_turbulent_contrast import PairedGasContrast, atomic_json, digest, exposure_response, read, split_members

from harness.models.turbulent_rescue import ANCHOR, PRIMARY, RescuePolicy

DEST = PRIOR.parent / "turbulent-disagreement"


def source_history(x, groups, names, cuts):
    held = np.empty((len(x), len(names)))
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        # 保留其余完整暴露，估计噪声时仍有29套条件包含两次实验。
        estimate = PairedGasContrast().fit(x[keep], groups[keep]).outputs(x[i:i + 1])
        held[i] = [estimate[k][0] - cuts[k] for k in names]
    return held


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("本开发作业已交付")
        meta = read(DATA / "source.json")
        package = DATA / "uci309.zip"
        if sha256(package.read_bytes()).hexdigest() != meta["原包SHA256"]:
            raise ValueError("源包不一致")
        previous = read(PRIOR / "scores.json")
        old = read(PRIOR / "accepted" / (previous["request"] + ".json"))["payload"]
        bundle = PRIOR / "models.joblib"
        if sha256(bundle.read_bytes()).hexdigest() != old["权重SHA256"] or digest(old) != previous["request"]:
            raise ValueError("主模型或上项回执身份不一致")
        model, cuts = joblib.load(bundle)
        rec = read(DATA / "development.json")
        ids = [v["id"] for v in rec["development"]]
        if ids != old["ids"] or set(ids) & set(meta["封存编号"]):
            raise ValueError("开发范围不一致")
        with ZipFile(package) as z:
            split = split_members(z.infolist())
            x, groups = {}, []
            for subset in ("source", "development"):
                rows = []
                for (uid, member, _), r in zip(split[subset], rec[subset], strict=True):
                    raw = z.read(member)
                    if uid != r["id"] or sha256(raw).hexdigest() != r["原件SHA256"]:
                        raise ValueError("输入身份错位")
                    rows.append(exposure_response(raw))
                    if subset == "source":
                        groups.append(Path(member.filename).name.lower().split("_")[2:])
                x[subset] = np.stack(rows)
        names = [k for k in cuts if k not in DYNAMIC]
        y_src = np.array([r["label"] == 2 for r in rec["source"]])
        held = source_history(x["source"], np.array(groups), names, cuts)
        # 源端与目标距离的量纲取自主模型；现场既无成分名称也无标签。
        policy = RescuePolicy(model.scale.transform(x["source"]) @ model.whitener, held, y_src, names)
        estimate = model.outputs(x["development"])
        margins = np.stack([estimate[k] - cuts[k] for k in names], axis=1)
        new, changes = policy.predict(model.scale.transform(x["development"]) @ model.whitener, margins)
        trace = {k: v.tolist() for k, v in new.items()}
        pred = dict(old["预测"])
        for i, name in enumerate(names):
            v = (1 + (margins[:, i] >= 0)).tolist()
            if v != pred[name]:
                raise ValueError("强固定基础预测未复现")
            for peer in (ANCHOR, PAPER):
                for mode, vote in (("一票报有", np.maximum), ("两票报有", np.minimum)):
                    pred[f"{peer}合用{name}/{mode}"] = vote(v, pred[peer]).tolist()
        # 纳入固定主模型的阈值曲线。源端或开发对阈值的选择成本不留给候选独享。
        for cut in (.2, .3, .4, .5, .6, .7, .8):
            pred[f"固定去噪阈值{cut}"] = (1 + (estimate[ANCHOR] >= cut)).tolist()
        pred.update(trace)
        weights = DEST / "policy.joblib"
        joblib.dump(policy, weights)
        payload = {"ids": ids, "预测": pred, "切换": changes, "基础回执": previous["request"],
                   "策略摘要": sha256(weights.read_bytes()).hexdigest()}
        key = digest(payload)
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        if not receiver.submit(key, payload)["accepted"]:
            raise ValueError("分歧作业未交接")
        sent = read(receiver.output / (key + ".json"))["payload"]
        if digest(sent) != key:
            raise ValueError("交付内容改变")
        yy = np.array([r["label"] for r in rec["development"]])
        scores = {}
        base_good = yy == pred[ANCHOR]
        for k, v in sent["预测"].items():
            ok = yy == v
            scores[k] = {"F1": float(f1_score(yy, v, average="macro")), "正确": int(ok.sum()),
                         "最差召回": float(recall_score(yy, v, average=None).min()),
                         "改对": int((ok & ~base_good).sum()), "改错": int((~ok & base_good).sum())}
        dyn = (*DYNAMIC, "标气距离选择背景", "标气距离组合背景", PRIMARY)
        fixed = max((k for k in scores if k not in (*dyn, PAPER)), key=lambda k: scores[k]["F1"])
        paper = max((k for k in scores if k.startswith(PAPER)), key=lambda k: scores[k]["F1"])
        diff = scores[PRIMARY]["F1"] - max(scores[fixed]["F1"], scores[paper]["F1"])
        decision = {"固定": fixed, "同工具论文": paper, "开发最小领先": diff,
                    "进入独立": diff >= .01 and scores[PRIMARY]["最差召回"] >= scores[fixed]["最差召回"] - .02}
        atomic_json(DEST / "scores.json", {"request": key, "scores": scores, "decision": decision})
        print(json.dumps({"现场": scores[PRIMARY], "固定": scores[fixed], "论文组合": scores[paper], "判定": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()
