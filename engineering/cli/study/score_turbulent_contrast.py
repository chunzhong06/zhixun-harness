"""对照同背景的源端成分变化。开发预测交接后再与实际成分核对。"""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

import joblib
import numpy as np
from sklearn.metrics import f1_score, recall_score

from score_turbulent_reference import DATA, DEST as LAST, PAPER, PRIOR, LocalAcceptance, LocalStore
from score_turbulent_reference import atomic_json, digest, exposure_response, read, split_members
from harness.models.turbulent_contrast import PairedGasContrast

DEST = PRIOR.parent / "turbulent-contrast"
DYNAMIC = ("按测量选配对背景", "按测量加权配对背景")


def train(x, groups):
    unique, codes, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
    if len(unique) != 30 or np.any(counts != 2):
        raise ValueError("源端每一释放设定须有两个独立测量")
    # 按读取时已经冻结的编号次序，每个条件各留一次。每一折仍有全套背景。
    # 单次样本无法估计同条件方差，阈值用源内交叉预测的分类排序，不用开发标签。
    alternate = np.zeros(len(x), dtype=bool)
    for i in range(len(unique)):
        alternate[np.flatnonzero(codes == i)[1]] = True
    held = {}
    for mask in (alternate, ~alternate):
        model = PairedGasContrast().fit(x[~mask], groups[~mask])
        for name, v in model.outputs(x[mask]).items():
            held.setdefault(name, np.zeros(len(x)))[mask] = v
    y = groups[:, 0] != "n"
    cutoffs = {}
    for name, values in held.items():
        def quality(cut):
            return f1_score(y, values >= cut, average="macro"), -abs(cut - .5)
        cutoffs[name] = max((.35, .425, .5, .575, .65), key=quality)
    return PairedGasContrast().fit(x, groups), cutoffs


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已完成同协议开发交付")
        meta = read(DATA / "source.json")
        archive = DATA / "uci309.zip"
        if sha256(archive.read_bytes()).hexdigest() != meta["原包SHA256"]:
            raise ValueError("归档身份改变")
        records = read(DATA / "development.json")
        features, groups = {}, []
        with ZipFile(archive) as z:
            split = split_members(z.infolist())
            for subset in ("source", "development"):
                rows = []
                for (uid, member, _), r in zip(split[subset], records[subset], strict=True):
                    raw = z.read(member)
                    if uid != r["id"] or sha256(raw).hexdigest() != r["原件SHA256"]:
                        raise ValueError("原件与交付编号不符")
                    rows.append(exposure_response(raw))
                    if subset == "source":
                        groups.append(Path(member.filename).name.lower().split("_")[2:])
                features[subset] = np.stack(rows)
        model, cuts = train(features["source"], np.array(groups))
        weights = DEST / "models.joblib"
        joblib.dump((model, cuts), weights)
        previous = read(LAST / "scores.json")
        old = read(LAST / "accepted" / (previous["request"] + ".json"))["payload"]
        ids = [r["id"] for r in records["development"]]
        if digest(old) != previous["request"] or old["ids"] != ids or set(ids) & set(meta["封存编号"]):
            raise ValueError("原有回执或开发集身份不符")
        columns = dict(old["预测"])
        prior_dynamic = ("标气距离选择背景", "标气距离组合背景")
        raw = model.outputs(features["development"])
        # 固定订好的论文和平均核也能使用新工具。比较等票的两种决胜规则。
        peers = {name: np.array(columns[name]) for name in (PAPER, "平均核")}
        for name, value in raw.items():
            label = 1 + (value >= cuts[name])
            columns[name] = label.tolist()
            if name not in DYNAMIC:
                for peer, baseline in peers.items():
                    for key, ensemble in (("均报有", np.minimum), ("任一报有", np.maximum)):
                        columns[f"{peer}和{name}/{key}"] = ensemble(baseline, label).tolist()
        payload = {"ids": ids, "预测": columns, "权重SHA256": sha256(weights.read_bytes()).hexdigest(),
                   "对照回执": previous["request"]}
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        key = digest(payload)
        if not receiver.submit(key, payload)["accepted"]:
            raise ValueError("配对开发交付未接受")
        sent = read(receiver.output / (key + ".json"))["payload"]
        if digest(sent) != key:
            raise ValueError("开发回执身份不符")
        y = np.array([r["label"] for r in records["development"]])
        scores = {}
        old_good = peers["平均核"] == y
        for name, labels in sent["预测"].items():
            correct = np.array(labels) == y
            scores[name] = {"F1": float(f1_score(y, labels, average="macro")),
                            "最差召回": float(recall_score(y, labels, average=None).min()), "正确": int(correct.sum()),
                            "修回旧固定错误": int(np.sum(correct & ~old_good)), "损失旧固定正确": int(np.sum(~correct & old_good))}
        best = max(DYNAMIC, key=lambda k: scores[k]["F1"])
        fixed = max((k for k in scores if k not in (*DYNAMIC, *prior_dynamic, PAPER)), key=lambda k: scores[k]["F1"])
        paper = max((k for k in scores if k == PAPER or k.startswith(PAPER + "和")), key=lambda k: scores[k]["F1"])
        diff = scores[best]["F1"] - max(scores[fixed]["F1"], scores[paper]["F1"])
        decision = {"候选": best, "固定": fixed, "论文同工具": paper, "最小领先": diff,
                    "进入独立": diff >= .01 and scores[best]["最差召回"] >= scores[fixed]["最差召回"] - .02}
        atomic_json(DEST / "scores.json", {"request": key, "scores": scores, "decision": decision})
        print(json.dumps({"候选": scores[best], "固定": scores[fixed], "论文组合": scores[paper], "判定": decision}, ensure_ascii=False))


if __name__ == "__main__":
    run()
