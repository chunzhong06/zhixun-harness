"""从源端标气构造背景参照，在相同的完整风洞物理暴露上与固定方法和论文预测配对。"""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from score_turbulent_source import DATA, DEST as PRIOR, LocalAcceptance, LocalStore, atomic_json, digest, split_members
from harness.models.turbulent_reference import BackgroundReference, exposure_response

DEST = PRIOR.parent / "turbulent-reference"
ADAPTIVE = ("标气距离选择背景", "标气距离组合背景")
PAPER = "论文全时序抑制核"


def read(path):
    return json.loads(path.read_bytes())


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有本协议开发交付")
        meta = read(DATA / "source.json")
        archive = DATA / "uci309.zip"
        if sha256(archive.read_bytes()).hexdigest() != meta["原包SHA256"]:
            raise ValueError("原始包身份改变")
        records = read(DATA / "development.json")
        source, target = [], []
        groups = []
        with ZipFile(archive) as z:
            split = split_members(z.infolist())
            for subset, features in (("source", source), ("development", target)):
                for (identity, member, _), row in zip(split[subset], records[subset], strict=True):
                    raw = z.read(member)
                    if identity != row["id"] or sha256(raw).hexdigest() != row["原件SHA256"]:
                        raise ValueError("实际读取的暴露与登记不符")
                    features.append(exposure_response(raw))
                    if subset == "source":
                        groups.append(Path(member.filename).name.lower().split("_")[2:])
        x, labels, groups = np.stack(source), np.array([r["label"] for r in records["source"]]), np.array(groups)
        # 成分和浓度是实验室标定记录，只给训练侧。阈值在源内留出预测上选定。
        full = BackgroundReference().fit(x, labels, groups)
        outputs = full.outputs(np.stack(target))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
        held = {name: np.empty(len(x)) for name in outputs}
        for tr, te in folds.split(x, labels):
            model = BackgroundReference().fit(x[tr], labels[tr], groups[tr])
            for name, values in model.outputs(x[te]).items():
                held[name][te] = values
        cutoffs = {}
        for name, values in held.items():
            def quality(cut):
                return (f1_score(labels, 1 + (values >= cut), average="macro"), -abs(cut - .5))
            cutoffs[name] = max((.35, .425, .5, .575, .65), key=quality)
        previous = read(PRIOR / "scores.json")
        old = read(PRIOR / "accepted" / (previous["request"] + ".json"))["payload"]
        ids = [r["id"] for r in records["development"]]
        if old["ids"] != ids or set(meta["封存编号"]) & set(ids) or digest([meta["协议SHA256"], old]) != previous["request"]:
            raise ValueError("原有同批论文及固定预测或划分身份不符")
        # 逐条引入原有对照回执，固定方也可取新模型或与论文等权投票。
        prior_dynamic = {"按当前把握选读法", "按当前把握组合"}
        predictions = {k: v for k, v in old["预测"].items() if k not in prior_dynamic}
        ties = np.array(predictions[PAPER]) == 2
        for name, values in outputs.items():
            predictions[name] = (1 + (values >= cutoffs[name])).tolist()
            if name not in ADAPTIVE:
                predictions["固定论文与" + name] = (1 + ((values >= cutoffs[name]).astype(int) + ties.astype(int) >= 2)).tolist()
        bundle = DEST / "models.joblib"
        joblib.dump((full, cutoffs), bundle)
        payload = {"ids": ids, "预测": predictions, "模型SHA256": sha256(bundle.read_bytes()).hexdigest(),
                   "对照回执": previous["request"]}
        key = digest(payload)
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        if not receiver.submit(key, payload)["accepted"]:
            raise ValueError("标气参照开发预测未接受")
        accepted = read(receiver.output / (key + ".json"))["payload"]
        if digest(accepted) != key:
            raise ValueError("计分回执身份改变")
        y = np.array([r["label"] for r in records["development"]])
        scores = {k: {"F1": float(f1_score(y, v, average="macro")), "最差召回": float(recall_score(y, v, average=None).min()),
                      "正确": int((y == v).sum())} for k, v in accepted["预测"].items()}
        best = max(ADAPTIVE, key=lambda k: scores[k]["F1"])
        fixed = max((k for k in scores if k not in (*ADAPTIVE, PAPER)), key=lambda k: scores[k]["F1"])
        diff = scores[best]["F1"] - max(scores[fixed]["F1"], scores[PAPER]["F1"])
        decision = {"候选": best, "固定": fixed, "对两对手的最小领先": diff,
                    "可进独立": diff >= .01 and scores[best]["最差召回"] >= scores[fixed]["最差召回"] - .02}
        atomic_json(DEST / "scores.json", {"request": key, "scores": scores, "decision": decision})
        print(json.dumps({"开发实测": scores, "判定": decision}, ensure_ascii=False))


if __name__ == "__main__":
    run()
