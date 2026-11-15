"""把论文的卷积迁移和已交付的标气校正交给同一套装机试验，按实发分类评分。"""

from hashlib import md5, sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from sklearn.metrics import f1_score, recall_score

from harness.models.cycle_install import balanced_f1
from harness.models.paired_cycle_teacher import INVERSE, PairedCycleTeacher
from score_sgp_cycle_entry import DEST as ENTRY, INSTALL, SOURCE, MD5, LocalStore, LocalAcceptance, digest, atomic_json, export, labels
from score_sgp_install_cycles import raw_part
from score_sgp_paired_teacher import DEST as PRIOR

DEST = ENTRY.parent / "sgp-transfer-network"
FIXED = "固定标气校正"
SELECTED = "装机试验选论文迁移或标气校正"


def compare(work, runner):
    infos = json.loads((INSTALL / "scores.json").read_bytes())["requests"]
    entry = json.loads((ENTRY / "scores.json").read_bytes())["装机"]
    prior = json.loads((PRIOR / "scores.json").read_bytes())
    trials = {}
    with h5py.File(SOURCE) as f:
        ids, source = raw_part(f, "A", "train")
        y = labels(f, "train", ids)
        threshold = float(np.median(y))
        export(f, "A", "train", ids, work / "source.npz", source=True)
        for chip in "BC":
            target_ids, target = raw_part(f, chip, "train")
            if not np.array_equal(target_ids, ids):
                raise ValueError("目标与源标气身份不对应")
            broad_key = infos[f"{chip}-80"]
            broad = json.loads((INSTALL / "accepted" / (broad_key + ".json")).read_bytes())["payload"]
            if digest(broad) != broad_key:
                raise ValueError("源端共同留出名单改变")
            excluded = np.asarray(broad["装机混合物"])[::4]
            for size in (20, 80):
                key = f"{chip}-{size}"
                install_key = infos[key]
                parent = json.loads((INSTALL / "accepted" / (install_key + ".json")).read_bytes())["payload"]
                if digest(parent) != install_key:
                    raise ValueError("装机名单回执改变")
                install_ids = np.asarray(parent["装机混合物"])
                ix = np.searchsorted(ids, install_ids)
                service = np.asarray(parent["服务混合物"])
                if set(ids) & set(service) or not np.array_equal(ids[ix], install_ids):
                    raise ValueError("装机或服务身份改变")
                export(f, chip, "train", install_ids, work / (key + "-install.npz"))
                if size == 20:
                    export(f, chip, "test", service, work / (chip + "-service.npz"))
                val = np.arange(size) % 4 == 0
                keep = ~np.isin(ids, excluded)
                corrector = PairedCycleTeacher().fit(
                    [group for group, use in zip(source, keep, strict=True) if use], y[keep],
                    [target[i] for i in ix[~val]], (np.cumsum(keep)-1)[ix[~val]], 1.,
                )
                pred = corrector.predict([target[i] for i in ix[val]])[INVERSE]
                trials[key] = (y[ix[val]], pred, install_ids[val], service)

    worker = Path(__file__).with_name("sgp_transfer_cpu.py")
    subprocess.run([str(runner), "-B", str(worker), str(work), str(DEST)], check=True)
    columns = json.loads((work / "outputs.json").read_bytes())
    receiver = LocalAcceptance()
    receiver.output = DEST / "accepted"
    receipts = {}
    for key, (known, fixed_trial, install_ids, service) in trials.items():
        part = columns[key]
        if part["trial"]["ids"] != install_ids.tolist() or part["service"]["ids"] != service.tolist():
            raise ValueError("网络返回的测量身份错误")
        oldkey = entry[key]["请求"]
        old = json.loads((ENTRY / "accepted" / (oldkey + ".json")).read_bytes())["payload"]
        if digest(old) != oldkey or old["mixture_ids"] != service.tolist():
            raise ValueError("固定预测回执改变")
        val = {FIXED: fixed_trial}
        predictions = {FIXED: np.asarray(old["fixed"])}
        for name, v in part["trial"]["values"].items():
            v = np.asarray(v)
            output = np.asarray(part["service"]["values"][name])
            val[name], predictions[name] = v, output
            blend = "论文与标气校正各半/" + name
            val[blend], predictions[blend] = (fixed_trial + v)/2, (predictions[FIXED] + output)/2
        def quality(arm):
            return balanced_f1(known, val[arm], threshold), -float(np.mean((known-val[arm])**2))
        winner = max(val, key=quality)
        predictions[SELECTED] = predictions[winner]
        payload = {"服务混合物": service.tolist(), "装机选用": winner, "装机试验配方": install_ids.tolist(),
                   "装机验证": {arm: list(quality(arm)) for arm in val}, "阈值": threshold, "上轮回执": oldkey,
                   "实发浓度": {k: v.tolist() for k,v in predictions.items()},
                   "实发分类": {k: (1+(v>threshold)).tolist() for k,v in predictions.items()}}
        request = digest(payload)
        if not receiver.submit(request, payload)["accepted"]:
            raise ValueError("交付未确认")
        receipts[key] = request

    with h5py.File(SOURCE) as f:
        yy = 1 + (labels(f, "test", service) > threshold)
    scores = {}
    for key, request in receipts.items():
        payload = json.loads((receiver.output / (request+".json")).read_bytes())["payload"]
        if digest(payload) != request or payload["服务混合物"] != service.tolist():
            raise ValueError("实发记录改变")
        scores[key] = {arm: {"F1": float(f1_score(yy,v,average="macro")),
                             "弱类召回": float(recall_score(yy,v,average=None).min())} for arm,v in payload["实发分类"].items()}
    means = {arm: {m: float(np.mean([s[arm][m] for s in scores.values()])) for m in ("F1","弱类召回")}
             for arm in next(iter(scores.values()))}
    fixed = max((a for a in means if a != SELECTED), key=lambda a: means[a]["F1"])
    paper = max((a for a in means if a not in (SELECTED, FIXED)), key=lambda a: means[a]["F1"])
    best_paper = max(means[paper]["F1"], prior["decision"]["论文特征组合F1"])
    gap = means[SELECTED]["F1"] - max(means[fixed]["F1"], best_paper)
    decision = {"现场F1": means[SELECTED]["F1"], "最强固定F1": means[fixed]["F1"], "最强论文组合F1": best_paper,
                "卷积论文同工具F1": means[paper]["F1"], "差值": gap,
                "过初筛": gap >= .01 and means[SELECTED]["弱类召回"] >= means[fixed]["弱类召回"]-.02}
    weights = {p.name: sha256(p.read_bytes()).hexdigest() for p in DEST.glob("*.pt")}
    atomic_json(DEST / "scores.json", {"decision": decision, "means": means, "scores": scores, "固定": fixed,
                                      "卷积对照": paper, "requests": receipts, "权重": weights})
    print(json.dumps({"神经迁移与标气校正": decision}, ensure_ascii=False))


if __name__ == "__main__":
    runner = Path(os.getenv("ZHIXUN_TORCH_PYTHON", str(Path(sys.executable).parents[1] / "aegis_env/python.exe")))
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists() or not runner.is_file() or md5(SOURCE.read_bytes()).hexdigest() != MD5:
            raise ValueError("本项已完成或输入与计算环境不匹配")
        with TemporaryDirectory(dir=DEST) as temp:
            compare(Path(temp), runner)
