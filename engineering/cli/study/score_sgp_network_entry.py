"""从装机试验回执恢复当时的取舍，在隔离的原始输入入口重发论文组合和自主版。"""
from hashlib import md5, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import subprocess
import sys

import h5py
import numpy as np
from sklearn.metrics import f1_score

from score_sgp_cycle_entry import DEST as ENTRY, SOURCE, MD5, command, export, labels, LocalStore, LocalAcceptance, atomic_json, digest
from score_sgp_transfer_network import DEST as NET, FIXED, SELECTED

DEST = NET.parent / "sgp-network-entry"


def verify(work):
    scored = json.loads((NET / "scores.json").read_bytes())
    entry = json.loads((ENTRY / "scores.json").read_bytes())["装机"]
    source_weight = NET / "service-source.pt"
    if sha256(source_weight.read_bytes()).hexdigest() != scored["权重"][source_weight.name]:
        raise ValueError("源网络身份改变")
    with h5py.File(SOURCE) as data:
        source_ids = data["targets_train/range"][:].ravel()
        concentration = data["targets_train/VOCsum"][:].ravel().astype(np.float32)
        if len(source_ids) != len(concentration):
            raise ValueError("原始标气数不符")
        center, scale = float(concentration.mean()), float(concentration.std())
        # 源权重来自迁移实测，用其训练时相同的周期行恢复输出单位。
        runner = os.getenv("ZHIXUN_TORCH_PYTHON", str(Path(sys.executable).parents[1] / "aegis_env/python.exe"))
        source_pack = DEST / "source-release.pt"
        subprocess.run([runner,"-B","-c",
            "import sys,torch; w=torch.load(sys.argv[1],map_location='cpu',weights_only=True);"
            "torch.save(dict(weights=w,center=float(sys.argv[3]),scale=float(sys.argv[4])),sys.argv[2])",
            str(source_weight),str(source_pack),str(center),str(scale)],check=True)
        records = {}
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        for key, request in scored["requests"].items():
            old = json.loads((NET / "accepted" / (request+".json")).read_bytes())["payload"]
            if digest(old) != request or old["上轮回执"] != entry[key]["请求"]:
                raise ValueError("实验交付身份变化")
            ids = np.asarray(old["服务混合物"])
            export(data, key.split("-")[0], "test", ids, work / "input.npz")
            prior = ENTRY / (key+".joblib")
            release = {"threshold": old["阈值"], "calibration": {"path": os.path.relpath(prior, DEST),
                       "sha256": entry[key]["sha256"]}, "networks": {}}
            for mode, arm in (("fixed", scored["固定"]), ("adaptive", old["装机选用"])):
                blend = arm.startswith("论文与标气校正各半/")
                name = arm.split("/")[-1] if arm != FIXED else None
                release[mode] = {"blend": blend, "network": name}
                if name:
                    if name == "直接用源网络":
                        w = source_pack
                        identity = sha256(w.read_bytes()).hexdigest()
                    else:
                        w = NET / (key+"-"+name+".pt")
                        identity = scored["权重"][w.name]
                    release["networks"][name] = {"path": os.path.relpath(w, DEST), "sha256": identity}
            manifest = DEST / (key+".json")
            atomic_json(manifest, release)
            manifest_hash = sha256(manifest.read_bytes()).hexdigest()
            pred = command("predict-release", "--artifact", manifest, "--sha256", manifest_hash, "--input", work / "input.npz")
            if pred["mixture_ids"] != ids.tolist():
                raise ValueError("服务混合物身份变化")
            for mode, arm in (("adaptive", SELECTED), ("fixed", scored["固定"])):
                if (pred[mode+"_class"] != old["实发分类"][arm]
                        or not np.allclose(pred[mode], old["实发浓度"][arm], atol=1e-9, rtol=1e-10)):
                    raise ValueError("接回原始入口后实发分类或浓度改变")
            delivered = digest(pred)
            if not receiver.submit(delivered, pred)["accepted"]:
                raise ValueError("提交预测未成功")
            records[key] = {"清单SHA256":manifest_hash,"预测回执":delivered,"选用":old["装机选用"],
                            "上轮回执":request,"阈值":old["阈值"]}
        truth = labels(data,"test",ids)
    for row in records.values():
        p = json.loads((receiver.output / (row["预测回执"]+".json")).read_bytes())["payload"]
        if digest(p) != row["预测回执"] or p["mixture_ids"] != ids.tolist():
            raise ValueError("重发服务身份错误")
        for mode in ("adaptive","fixed"):
            row[mode] = float(f1_score(1+(truth>row["阈值"]),p[mode+"_class"],average="macro"))
    summary = {"固定F1":float(np.mean([r["fixed"] for r in records.values()])),
               "现场F1":float(np.mean([r["adaptive"] for r in records.values()])),
               "每种策略交付次数":len(ids)*len(records)}
    atomic_json(DEST/"scores.json",{"装机":records,"结果":summary})
    print(json.dumps({"卷积原始入口":summary},ensure_ascii=False))


if __name__ == "__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists() or md5(SOURCE.read_bytes()).hexdigest()!=MD5:
            raise ValueError("作业已完成或作者原始包变更")
        with TemporaryDirectory(dir=DEST) as temp:
            verify(Path(temp))
