"""从发布入口重新装机，只向预测进程传感器原始周期，核对每种混合物的交付。"""

from hashlib import md5, sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from sklearn.metrics import f1_score

from score_sgp_install_cycles import DEST as INSTALL, SOURCE, MD5, LocalStore, LocalAcceptance, atomic_json, digest, labels
from score_sgp_paired_teacher import DEST as PRIOR, SELECT

DEST = PRIOR.parent / "sgp-cycle-entry"
CLI = Path(__file__).resolve().parents[1] / "workflow/sgp_cycle_install.py"


def export(file, chip, split, ids, path, source=False):
    codes = np.asarray(file["targets_" + split + "/range"]).ravel().astype(int)
    mask = np.isin(codes, ids)
    raw = np.stack([np.asarray(file[r]).T[mask] for r in file[f"sensor{chip}_{split}"][:].ravel()], axis=1)
    values = {"cycle_ids": codes[mask], "mixture_ids": ids, "cycles": raw}
    if source:
        values["concentrations"] = labels(file, split, ids)
    np.savez(path, **values)


def command(*args):
    proc = subprocess.run([sys.executable, "-B", str(CLI), *map(str, args)],
                          stdout=subprocess.PIPE, check=True, text=True, encoding="utf-8")
    return json.loads(proc.stdout)


def deliver(work):
    prior = json.loads((PRIOR / "scores.json").read_bytes())
    original = json.loads((INSTALL / "scores.json").read_bytes())
    fixed = prior["decision"]["固定"]
    receiver = LocalAcceptance()
    receiver.output = DEST / "accepted"
    releases = {}
    with h5py.File(SOURCE) as f:
        ids = np.unique(f["targets_train/range"][:]).astype(int)
        export(f, "A", "train", ids, work / "source.npz", source=True)
        for key, request in prior["requests"].items():
            chip, _ = key.split("-")
            old = json.loads((PRIOR / "accepted" / (request + ".json")).read_bytes())["payload"]
            install_key = original["requests"][key]
            installation = json.loads((INSTALL / "accepted" / (install_key + ".json")).read_bytes())["payload"]
            if digest(old) != request or digest(installation) != install_key or old["上轮回执"] != install_key:
                raise ValueError("回执链发生变化")
            install_ids = np.asarray(installation["装机混合物"])
            service_ids = np.asarray(old["服务混合物"])
            export(f, chip, "train", install_ids, work / "target.npz")
            export(f, chip, "test", service_ids, work / "service.npz")
            weight = DEST / (key + ".joblib")
            setup = command("install", "--source", work / "source.npz", "--install", work / "target.npz", "--artifact", weight)
            pred = command("predict", "--input", work / "service.npz", "--artifact", weight, "--sha256", setup["sha256"])
            if pred["mixture_ids"] != service_ids.tolist() or setup["selected_strength"] != old["采用强度"]:
                raise ValueError("新进程的标定决定或服务身份改变")
            for field, name in (("fixed", fixed), ("adaptive", SELECT)):
                if (pred[field + "_class"] != old["实发分类"][name]
                        or not np.allclose(pred[field], old["实发浓度"][name], rtol=1e-10, atol=1e-9)):
                    raise ValueError("新进程的浓度或分类未复现")
            receipt = digest(pred)
            if not receiver.submit(receipt, pred)["accepted"]:
                raise ValueError("交付失败")
            releases[key] = {**setup, "请求": receipt, "上轮回执": request}
        truth = labels(f, "test", service_ids)
    for part in releases.values():
        accepted = json.loads((receiver.output / (part["请求"] + ".json")).read_bytes())["payload"]
        if digest(accepted) != part["请求"] or accepted["mixture_ids"] != service_ids.tolist():
            raise ValueError("已交付记录不一致")
        binary = 1 + (truth > part["threshold"])
        for arm in ("adaptive", "fixed"):
            part[arm] = float(f1_score(binary, accepted[arm + "_class"], average="macro"))
    summary = {"固定F1": float(np.mean([v["fixed"] for v in releases.values()])),
               "现场F1": float(np.mean([v["adaptive"] for v in releases.values()])),
               "每种策略重发次数": sum(len(service_ids) for _ in releases)}
    atomic_json(DEST / "scores.json", {"装机": releases, "结果": summary, "入口SHA256": sha256(CLI.read_bytes()).hexdigest()})
    print(json.dumps({"原始逐混合物交付": summary}, ensure_ascii=False))


if __name__ == "__main__":
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists() or md5(SOURCE.read_bytes()).hexdigest() != MD5:
            raise ValueError("本项已交付或原始数据身份改变")
        with TemporaryDirectory(dir=DEST) as temporary:
            deliver(Path(temporary))
