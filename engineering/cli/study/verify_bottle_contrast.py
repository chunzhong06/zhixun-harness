"""从七芯片第一年原始文件核验每种购瓶顺序、装机权重和79瓶服务浓度。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from verify_bottle_install import run, LocalStore, digest
from score_bottle_contrast import DEST as LATEST, ACTIVE, PRIMARY
from score_bottle_challenge import DEST as BALANCED, ACTIVE as BALANCED_RULE, PRIMARY as BALANCED_RELEASE
from score_bottle_alignment import DEST as ALIGNED, PRIOR as ORIGINAL

DEST = LATEST.parent/"bottle-contrast-entry"


def packet(folder, key, req):
    data = json.loads((folder/"accepted"/(req+".json")).read_bytes())["payload"]
    h = digest(data) if folder==ORIGINAL else digest([key,data])
    if h!=req:
        raise ValueError("旧版购瓶回执身份变化")
    return data


def load_plan():
    stages = (ORIGINAL, ALIGNED, BALANCED, LATEST)
    reports = [json.loads((s/"scores.json").read_bytes()) for s in stages]

    def plan(key, latest, old):
        suffix = key.split("-",1)[1]
        balanced = packet(BALANCED,key,old["上轮"])
        if reports[2]["requests"][key]!=old["上轮"]:
            raise ValueError("前次装机回执与当前权重不符")
        recipes = {"现场":(ACTIVE,PRIMARY,old,PRIMARY),
                   "保留量程现场":(BALANCED_RULE,BALANCED_RELEASE,balanced,BALANCED_RELEASE)}
        for role, choices in latest["controls"].items():
            rule, recorded = choices[suffix]
            method = recorded
            stage = 3
            while method in ("上轮最强固定","上轮最强论文","前次最强固定","前次最强论文"):
                stage = 0 if method.startswith("上轮") else stage-1
                previous_rule, method = reports[stage]["controls"][method[-2:]][suffix]
                if previous_rule != rule:
                    raise ValueError("实际购瓶方法与历史对照不符")
            recipes[role] = (rule,method,old,recorded)
        return recipes
    return plan


if __name__ == "__main__":
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as folder:
        if (DEST/"scores.json").exists():
            raise ValueError("已完成从原始标气复测")
        run(Path(folder),LATEST,DEST,load_plan(),lambda k,p:digest([k,p]))
