"""依次交付赛方权重候选并核对回执，评分时才打开服务答案。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np

from compare_calibration_expiry import source_window
from compare_fixed_tree_references import OUT,paths
from compare_group_objectives import ROOT,official,read,summary,remaining_cycles,remaining_minutes,atomic_json
from engineering.cli.workflow.serve_life_checkpoint import run
from harness.contracts import digest


def score_cohort(task,name):
    pkg,_,_=official(task)
    source,_=paths(name,task)
    base=OUT/"tree-hidden"/task
    out=base/"live"/name
    regions=read(source/"state.json")["samples"]
    strong=read(OUT/"routing"/name/"issued.json")["predictions"]
    baseline=read(base/name/"issued.json")["predictions"]
    predictions={k:strong[k] for k in ("current","fixed","paper","median","tree-confirm","median-confirm","catboost-only")}
    predictions["fixed-native"]=baseline["whole-teacher"]
    weights={"tree-native":base/"models/whole-teacher.npz","constant-native":OUT/"constant-weights"/task/"models/median.npz",
             "old-readout":OUT/"teacher-readout"/task/"models/teacher-ridge.npz"}
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    checks={}
    for arm,path in weights.items():
        work=out/arm
        if not (work/"run.json").exists():
            run(task,archive,path,source,work,4,True)
        result=run(task,archive,path,source,work,10,True)
        current=predictions["current"]
        fixed=baseline["whole-teacher" if arm=="tree-native" else "teacher-ridge"] if arm!="constant-native" else strong["median"]
        issued={}
        decisions=[]
        for j,entry in enumerate(result["accepted"]):
            rec=read(work/"accepted"/(entry["request"]+".json"))
            payload=rec["payload"]
            if digest([result["identity"],payload])!=entry["request"] or rec["request_id"]!=entry["request"]:
                raise ValueError("当前权重交接回执身份不符")
            actual=payload["predictions"]
            _,_,serve=source_window(regions,j)
            if set(actual)!={r["sample_id"] for r in serve} or issued.keys()&actual.keys():
                raise ValueError("当前批次的服务记录不符")
            allow=len(payload["checks"])==2 and all(pair[1]+1e-12<pair[0] for pair in payload["checks"])
            expected=fixed if allow else current
            if (("candidate" if allow else "current")!=payload["chosen"] or
                max(abs(actual[k]-expected[k]) for k in actual)>1e-8):
                raise ValueError("交付预测与独立固定模型或旧版不符")
            decisions.append(payload["chosen"])
            issued.update(actual)
        if arm=="constant-native" and max(abs(v-strong["median-confirm"][k]) for k,v in issued.items())>1e-8:
            raise ValueError("同样的常数候选未复现已知双确认运行")
        predictions[arm]=issued
        checks[arm]={"rounds":decisions,"sha256":sha256((work/"run.json").read_bytes()).hexdigest()}
    sealed=out/"issued.json"
    atomic_json(sealed,{"predictions":predictions,"checks":checks})
    saved=read(sealed)["predictions"]
    answers=({r.sample_id:r.label for r in pkg.validation} if name==task else
             (remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")))
    groups={k:r["unit_id"] for k,r in regions.items() if r["role"]=="serve"}
    table=summary(task,answers,{"parent":saved["current"],**saved},groups)
    atomic_json(out/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table,"checks":checks})
    print(name,{k:(round(table[k]["score"],2),round(table[k]["worst_group"],2),checks[k]["rounds"].count("candidate")) for k in weights},flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("bearing","engine"))
    task=parser.parse_args().task
    for name in (task,"phm08" if task=="engine" else "pronostia"):
        score_cohort(task,name)
