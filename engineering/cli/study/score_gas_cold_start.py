"""保留推荐作品的首批输出，再比较同起点的现场学习器与固定控制。"""
from hashlib import sha256
import json

import numpy as np
import sklearn

from score_gas_field_kernel import evaluate
from score_gas_field_families import fitter
from score_gas_probability_fusion import (ROOT,BASE,ARCHIVE,SEEDS,stream,read_official_gas,
    GasFeedbackBank,LocalStore,atomic_json)

DEST=BASE/"whole-maintenance/gas-cold-start"
PROTOCOL=ROOT/"engineering/configs/experiments/gas-cold-start.json"
ARMS=("ridge-source",)


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有完整对照，请复用")
        if sklearn.__version__!="1.7.2":raise ValueError("岭模型实现版本不同")
        pkg=read_official_gas(ARCHIVE);batches=stream(pkg,GasFeedbackBank(pkg.train))
        runs=[evaluate(s,pkg,batches,destination=DEST,protocol=PROTOCOL,arms=ARMS,
            fitter=fitter(DEST,ARMS),shared_cold_start=True,control_cold_start="observer") for s in SEEDS]
        mean={a:{k:float(np.mean([r["scores"][a][k] for r in runs])) for k in ("macro","worst")}
              for a in runs[0]["scores"]};default=runs[0]["scores"]
        controls=("pooled-source","pooled-separate","fixed-midway","fixed-midway-linear-cold")
        best=max(controls,key=lambda a:mean[a]["macro"])
        protected=[a for a in (*ARMS,*controls) if mean[a]["worst"]>=mean["observer"]["worst"]-.02
            and default[a]["worst"]>=default["observer"]["worst"]-.02
            and default[a]["macro"]>default["observer"]["macro"]]
        selected=max(protected,key=lambda a:mean[a]["macro"]) if protected else "observer"
        ridge_pass=("ridge-source" in protected and mean["ridge-source"]["macro"]>mean[best]["macro"])
        result={"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"means":mean,
            "strongest_control":best,"protected":protected,"selected":selected,"ridge_beats_fixed":ridge_pass,
            "advances":selected!="observer" and mean[selected]["macro"]>mean["fixed-midway-linear-cold"]["macro"],
            "development_only":True,"seeds":list(SEEDS),"fit_cpu_seconds":sum(r["fit_cpu_seconds"] for r in runs)}
        atomic_json(DEST/"scores.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
