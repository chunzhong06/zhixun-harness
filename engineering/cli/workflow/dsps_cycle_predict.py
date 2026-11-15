"""一行输入一次目标器件暴露的Bl/Tr/St，输出冻结权重给出的三类预测。"""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import sys
import joblib

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.models.dsps_cycle_release import FIXED,PRIMARY,target_features
from harness.models.dsps_dose_contrast import EQUAL, TRIAL
from harness.models.dsps_cycle_trials import EXPANDED, PRIMARY as CYCLE_TRIAL, cycles


def predict(model,record,method):
    x=record["inputs"]
    if set(x)!={"Bl","Tr","St"}:
        raise ValueError("推理只接受三个原始测量阶段")
    features=target_features(x["Bl"],x["Tr"],x["St"])[None,:,:]
    if method in (EXPANDED,CYCLE_TRIAL):
        decisions,_=model.predict(cycles(x["St"])[None,:,:])
        return {"outputs":{"class":int(decisions[method][0])}}
    if method=="direct-calibration":
        return {"outputs":{"class":int(model.predict_class(features)[0])}}
    output,_=model.predict(features)
    return {"outputs":{"class":int(output[method][0])}}


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--artifact",type=Path,required=True);cli.add_argument("--sha256",required=True)
    cli.add_argument("--method",choices=("fixed","adaptive","direct-calibration","dose-equal","install-trials",
                                         "cycle-average","cycle-trials"),default="fixed")
    args=cli.parse_args()
    with args.artifact.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=args.sha256:raise ValueError("目标权重身份不一致")
        f.seek(0);model=joblib.load(f)
    for line in sys.stdin:
        method={"fixed":FIXED,"adaptive":PRIMARY,"direct-calibration":"direct-calibration",
                "dose-equal":EQUAL,"install-trials":TRIAL,"cycle-average":EXPANDED,"cycle-trials":CYCLE_TRIAL}[args.method]
        if line.strip(): print(json.dumps(predict(model,json.loads(line),method)),flush=True)
