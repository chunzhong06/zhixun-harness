"""在已封存的轴承预测上分解误差，并用事后选版估计重新选择候选能达到的范围。"""
from datetime import datetime

import numpy as np

from compare_error_slice import BASE, ROOT, read, reference_folders
from compare_group_objectives import delivered, remaining_minutes
from compare_trajectory_training import error_parts
from harness.runtime.storage import atomic_json
from score_unseen_bearing_windows import FEMTO, clock


def summarize(predictions, truth, units):
    ids=sorted(truth)
    if any(set(stream)!=set(ids) for stream in predictions.values()):
        raise ValueError("配对诊断需要相同的服务编号")
    sample_groups={i:{"unit_id":units[i]} for i in ids}
    report={a:error_parts(p,sample_groups,truth) for a,p in predictions.items()}
    y=np.array([truth[i] for i in ids])
    xs=np.array([[p[i] for i in ids] for p in predictions.values()])
    err=(xs-y)**2
    u=np.asarray([units[i] for i in ids])
    groups=np.unique(u)
    per_device={k:err[:,u==k].sum(axis=1) for k in groups}
    return {"arms":report,"devices":len(groups),"rows":len(ids),
            "best_one_arm":float(np.sqrt(err.mean(axis=1).min())),
            "hindsight_per_device":float(np.sqrt(sum(loss.min() for loss in per_device.values())/len(ids))),
            "hindsight_per_row":float(np.sqrt(err.min(axis=0).mean())),
            "oracle_arms":list(predictions)}


def main():
    source=BASE/"readout-live"
    fixed,_=reference_folders("pronostia","mature")
    groups=read(fixed/"state.json")["samples"]
    streams={k:delivered(folder) for k,folder in (("old",fixed),("bias",source/"pronostia"),("simple",fixed.parent/"simple"))}
    truth=remaining_minutes("full-test-manifest.json")
    out={"original_windows":summarize(streams,{i:truth[i] for i in streams["old"]},{i:r["unit_id"] for i,r in groups.items()})}
    streams=read(source/"fresh-windows"/"issued.json")["predictions"]
    old=read(FEMTO/"full-test-manifest.json")
    ends={r["unit"]:r["end"] for r in old}
    truth,units={},{}
    for i in streams["parent"]:
        _,unit,step=i.split("/")
        root=FEMTO/"Full_Test_Set"/unit
        truth[i]=(clock(root/f"acc_{ends[unit]:05d}.csv")-clock(root/f"acc_{int(step):05d}.csv"))%86400/60
        units[i]=unit
    out["new_windows"]=summarize({"old":streams["parent"],"bias":streams["bias"],"simple":streams["simple"]},truth,units)
    streams=read(source/"transfer"/"bearing"/"issued.json")["predictions"]
    rows=read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
    end=datetime.fromisoformat(rows[-1]["time"])
    truth={"KAIST/"+r["file"]:(end-datetime.fromisoformat(r["time"])).total_seconds()/60 for r in rows if "features" in r}
    out["kaist"]=summarize({"old":streams["fixed"],"bias":streams["readout-bias"],"simple":streams["simple"]},
                           truth,dict.fromkeys(truth,"KAIST"))
    atomic_json(source/"error-parts.json",out)
    for k,v in out.items():
        print(k,{a:round(p["bias_fraction_mse"],3) for a,p in v["arms"].items()},
              [round(v[f],2) for f in ("best_one_arm","hindsight_per_device","hindsight_per_row")])


if __name__=="__main__":
    main()
