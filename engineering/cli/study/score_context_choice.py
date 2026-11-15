"""用已见伦敦候选比较可见读数是否足以辨别下一条应信任谁。"""
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from harness.models.context_choice import select
from harness.models.sparse_reference import correct
from harness.models.station_residual import error
from harness.runtime.storage import LocalStore,atomic_json


def main():
    base=ROOT/"results/experiments/local-defaults/whole-maintenance"
    original=base/"london-adaptive-recovery"
    out=base/"london-visible-choice"
    archive=ROOT/"data/public/london-air-2024/stations.zip"
    previous=json.loads((original/"scores.json").read_bytes())
    scene.configure(previous["start"])
    tools=scene.prepare(archive,nonlinear=True)
    with LocalStore(out).exclusive():
        if (out/"scores.json").exists():raise ValueError("不可覆盖已见候选的选择比较")
        scores={}
        for site in previous["counts"]:
            truth=scene.readings(archive,site).to_numpy()
            y,fit,windows,service,returned=scene.roles(truth)
            clock=np.arange(np.flatnonzero(windows[0])[0],len(y))
            known=np.concatenate((y[clock[clock<service[0]]],returned))
            valid=np.isfinite(truth[service])
            for arm in ("project","paper","numeric"):
                packet=joblib.load(original/site/(arm+"-adaptive.joblib"))
                ids=[joblib.hash(pair) for pair in packet["models"]]
                unique=list(dict.fromkeys(ids))
                kept=[ids.index(key) for key in unique]
                initial=unique.index(ids[packet["initial"]])
                columns=[correct(tools.issue(m,clock)[:,None],known,clock,s)[:,0] for m,s in
                         (packet["models"][i] for i in kept)]
                pred=np.column_stack(columns)
                contexts={"predictions":np.empty((len(clock),0)),
                          "readings":tools.x[clock,:len(scene.DONORS)+2],
                          "readings-clock":tools.x[clock,:len(scene.DONORS)+2+len(scene.SEASON)]}
                emitted={}
                for memory in (7,28):
                    for rule in ("global-hard","global-soft","tree-hard","tree-soft","tree-half-global"):
                        names=("predictions",) if rule.startswith("global") else contexts
                        for context_name in names:
                            output,choices,journal=select(pred,contexts[context_name],known,clock,service[0],initial,rule,memory)
                            key=f"{rule}/{context_name}/{memory}"
                            issued=output[clock>=service[0]]
                            scores.setdefault(arm+"/"+key,{})[site]=error(truth[service][valid],issued[valid])
                            emitted[key]=issued
                            if any(v["last_reference"]>=v["publish_hour"] for v in journal):
                                raise ValueError("误差模型使用了未到期答案")
                dest=out/site
                dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/(arm+"-issued.npz"),**emitted)
                print(f"{site} {arm} 已完成",flush=True)
        means={key:float(np.mean(list(v.values()))) for key,v in scores.items()}
        best={arm:min((k for k in means if k.startswith(arm+"/")),key=means.get)
              for arm in ("project","paper","numeric")}
        result={"scope":"已见2024伦敦，当前模型和全部候选保留，选择只学习到期参考",
                "station_rmse":scores,"macro_rmse":means,"best":best}
        atomic_json(out/"scores.json",result)
        print(json.dumps({arm:{"setting":key,"rmse":means[key]} for arm,key in best.items()},ensure_ascii=False))


if __name__=="__main__":main()
