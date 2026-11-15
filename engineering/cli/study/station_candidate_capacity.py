"""用封存候选核对修错空间。使用终局答案的理想选择只作为开发上界。"""
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from harness.models.sparse_reference import correct
from harness.models.station_residual import error
from harness.runtime.storage import LocalStore,atomic_json


def main():
    base=ROOT/"results/experiments/local-defaults/whole-maintenance"
    out=base/"london-candidate-capacity"
    original=base/"london-adaptive-recovery"
    archive=ROOT/"data/public/london-air-2024/stations.zip"
    old=json.loads((original/"scores.json").read_bytes())
    scene.configure(old["start"])
    tools=scene.prepare(archive,nonlinear=True)
    with LocalStore(out).exclusive():
        if (out/"scores.json").exists():raise ValueError("不可覆盖候选空间诊断")
        sites={}
        for site in old["counts"]:
            truth=scene.readings(archive,site).to_numpy()
            y,fit,windows,service,returned=scene.roles(truth)
            clock=np.arange(np.flatnonzero(windows[2])[0],len(y))
            known=np.concatenate((y[clock[clock<service[0]]],returned))
            packet=joblib.load(original/site/"project-adaptive.joblib")
            candidates=np.column_stack([correct(tools.issue(m,clock)[:,None],known,clock,s)[:,0]
                                        for m,s in packet["models"]])[clock>=service[0]]
            valid=np.isfinite(truth[service])
            losses=(candidates[valid]-truth[service][valid,None])**2
            standalone=np.sqrt(losses.mean(axis=0))
            sites[site]={"actual":old["station_rmse"]["project"][site],
                         "best_fixed":old["station_rmse"][old["best_fixed"]][site],
                         "best_candidate_after_answers":float(standalone.min()),
                         "best_hourly_choice_after_answers":float(np.sqrt(losses.min(axis=1).mean())),
                         "candidate_rmse":standalone.tolist(),
                         "max_pred_spread":float(np.max(np.ptp(candidates[valid],axis=1)))}
        means={k:float(np.mean([v[k] for v in sites.values()])) for k in
               ("actual","best_fixed","best_candidate_after_answers","best_hourly_choice_after_answers")}
        atomic_json(out/"scores.json",{"scope":"答案已见的候选能力上界；理想选择使用终局答案，不能作为实际方法分数",
                    "sites":sites,"mean":means})
        print(json.dumps(means,ensure_ascii=False))


if __name__=="__main__":main()
