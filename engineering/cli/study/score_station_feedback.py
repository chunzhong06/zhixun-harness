"""在已开分的北京任务中，从实际发出的基础预测比较每六小时一次的在线校正。"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow.station_field_driver import table
from harness.models.sparse_reference import SETTINGS, correct, name
from harness.runtime.storage import LocalStore, atomic_json


def run():
    base=ROOT/"results/experiments/local-defaults/whole-maintenance"
    target=base/"station-reference-development"
    archive=ROOT/"data/public/beijing-multisite/stations.zip"
    with LocalStore(target).exclusive():
        if (target/"scores.json").exists():
            raise ValueError("开发回放不可覆盖")
        summary={}
        for run_id in ("station-luna-transfer","station-history-forward"):
            old=base/run_id
            meta=json.loads((old/"scores.json").read_bytes())
            start=pd.Timestamp(meta.get("start","2016-09-01"))
            totals={}
            for site in meta["counts"]:
                with np.load(old/site/"issued.npz",allow_pickle=False) as packet:
                    ids=packet["rows"]
                    keys=[k for k in packet.files if k!="rows"]
                    pred=np.column_stack([packet[k] for k in keys])
                truth=table(archive,site,["PM2.5"],start).iloc[ids,0].to_numpy()
                known=np.where(ids%6==0,truth,np.nan)
                for setting in SETTINGS:
                    emitted=correct(pred,known,ids,setting)
                    errors=np.sqrt(np.nanmean((emitted-truth[:,None])**2,axis=0))
                    for k,value in zip(keys,errors,strict=True):
                        totals.setdefault((k,name(setting)),[]).append(float(value))
            means={k:np.mean(v) for k,v in totals.items()}
            choices={}
            for arm in ("project","paper","fixed"):
                allowed=[k for k in totals if (k[0].startswith(("fixed-","raw-")) or k[0]=="static")
                         ] if arm=="fixed" else [k for k in totals if k[0]==arm]
                best=min(allowed,key=means.get)
                keep=min((k for k in allowed if k[1]==name(SETTINGS[0])),key=means.get)
                choices[arm]={"base":best[0],"setting":best[1],"with":means[best],"without":means[keep],
                              "per_site":totals[best]}
            summary[run_id]=choices
        atomic_json(target/"scores.json",summary)
        print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    run()
