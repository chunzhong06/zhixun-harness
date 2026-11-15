"""逐一留出完整机队，用另外五组开发航班的预测回执和寿命验证修正方案。"""
import json
import joblib
import numpy as np
from score_flight_masked_readings import BASE, DEST as PRIOR, LocalStore, atomic_json, file_hash, fixed_mix, receipt, issue, PAPER
from harness.models.flight_sensor_trials import PRIMARY, SWITCH, FIXED_BIAS, MASK_BIAS, ANCHOR, JOINT, ROBUST, TrialCorrection

DEST=BASE/"flight-sensor-trials"
MIX="三读法与论文同证据固定配比"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("同一协议已运行")
        old=json.loads((PRIOR/"scores.json").read_bytes())
        if file_hash(PRIOR/"profiles.npz")!=old["缓存SHA256"] or file_hash(PRIOR/"release.joblib")!=old["发布权重SHA256"]:
            raise ValueError("遮路航班或模型身份改变")
        model=joblib.load(PRIOR/"release.joblib")["readouts"]
        cohorts={}
        with np.load(PRIOR/"profiles.npz") as z:
            for s,item in old["sites"].items():
                ids,x,y=(z[f"开发_{s}_{k}"] for k in ("ids","x","y"))
                sent=receipt(PRIOR,s,item)
                if ids.tolist()!=sent["ids"]:raise ValueError("开发机队身份不同")
                pred=sent["predictions"]
                b=np.column_stack([pred[k] for k in (ANCHOR,JOINT,ROBUST)])
                c=model.recovery.errors(x)
                ctx=np.column_stack((np.abs(c).mean(1),np.sqrt((c**2).mean(axis=(1,2)))))
                cohorts[s]=(ids,b,ctx,y,pred)
        scores={}; models={}
        for s,(ids,b,c,truth,pred) in cohorts.items():
            others=[k for k in cohorts if k!=s]
            m,ctx,y=[np.concatenate([cohorts[k][i] for k in others]) for i in (1,2,3)]
            tags=np.concatenate([np.repeat(k,len(cohorts[k][3])) for k in others])
            policy=TrialCorrection().fit(m,ctx,y,tags)
            models[s]=policy
            updated={**pred,**policy.predict(b,c)}
            bank=np.column_stack((m,np.concatenate([cohorts[k][4][PAPER] for k in others])))
            weights=fixed_mix(bank,y,tags)
            updated[MIX]=np.column_stack((b,pred[PAPER]))@weights
            scores[s]=issue(DEST,s,ids.tolist(),updated,truth)
        means={k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in updated}
        controls=[old["decision"]["最强固定"],old["decision"]["同工具论文"],FIXED_BIAS,MASK_BIAS,
                  "全部诊断固定校准","固定校准与原版各半",MIX]
        fixed=min(controls,key=means.get)
        paper=min((PAPER,MIX,old["decision"]["同工具论文"]),key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],
           "论文误差":means[paper],"降幅":gain,"进独立":gain>=.01}
        bundle=DEST/"folds.joblib";joblib.dump(models,bundle)
        atomic_json(DEST/"scores.json",{"decision":d,"sites":scores,"means":means,"留组权重SHA256":file_hash(bundle)})
        print(json.dumps({"开发交付":d,"只学净改判":means[SWITCH],"固定分段校准":means[FIXED_BIAS]},ensure_ascii=False))


if __name__=="__main__":main()
