"""每个开发机队沿自己的合法装机偏差重发，并用去重历史给未来陌生机队冻结统一权重。"""
import json
import subprocess
import sys
import h5py
import joblib
import numpy as np
from score_flight_component import DEST as FEATURES, BASE, other_masks, VIEWS, dataset_path, file_hash, LocalStore, atomic_json, receipt
from score_flight_local_trials import DEST as PRIOR
from harness.models.flight_local_trials import GLOBAL
from score_flight_complement import LocalAcceptance, digest, measures
from verify_flight_segment_entry import ROOT, CLI
from engineering.cli.workflow.flight_input_predict import infer_component

DEST=BASE/"flight-component-entry"


def offset(data, masks):
    residuals=[]
    for s,mask in masks.items():
        y,p,g=data[s]
        residuals.extend(float((y[g==u]-p[g==u]).mean()) for u in np.unique(g[mask]))
    return float(np.mean(residuals))


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("该入参已从原始航班发放")
        previous=json.loads((PRIOR/"scores.json").read_bytes())
        meta=json.loads((FEATURES/"scores.json").read_bytes())
        if file_hash(FEATURES/"release.joblib")!=meta["发布权重SHA256"] or file_hash(FEATURES/"profiles.npz")!=meta["缓存SHA256"]:
            raise ValueError("训练权重或历史资料改变")
        component=joblib.load(FEATURES/"release.joblib")
        shared={"reader":component["model"].readouts[-1],"extractor":component["extractor"]}
        joblib.dump(shared,DEST/"component.joblib")
        shared_hash=file_hash(DEST/"component.joblib")
        data={};groups={};expected={};events={};results={}
        with np.load(FEATURES/"profiles.npz") as z:
            for s,v in previous["sites"].items():
                payload=receipt(PRIOR,s,v)
                if payload["ids"]!=z[s+"_ids"].tolist():raise ValueError("回执身份与校准资料不同")
                g=groups[s]=z[s+"_group"]
                data[s]=(z[s+"_y"],np.asarray(payload["predictions"][VIEWS[-1]]),g)
                expected[s]=payload
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        for s,payload in expected.items():
            correction=offset(data,other_masks(groups,s))
            model={**shared,"offset":correction}
            # 只保存一次共享森林；每折偏移和共享权重摘要进入交接回执。
            issued={}
            with h5py.File(dataset_path(s)) as f:
                aux=f["A_test"][:,:2]; bounds=np.r_[0,np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))+1,len(aux)]
                for a,b in zip(bounds[:-1],bounds[1:],strict=True):
                    req={"inputs":{"sensors":f["X_s_test"][a:b],"settings":f["W_test"][a:b]}}
                    key=tuple(map(int,aux[a]))
                    if key in issued:raise ValueError("航班重复")
                    issued[key]=next(infer_component(model,[req]))["outputs"]["value"]
            ids=payload["ids"]
            pred=[issued[tuple(k)] for k in ids]
            item={"ids":ids,"predictions":{GLOBAL:pred},"整机队装机修正":correction,"共享权重":shared_hash}
            key=digest([s,item])
            if not receiver.submit(key,item)["accepted"]:raise ValueError("未接收航班预测")
            actual=receipt(DEST,s,{"request":key})["predictions"][GLOBAL]
            np.testing.assert_allclose(actual,payload["predictions"][GLOBAL],rtol=0,atol=1e-8)
            events[s]=key
            results[s]=measures(data[s][0],np.asarray(actual),np.array(ids)[:,0])
        # 装机给未知设备时可用全部已结束的开发轨迹；尚无其寿命成绩。
        all_masks=other_masks({**groups,"未见":np.array([],dtype=int)},"未见")
        correction=offset(data,all_masks)
        release=DEST/"release.joblib";joblib.dump({**shared,"offset":correction},release)
        sha=file_hash(release)
        lines=[];check=[]
        for s in expected:
            with h5py.File(dataset_path(s)) as f:
                aux=f["A_test"][:,:2]
                ix=np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))[0]+1
                req={"inputs":{"sensors":f["X_s_test"][:ix].tolist(),"settings":f["W_test"][:ix].tolist()}}
            lines.append(json.dumps(req)); check.append(next(infer_component({**shared,"offset":correction},[req]))["outputs"]["value"])
        cmd=[sys.executable,"-B",str(CLI),"--artifact",str(release),"--sha256",sha,"--method","component-calibration"]
        run=subprocess.run(cmd,input="\n".join(lines)+"\n",capture_output=True,text=True,check=True,cwd=ROOT)
        external=[json.loads(v)["outputs"]["value"] for v in run.stdout.splitlines()]
        np.testing.assert_allclose(external,check,atol=1e-8,rtol=0)
        summary={"机队等权误差":float(np.mean([r["rmse"] for r in results.values()])),"航班":sum(len(v[0]) for v in data.values()),
                 "新装机偏移":correction,"校准轨迹":sum(len(set(groups[s][m])) for s,m in all_masks.items()),"外部入口样本":len(external)}
        atomic_json(DEST/"scores.json",{"summary":summary,"sites":results,"requests":events,"权重SHA256":sha,"共享权重SHA256":shared_hash})
        print(json.dumps({"部署复核":summary},ensure_ascii=False))


if __name__=="__main__":main()
