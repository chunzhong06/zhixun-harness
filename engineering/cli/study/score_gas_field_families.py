"""同冷启动与延迟回标下比较现场学习器；先导筛选与开发复核分别封存。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from time import process_time

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from score_gas_field_kernel import evaluate
from score_gas_probability_fusion import (ROOT,BASE,ARCHIVE,SEEDS,stream,read,read_official_gas,
    GasFeedbackBank,LocalStore,atomic_json)
from harness.models.field_classification import weights
from harness.models.training_tools import SklearnBackend

DEST=BASE/"whole-maintenance/gas-field-families/histogram"
PROTOCOL=ROOT/"engineering/configs/experiments/gas-field-families.json"
ARMS=tuple(f"{family}-{scale}" for family in ("ridge","extra","boost") for scale in ("source","separate"))


def estimator(family):
    if family=="ridge":return RidgeClassifier(alpha=1.,solver="svd")
    if family=="extra":return ExtraTreesClassifier(n_estimators=64,max_features="sqrt",random_state=17,n_jobs=1)
    if family=="boost":return HistGradientBoostingClassifier(max_iter=64,max_leaf_nodes=15,max_bins=32,
        min_samples_leaf=10,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=17)
    raise ValueError("未知学习器家族")


def fitter(destination,arms):
    def fit(source,sy,history,answers,folder):
        """仅用过去回标；源/现场分别变换后合训，部署流水线保留现场标准化器。"""
        if not history:raise ValueError("无现场标签时应使用共享线性冷启动")
        source_scale=StandardScaler().fit(source);source_z=source_scale.transform(source)
        local_scale=StandardScaler().fit(np.asfortranarray(history))
        y=np.r_[sy,answers];w=np.r_[weights(sy,len(sy)),weights(np.asarray(answers),len(sy))]
        backend=SklearnBackend(execution_enabled=True);models={};identity={};cost=0.
        for arm in arms:
            family,scale=arm.split("-")
            scaler=source_scale if scale=="source" else local_scale
            x=np.vstack((source_z,scaler.transform(np.asfortranarray(history))))
            model=estimator(family);start=process_time();model.fit(x,y,sample_weight=w);cost+=process_time()-start
            pipeline=Pipeline([("scale",scaler),("model",model)])
            blob=backend.dump(pipeline);path=folder/(arm+".joblib");path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as out:out.write(blob)
            restored=backend.load(path.read_bytes());query=np.asarray(history[-3:])
            np.testing.assert_array_equal(restored.predict(query),pipeline.predict(query))
            models[arm]=restored;identity[arm]={"sha256":sha256(blob).hexdigest(),
                "file":path.relative_to(destination).as_posix(),"fit_rows":len(y),"weight_mass":float(w.sum())}
        return models,identity,cost
    return fit


def main():
    ap=ArgumentParser();ap.add_argument("stage",choices=("pilot","confirm"));args=ap.parse_args()
    destination=DEST/args.stage
    with LocalStore(destination/"lock").exclusive():
        if (destination/"scores.json").exists():raise ValueError("阶段已完成，请复用")
        if sklearn.__version__!="1.7.2":raise ValueError("学习器实现版本变化")
        if args.stage=="pilot":arms=ARMS;seeds=SEEDS[:3]
        else:
            prior=read(DEST/"pilot/scores.json");arms=tuple(prior["screened"]);seeds=SEEDS
            if not arms:raise ValueError("先导没有可复核家族")
            if prior["protocol_sha256"]!=sha256(PROTOCOL.read_bytes()).hexdigest():raise ValueError("先导协议已变化")
        pkg=read_official_gas(ARCHIVE);batches=stream(pkg,GasFeedbackBank(pkg.train))
        runs=[evaluate(s,pkg,batches,destination=destination,protocol=PROTOCOL,arms=arms,
                       fitter=fitter(destination,arms),shared_cold_start=True) for s in seeds]
        mean={a:{k:float(np.mean([r["scores"][a][k] for r in runs])) for k in ("macro","worst")}
              for a in runs[0]["scores"]};default=runs[0]["scores"]
        screened=[a for a in arms if mean[a]["macro"]>=mean["fixed-midway"]["macro"]-.005
            and mean[a]["worst"]>=mean["observer"]["worst"]-.02
            and default[a]["worst"]>=default["observer"]["worst"]-.02
            and default[a]["macro"]>default["observer"]["macro"]]
        eligible=[a for a in arms if mean[a]["macro"]>mean["fixed-midway"]["macro"]
            and default[a]["macro"]>default["observer"]["macro"]
            and mean[a]["worst"]>=mean["observer"]["worst"]-.02
            and default[a]["worst"]>=default["observer"]["worst"]-.02]
        best=max(arms,key=lambda a:mean[a]["macro"])
        result={"stage":args.stage,"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"means":mean,
            "best":best,"screened":screened,"eligible":eligible,"advances":bool(screened if args.stage=="pilot" else eligible),
            "fit_cpu_seconds":sum(r["fit_cpu_seconds"] for r in runs),"development_only":True,"seeds":list(seeds)}
        atomic_json(destination/"scores.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
