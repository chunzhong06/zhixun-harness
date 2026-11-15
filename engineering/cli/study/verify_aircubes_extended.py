"""独立进程读取同一台仪器的七列原始测量，重建训练与校正的实际得分。"""
import json
import numpy as np
import pandas as pd
from verify_aircubes_entry import run_site,DATA,known,RAW,LocalAcceptance,LocalStore,atomic_json
from score_aircubes_bias import DEST as PRIOR,PRIMARY
from harness.models.aircubes_joint import AUX,PRIMARY as JOINT
from harness.models.aircubes_innovation import PRIMARY as TRANSFER
from harness.models.aircubes_retrain import PRIMARY as RETRAIN

DEST=PRIOR.parent/"aircubes-extended-entry"


def verify(source_dir,destination,primary,additional):
    meta=json.loads((DATA/"source.json").read_bytes())
    source=json.loads((source_dir/"scores.json").read_bytes())
    ref=pd.read_csv(known("reference-first.csv",meta),usecols=["timestamp","NO2_ppb"]).set_index("timestamp")["NO2_ppb"]
    receiver=LocalAcceptance();receiver.output=destination/"accepted";decision=source["decision"]
    rules=tuple(dict.fromkeys((primary,decision["固定"],decision["论文"],*additional)))
    reports={site:run_site(site,meta,source,ref,receiver,source_dir,(*RAW,*AUX),rules) for site in source["requests"]}
    means={n:float(np.mean([v["scores"][n] for v in reports.values()])) for n in rules}
    if any(abs(v-source["means"][n])>1e-7 for n,v in means.items()):raise ValueError("完整入口的分数与实际选案不符")
    summary={"实发":sum(r["实发"] for r in reports.values()),"可评分":sum(r["可评分"] for r in reports.values()),
         "已回标":sum(r["已回标"] for r in reports.values()),"现场误差":means[primary],"固定误差":means[decision["固定"]],
         "论文误差":means[decision["论文"]],"最大复现差":max(r["最大复现差值"] for r in reports.values())}
    atomic_json(destination/"scores.json",{"summary":summary,"sites":reports,"means":means,"源模型SHA256":source["源模型SHA256"]})
    print(json.dumps({"原始共址七通道的部署":summary},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已完成原始入口复测")
        verify(PRIOR,DEST,PRIMARY,(RETRAIN,TRANSFER,JOINT,"固定现场配对记忆0.95",
               "论文强校正与仅原两电极重训0.82各半","论文强校正与加入同机NO读数重训0.82各半","冻结漂移模型初始配比与论文"))
