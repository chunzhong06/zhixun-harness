"""独立子进程拟合并重载生成模型；输入中只有合法训练标签，预测逐行进行。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.integrations.generated_regressor import load_module,check_model


def main():
    ap=ArgumentParser();ap.add_argument("--source",required=True,type=Path);ap.add_argument("--data",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--reuse-model",type=Path,help="只重载既有权重逐行交付；不再训练")
    args=ap.parse_args()
    module=load_module(args.source)
    with np.load(args.data,allow_pickle=False) as z:
        x,ids=z["predict_x"],z["predict_ids"]
        if args.reuse_model:
            fallback,limit=z["fallback"],float(z["prediction_limit"])
        else:train_x,train_y=z["train_x"],z["train_y"]
    if args.reuse_model:model=joblib.load(args.reuse_model)
    else:
        model=module.build_model();check_model(model);model.fit(train_x,train_y)
        joblib.dump(model,args.output/"model.joblib");model=joblib.load(args.output/"model.joblib")
    # 每次只传当前一行，生成的transform不能使用整段未来输入计算统计量。
    predictions=[];raw=[];failed=[]
    for index,row in enumerate(x):
        try:
            value=np.asarray(model.predict(row[None,:]),dtype=float)
            if value.shape!=(1,) or not np.isfinite(value).all():raise ValueError("预测须为一维有限数")
            number=float(value[0]);raw.append(number)
            if args.reuse_model and abs(number)>limit:raise ValueError("当前预测超出训练量级上限")
            predictions.append(max(0.,number))
        except Exception as exc:
            if not args.reuse_model:raise
            # 只在当前输入失败时回退，后续输入的失败不追改已经发出的预测。
            if len(raw)==index:raw.append(float("nan"))
            predictions.append(float(fallback[index]));failed.append({"id":int(ids[index]),"error":str(exc)})
    np.savez_compressed(args.output/"base.npz",ids=ids,prediction=predictions,raw_prediction=raw)
    (args.output/"prediction-failures.json").write_text(json.dumps(failed,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"训练参考":0 if args.reuse_model else len(train_y),"逐行预测":len(predictions),"当前输入回退":len(failed)},ensure_ascii=False))


if __name__=="__main__":main()
