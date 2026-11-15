"""用已安装气体芯片的权重从原始传感器周期预测浓度和类别。"""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.models.year_gas_calibration import feature_cycles
from harness.models.year_sensor_pairs import PRIMARY
from harness.models.mixture_response import PRIMARY as COMPOSITION


def load(path,identity):
    with path.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=identity:
            raise ValueError("装机权重身份与清单不同")
        f.seek(0)
        return joblib.load(f)


def predict(args):
    with args.release.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=args.sha256:
            raise ValueError("发布清单身份不同")
        f.seek(0)
        release = json.load(f)
    folder = args.release.parent
    installed,_ = load(folder/release["旧权重"],release["旧摘要"])
    mapped,selector = load(folder/release["配对权重"],release["配对摘要"])
    with np.load(args.input,allow_pickle=False) as src:
        if set(src.files)!={"group","cycles"}:raise ValueError("服务包只接收周期和分组编号")
        groups = src["group"]
        if groups.ndim!=1 or groups.dtype.kind not in "iu" or len(groups)!=len(src["cycles"]) or not len(groups):
            raise ValueError("原始周期与混合物分组不同")
        features = feature_cycles(src["cycles"])
        ids = list(dict.fromkeys(groups.tolist()))
        x = np.stack([features[groups==i].mean(0) for i in ids])
    old = installed.outputs(x)
    corrected,dist = mapped.predict(x)
    table = np.column_stack(((old[:,3]+old[:,4])/2,corrected,(old[:,5]+old[:,4])/2))
    outputs = selector.predict(table,dist)
    # 固定端也知道获得的标气数量。两条原始训练路径使用完全相同的装机反馈。
    general = outputs["统一装机验证配对权重"]
    curves = corrected[:,3]
    budget_fixed = general if release["装机数"]==20 else curves
    forecasts = {"现场":outputs[PRIMARY],"统一":general,"全曲线":curves,"按预算固定":budget_fixed}
    if "配方权重" in release:
        models,selection = load(folder/release["配方权重"],release["配方摘要"])
        candidates = np.column_stack((budget_fixed,*[model.predict(x) for model in models]))
        forecasts.update({"配方现场":selection.predict(candidates)[COMPOSITION],
                          "配方固定":(budget_fixed+candidates[:,1])/2,
                          "配方论文":(curves+candidates[:,2])/2})
        if "目标气体" in release:
            # 这两项装机元数据在服务前已知，部署清单固定对应规则。
            gas,size = release["目标气体"],release["装机数"]
            if gas=="acetone":
                preset = forecasts["配方固定"] if size==20 else candidates[:,1]
                paper = table[:,-1] if size==20 else (curves+candidates[:,1])/2
            elif gas=="toluene":
                paper = (curves+candidates[:,3])/2
                preset = paper if size==20 else candidates[:,3]
            else:
                raise ValueError("该发布清单未登记目标气体")
            forecasts.update({"按气体与预算固定":preset,"按气体与预算论文":paper})
    return {"ids":ids,"浓度":{k:v.tolist() for k,v in forecasts.items()},
            "分类":{k:(1+(v>release["阈值"])).tolist() for k,v in forecasts.items()}}


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--sha256",required=True)
    parser.add_argument("--input",type=Path,required=True)
    print(json.dumps(predict(parser.parse_args()),allow_nan=False))
