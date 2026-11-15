"""从源日权重启动风洞监测。标准输入依次接收启动校准、单窗的三路波形及延迟回标。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from harness.models.vapor_correction import VaporService
from harness.models.vapor_field import voltage_ratio
from harness.models.vapor_waveform import FIXED_PAPER, Waveform, WaveMix


def waveform(value):
    points = np.asarray(value,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)==0:
        raise ValueError("必须提供半秒内的三个慢通道原始采样")
    return points


def serve(path, checksum):
    if sha256(path.read_bytes()).hexdigest()!=checksum:
        raise ValueError("已冻结的装机权重身份不符")
    tools=joblib.load(path)
    session = None
    wave_reader=tools.get("wave")
    environmental=tools.get("environment")
    airflow=tools.get("airflow")
    warm_keys={"启动波形","启动参考"}
    measure_keys={"窗口","慢通道"}
    if wave_reader is not None:
        warm_keys.add("启动时钟")
        measure_keys.add("时钟")
    if environmental is not None:
        warm_keys.add("启动温湿度")
        measure_keys.add("温湿度")
    if airflow is not None:
        warm_keys.update(("启动气流","启动时钟"))
        measure_keys.update(("气流","时钟"))
    for line in sys.stdin:
        req = json.loads(line)
        if session is None and set(req)==warm_keys:
            warm = [waveform(a) for a in req["启动波形"]]
            pid = [np.asarray(a,dtype=float) for a in req["启动参考"]]
            if len(warm)!=20 or len(pid)!=20 or any(p.shape!=(len(a),) or not np.isfinite(p).all()
                    for a,p in zip(warm,pid,strict=True)):
                raise ValueError("启动校准需十秒同步实测")
            session=VaporService(tools,warm,np.median(np.concatenate(pid)),pid[-1].mean())
            if environmental is not None:
                environmental.begin(warm,req["启动温湿度"],tools["scale"])
            if airflow is not None:
                airflow.begin(warm,req["启动气流"],req["启动时钟"],tools["scale"])
            if wave_reader is not None:
                encoder=Waveform(np.median(voltage_ratio(np.vstack(warm)),axis=0),tools["scale"])
                for values,clock in zip(warm,req["启动时钟"],strict=True):
                    encoder.next(values,clock)
                mixture=WaveMix()
            answer={"下个窗口":20}
        elif session is not None and set(req)==measure_keys and req["窗口"]==session.position:
            values=waveform(req["慢通道"])
            pos,out=session.measure(values)
            if environmental is not None:
                out.update(environmental.measure(values,req["温湿度"],out[FIXED_PAPER]))
            if airflow is not None:
                out.update(airflow.measure(values,req["气流"],req["时钟"],out["固定温湿比例0-1"]))
            if wave_reader is not None:
                feature=encoder.next(values,req["时钟"])[None,:]
                tree,ridge=wave_reader.predict(feature)
                out.update(mixture.predict(out[FIXED_PAPER],tree[0],ridge[0]))
            answer={"窗口":pos,"预测":out}
        elif session is not None and set(req)=={"回标窗","参考电压"}:
            session.feedback(req["回标窗"],req["参考电压"])
            if environmental is not None:
                environmental.feedback(session.anchor)
            if airflow is not None:
                airflow.feedback(session.anchor)
            if wave_reader is not None:
                wave_reader.update(feature,np.array([session.anchor]))
                mixture.update(session.anchor)
            answer={"下个窗口":session.position}
        else:
            raise ValueError("请求身份、时序或字段不符")
        print(json.dumps(answer,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    arg=ArgumentParser(description=__doc__)
    arg.add_argument("--release",required=True,type=Path)
    arg.add_argument("--sha256",required=True)
    opts=arg.parse_args()
    serve(opts.release,opts.sha256)
