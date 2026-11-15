"""冻结当窗两份可用标注的选择，再与十轮交付和独立服务答案配对。"""
from hashlib import sha256

import numpy as np

from gas_complementary_feedback import OUT,BASE,read,atomic_json
from check_feedback_controls import scores
from compare_calibration_expiry import source_window
from score_guarded_gas import public_stream
from harness.models.gas_confirmed_choice import MODES,candidates,choose
from harness.models.feedback_routing import apply_choices


def main():
    stream,truth,controls=public_stream()
    samples=read(BASE/"selected/gas/state.json")["samples"]
    predictions={m:{} for m in ("current",*MODES)}
    decisions=[]
    groups=[]
    for j,(keys,base,ids,feedback,answers) in enumerate(stream):
        explore,confirm,_=source_window(samples,j)
        if ids!=[r["sample_id"] for r in explore+confirm]:
            raise ValueError("当窗两份已见记录与分组信息不符")
        split=len(explore)
        pools=candidates(feedback)
        a={k:v[:split] for k,v in pools.items()}
        b={k:v[split:] for k,v in pools.items()}
        service=candidates(base)
        decisions.append({})
        groups.append(keys)
        for mode in predictions:
            choices=(["current"]*6 if mode=="current" else choose(mode,a,answers[:split],b,answers[split:]))
            decisions[-1][mode]=choices
            predictions[mode].update(zip(keys,apply_choices(service,choices).tolist(),strict=True))
    keys=[i for row in groups for i in row]
    folder=OUT/"current-window"
    issued=folder/"issued.json"
    atomic_json(issued,{"keys":groups,"decisions":decisions,"predictions":{**controls,**predictions}})
    y=np.array([truth[i] for i in keys])
    blocks=np.split(np.arange(len(y)),np.cumsum([len(g) for g in groups])[:-1])
    result={k:scores(y,np.asarray([p[i] for i in keys]),blocks,6) for k,p in read(issued)["predictions"].items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(issued.read_bytes()).hexdigest(),"arms":result})
    for k,v in result.items():
        print(k,round(v["pooled"],5),round(v["equal_batches"],5),round(v["worst_batch"],5),
              sum(a<b-1e-12 for a,b in zip(v["by_batch"],result["current"]["by_batch"],strict=True)))


if __name__=="__main__":
    main()
