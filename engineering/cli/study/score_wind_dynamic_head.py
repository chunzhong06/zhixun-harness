"""源期训练接收连续历史的A1网络，再在每台设备的相同已返功率上微调。"""
import json
from score_wind_field import run,FIELDS,LocalStore
from score_wind_history import load_previous
from score_wind_transition import DEST as PRIOR
from harness.models.wind_dynamic_head import DynamicHeadRelease,train_network,PRIMARY

DEST = PRIOR.parent/"wind-dynamic-head"

def train(rows):
    net = train_network(rows,FIELDS)
    print(json.dumps({"连续风况论文预训练":{"源训练条数":len(rows),"真实选择轮数":net.epochs}},ensure_ascii=False))
    return DynamicHeadRelease(load_previous(PRIOR,rows),net)


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("历史概率模型已交接")
        run(DEST,train,None,PRIMARY,PRIOR)
