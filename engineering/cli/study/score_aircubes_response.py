"""给浓度假设预测两只电极的响应，只在已返标签上更新其标定模型。"""
from score_aircubes_stack import DEST as PRIOR,evaluate,extend,ADAPTIVE,PREVIOUS,JOINT,DRIFT,PRIMARY as STACK,LocalStore
from harness.models.aircubes_response import ResponseRelease,ResponseMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-response"


def train(tools,t,x,y):
    tools["response"]=ResponseRelease().fit(t,x,y,tools["joint"].scale)
    return tools


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("电极响应反演已经交付")
        evaluate(DEST,PRIOR,train,ResponseMaintenance,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,DRIFT,STACK,PRIMARY,"后验漂移模型"},extend)
