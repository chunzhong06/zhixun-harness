"""在仅有源期的逐段真实试验上训练固定及现场组合，然后按全部原开发时点部署。"""
from score_aircubes_bias import DEST as PRIOR,extend,evaluate,LocalStore,ADAPTIVE,PREVIOUS,JOINT,PRIMARY as DRIFT
from harness.models.aircubes_stack import StackRelease,StackMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-stack"


def train(tools,t,x,y):
    tools["stack"]=StackRelease().fit(t,x,y)
    return tools


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("前向组合已有发放")
        evaluate(DEST,PRIOR,train,StackMaintenance,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,DRIFT,PRIMARY,"后验漂移模型"},extend)
