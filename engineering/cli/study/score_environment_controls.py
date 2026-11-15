"""同一已发预测开放两种补偿的全部十分位配比，先交接固定输出再比较现场改判。"""
import json
import numpy as np
from sklearn.metrics import f1_score

from score_vapor_environment import DATA,DEST as PRIOR,LocalAcceptance,LocalStore,atomic_json,digest,read_flight,score,PRIMARY,FIXED_PAPER
from harness.models.vapor_environment import AUTHOR,JOINT,FIXED,fixed_grid

DEST=PRIOR.parent/"vapor-environment-controls"


def run():
    prior=json.loads((PRIOR/"scores.json").read_bytes())
    meta=json.loads((DATA/"source.json").read_bytes())
    cut=json.loads((PRIOR.parent/"vapor-field/scores.json").read_bytes())["阈值伏"]
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    sequences={}
    for site,keys in prior["requests"].items():
        seq=[]
        for key in keys:
            p=json.loads((PRIOR/"accepted"/(key+".json")).read_bytes())["payload"]
            if digest(p)!=key:
                raise ValueError("前序原始温湿服务记录改变")
            old=p["阈值倍数"]
            bank=np.column_stack([old[n] for n in (FIXED_PAPER,AUTHOR,JOINT)])
            vals={n:v.tolist() for n,v in fixed_grid(bank).items()}
            vals.update({k:old[k] for k in (PRIMARY,FIXED,FIXED_PAPER)})
            payload={"上次温湿回执":key,"窗口":p["窗口"],"阈值倍数":vals}
            req=digest(payload)
            if not receiver.submit(req,payload)["accepted"]:
                raise ValueError("同工具固定配比发放失败")
            seq.append(req)
        sequences[site]=seq
    labels=[];columns={};by_site={};sizes=[];reports={}
    for site,seq in sequences.items():
        _,y,_=read_flight(int(site),meta)
        yy=y[20:]/cut>=1
        rows=[];cols={}
        for req in seq:
            p=json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
            if digest(p)!=req:
                raise ValueError("固定预测在交接后改变")
            rows.extend(p["窗口"])
            for n,v in p["阈值倍数"].items():
                cols.setdefault(n,[]).extend(v)
        if rows!=list(range(20,len(y))):
            raise ValueError("服务窗口没有完整覆盖")
        by_site[site]=cols;sizes.append(len(yy))
        reports[site]=score(yy,cols);labels.extend(yy)
        for n,v in cols.items():
            columns.setdefault(n,[]).extend(v)
    truth=np.array(labels);means=score(truth,columns)
    fixed=max((n for n in means if n!=PRIMARY),key=lambda n:means[n]["F1"])
    best=np.array(columns[fixed])>=1;candidate=np.array(columns[PRIMARY])>=1
    # 赢家照各类召回与宏F1直接算；以下事后开关只衡量这个工具库在何种尺度有空间。
    oracles={}
    for width in ("整场","十窗","单窗"):
        output=[];pos=0
        for site,length in zip(sequences,sizes,strict=True):
            bank=np.column_stack([v for n,v in by_site[site].items() if n!=PRIMARY])>=1
            target=truth[pos:pos+length];pos+=length
            step=length if width=="整场" else (10 if width=="十窗" else 1)
            for i in range(0,length,step):
                yy=target[i:i+step];b=bank[i:i+step]
                if width=="整场":
                    names=[n for n in by_site[site] if n!=PRIMARY]
                    pick=max(range(len(names)),key=lambda j:reports[site][names[j]]["F1"])
                else:
                    # 以整日正负样本的逆频率作为局部择优代价；不充作可部署分数。
                    w=np.where(yy,1/max(1,truth.sum()),1/max(1,(~truth).sum()))
                    pick=np.argmin(w@(b!=yy[:,None]))
                output.extend(b[:,pick])
        oracles[width]=float(f1_score(truth,output,average="macro"))
    d={"现场":PRIMARY,"固定":fixed,"论文":fixed,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
       "论文F1":means[fixed]["F1"],"差值":means[PRIMARY]["F1"]-means[fixed]["F1"],
       "改对":int(((candidate==truth)&(best!=truth)).sum()),"改错":int(((candidate!=truth)&(best==truth)).sum())}
    d["进入独立"]=d["差值"]>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":reports,"requests":sequences,
        "窗口":len(truth),"各尺度事后择优":oracles})
    print(json.dumps({"温湿补偿公平配比":d,"各尺度事后择优":oracles,"窗口":len(truth)},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("相同温湿固定配比已完成")
        run()
