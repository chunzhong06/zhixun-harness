"""在相同实发读数上交付各档固定混合及零更新对照，检验现场学习的净作用。"""
import json
import numpy as np
from sklearn.metrics import f1_score

from score_vapor_waveform import DEST as PRIOR, DATA, LocalAcceptance, LocalStore, atomic_json, digest, load_verified, read_flight, score
from harness.models.vapor_waveform import FIXED_PAPER, NAMES, PRIMARY, WaveMix

DEST=PRIOR.parent/"vapor-wave-controls"


def run():
    tools,meta=load_verified(PRIOR)
    manifest=json.loads((DATA/"source.json").read_bytes())
    receiver=LocalAcceptance(); receiver.output=DEST/"accepted"
    requests={}; per_flight={}; pooled={}; all_y=[]; gain={}; oracle=[]
    for site,sequence in meta["requests"].items():
        frozen=WaveMix()
        issued=[]
        for key in sequence:
            pkt=json.loads((PRIOR/"accepted"/(key+".json")).read_bytes())["payload"]
            if digest(pkt)!=key:
                raise ValueError("原始入口实发记录不符")
            old=pkt["阈值倍数"]
            out={PRIMARY:old[PRIMARY],FIXED_PAPER:old[FIXED_PAPER]}
            paper=np.arcsinh(old[FIXED_PAPER])
            for n in NAMES[:3]:
                v=np.arcsinh(old[n])
                for k in range(1,11):
                    out[f"{n}占{k}成"]=np.sinh(paper+(v-paper)*k/10).tolist()
            out["冻结现场初始权重"]=[frozen.predict(p,t,r)[PRIMARY] for p,t,r in zip(old[FIXED_PAPER],
                np.arcsinh(old[NAMES[0]]),np.arcsinh(old[NAMES[1]]),strict=True)]
            payload={"上轮":key,"窗口":pkt["窗口"],"输出":out}
            h=digest(payload)
            if not receiver.submit(h,payload)["accepted"]:
                raise ValueError("固定混合对照未交付")
            issued.append(h)
        requests[site]=issued
    # 先发完所有确定性变换，再取得开发标签。在此前的原始进程中已逐窗核验双方输入。
    for site,sequence in requests.items():
        _,y,_=read_flight(int(site),manifest)
        labels=y[20:]/tools["cut"]>=1
        cols={};positions=[]
        for h in sequence:
            pkt=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
            if digest(pkt)!=h:
                raise ValueError("固定读法在评分时变化")
            positions.extend(pkt["窗口"])
            for n,values in pkt["输出"].items():
                cols.setdefault(n,[]).extend(values)
        if positions!=list(range(20,len(y))):
            raise ValueError("实际发放窗口不完整")
        report=score(labels,cols);per_flight[site]=report
        best=max((n for n in cols if n!=PRIMARY),key=lambda n:report[n]["F1"])
        oracle.extend(np.array(cols[best])>=1)
        for n,v in cols.items():
            pooled.setdefault(n,[]).extend(v)
        all_y.extend(labels)
    truth=np.array(all_y); means=score(truth,pooled)
    fixed=max((n for n in means if n!=PRIMARY),key=lambda n:means[n]["F1"])
    best=np.array(pooled[fixed])>=1
    for n in (PRIMARY,"冻结现场初始权重",FIXED_PAPER):
        cand=np.array(pooled[n])>=1
        gain[n]={"比最强固定改对":int(np.sum((cand==truth)&(best!=truth))),
                 "比最强固定改错":int(np.sum((cand!=truth)&(best==truth)))}
    # 各档混合除十成纯波形外均显式包含论文读法。
    paper=max((n for n in means if n!=PRIMARY and not n.endswith("占10成")),key=lambda n:means[n]["F1"])
    d={"现场":PRIMARY,"现场F1":means[PRIMARY]["F1"],"固定":fixed,"固定F1":means[fixed]["F1"],
       "论文":paper,"论文F1":means[paper]["F1"],"冻结初始F1":means["冻结现场初始权重"]["F1"],
       "差值":means[PRIMARY]["F1"]-max(means[n]["F1"] for n in (fixed,paper))}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":per_flight,"requests":requests,
        "窗口":len(truth),"得失":gain,"事后逐场选固定的F1":float(f1_score(truth,oracle,average="macro"))})
    print(json.dumps({"高频读法同工具固定对照":d,"配对得失":gain,"总窗数":len(truth)},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("同工具比例已核对")
        run()
