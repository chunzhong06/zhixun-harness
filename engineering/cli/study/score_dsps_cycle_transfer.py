"""将周期内读数和同器件自身参照迁到DS-PS原三组开发暴露，只用当次目标阵列测量作答。"""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from score_dsps_response import SOURCE, DEST as PHYSICAL, read, LocalAcceptance, LocalStore, atomic_json, digest, tally

DEST = PHYSICAL.parent/"dsps-cycle-transfer"
PREVIOUS = PHYSICAL.parent/"dsps-concentration"
PRIMARY = "按当前周期一致性合用读法"


def acquire():
    meta=read(SOURCE/"source.json")
    hashes={r["成员"]:r["SHA256"] for r in meta["已核验成员"]}
    with np.load(SOURCE/"development.npz") as data:
        ids,group,y=[data[k].copy() for k in ("ids","repetition","y")]
    bank={}
    for key in ids:
        path=SOURCE/"members"/key
        if sha256(path.read_bytes()).hexdigest()!=hashes[key]:
            raise ValueError("开发原件身份改变")
        struct=loadmat(path,squeeze_me=True,struct_as_record=False)[path.stem]
        for device in (0,11):
            base, early, late = [getattr(struct,k)[:,128*device:128*(device+1)].reshape(-1,100,16,8) for k in ("Bl","Tr","St")]
            def phases(x):
                # 按作者100步循环保留顺序，数字为实测输出，不作电阻的物理量标定。
                return np.median(x.reshape(-1,20,5,16,8),axis=2)
            clean,trans,stable = map(phases,(base,early,late))
            steady=np.median(stable,axis=0)
            scale=np.maximum(abs(clean[0]),1e-4)
            ratio=np.arcsinh((steady-clean[0])/scale)
            drift=(np.median(trans[:2],axis=0)-steady)/scale
            for name,values in (("完整周期原读数",steady),("同器件空气相对周期",ratio),
                                ("进入与稳定期差",np.arcsinh(drift)),("重复周期波动",np.median(abs(stable-steady),axis=0)/scale)):
                centered=values-values.mean(axis=0,keepdims=True)
                pieces={name+"单负载":values[:,7,:],name+"多负载":values[:,[1,4,7,11,15],:],
                        name+"周期内相对":centered[:,7,:]}
                for feature,mat in pieces.items():bank.setdefault((feature,device),[]).append(mat.ravel())
    return ids,group,y,{k:np.stack(v) for k,v in bank.items()}


def prediction_bank(bank,tr,te,y):
    prob={}
    for feature,device in bank:
        if device!=11:continue
        target=bank[feature,11];old=bank[feature,0]
        for setting in ("目标固定","双器件合训"):
            fitx=target[tr] if setting=="目标固定" else np.vstack((target[tr],old[tr]))
            fity=y[tr] if setting=="目标固定" else np.tile(y[tr],2)
            for family in ("线性","核"):
                classifier=LogisticRegression(max_iter=800,C=.5) if family=="线性" else SVC(C=4,gamma="scale",probability=True,random_state=17)
                model=make_pipeline(StandardScaler(),classifier).fit(fitx,fity)
                prob[f"{feature}/{setting}/{family}"]=model.predict_proba(target[te])
    return prob


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("周期迁移已交付")
        ids,groups,y,bank=acquire();prior=read(PREVIOUS/"scores.json")
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        events=[];truth=[];columns={}
        parents=read(PHYSICAL/"scores.json")["requests"]
        for rep,oldkey,parent in zip(np.unique(groups),prior["requests"],parents,strict=True):
            tr=groups!=rep;te=~tr;prob=prediction_bank(bank,tr,te,y)
            held={k:[] for k in prob};valid_y=[]
            for g in np.unique(groups[tr]):
                mask=groups==g
                h=prediction_bank(bank,tr&~mask,mask,y)
                valid_y.extend(y[mask].tolist())
                for k,v in h.items():held[k].extend((1+v.argmax(1)).tolist())
            rates=np.array([f1_score(valid_y,held[k],average="macro",zero_division=0) for k in prob])
            best=list(prob)[int(np.argmax(rates))]
            table=np.stack(list(prob.values()),axis=1)
            confidence=table.max(2)-np.median(table,axis=2)
            w=(.01+rates[None,:])*np.maximum(.05,confidence)
            pred={k:1+v.argmax(1) for k,v in prob.items()}
            pred["固定均匀周期票"]=1+table.mean(1).argmax(1)
            pred["历史选择周期读法"]=pred[best]
            pred["历史固定周期权重"]=1+(table*(rates+.01)[None,:,None]).sum(1).argmax(1)
            pred[PRIMARY]=1+(table*w[:,:,None]).sum(1).argmax(1)
            old=read(PREVIOUS/"accepted"/(oldkey+".json"))["payload"]
            if digest([int(rep),parent,old])!=oldkey or old["ids"]!=ids[te].tolist():raise ValueError("旧交付身份不同")
            combined={**old["预测"],**{k:v.tolist() for k,v in pred.items()}}
            item={"ids":ids[te].tolist(),"预测":combined,"选中":best}
            key=digest([int(rep),item])
            if not receiver.submit(key,item)["accepted"]:raise ValueError("DS-PS开发交付失败")
            sent=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),sent])!=key:raise ValueError("DS-PS回执内容不同")
            events.append(key);truth.extend(y[te])
            for k,v in sent["预测"].items():columns.setdefault(k,[]).extend(v)
        scores={k:tally(np.array(truth),np.array(v)) for k,v in columns.items()}
        fixed=max([prior["decision"]["最强固定"],*list(prob),"固定均匀周期票","历史固定周期权重","历史选择周期读法"],key=lambda k:scores[k]["F1"])
        # 保留基于相关校准论文的直接及分段标准化，以及已实跑的全部强固定读法。
        paper=max(("直接标准化","分段直接标准化"),key=lambda k:scores[k]["F1"])
        gap=scores[PRIMARY]["F1"]-max(scores[fixed]["F1"],scores[paper]["F1"])
        d={"现场":PRIMARY,"固定":fixed,"成熟校准":paper,"最小领先":gap,"进独立":gap>=.01 and scores[PRIMARY]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"requests":events,"scores":scores,"decision":d})
        print(json.dumps({"周期迁移":{k:scores[k] for k in (PRIMARY,fixed,paper)},"判定":d},ensure_ascii=False))


if __name__=="__main__":main()
