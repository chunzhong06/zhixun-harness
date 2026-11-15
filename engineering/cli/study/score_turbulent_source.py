"""不同物理释放先确认分组，以源内交叉验证封存模型和阈值，再向完整开发暴露交付。"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import sys

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"engineering/src"),str(ROOT/"engineering/cli/data")]
from prepare_turbulent_mixture import DEST as DATA,split_members
from harness.contracts import digest
from harness.runtime.storage import LocalStore, atomic_json
from harness.runtime.handoff import LocalAcceptance

DEST=ROOT/"results/experiments/local-defaults/whole-maintenance/turbulent-source"


def sampled(raw):
    a=np.loadtxt(BytesIO(raw),delimiter=",")
    t=a[:,0]
    # 3.0 至 300.0 秒含两端共2971点；论文同时写了2961点，采用其明确的时间范围。
    grid=np.arange(30,3001)/10
    hi=np.searchsorted(t,grid).clip(0,len(t)-1)
    lo=np.maximum(0,hi-1)
    pick=np.where(abs(t[lo]-grid)<=abs(t[hi]-grid),lo,hi)
    return a[pick,3:].ravel()


def paper(alpha_kernel, y, cost):
    """二类μ=1/2时，两条决策函数互为相反数。无截距对偶为带盒约束的一个二次问题。"""
    target=np.where(y==2,1.,-1.)
    q=alpha_kernel*target[:,None]*target[None,:]
    def loss(a):
        gradient=q@a
        return .5*a@gradient-a.sum(),gradient-1
    fit=minimize(loss,np.zeros(len(y)),jac=True,method="L-BFGS-B",bounds=[(0,cost)]*len(y),
                 options={"maxiter":2000,"ftol":1e-12,"gtol":1e-7})
    if not fit.success:
        raise ValueError(f"论文适配未收敛：{fit.message}")
    return fit.x*target


def run():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已经打开此开发划分")
        meta=json.loads((DATA/"source.json").read_bytes())
        rawfile=DATA/"uci309.zip"
        if sha256(rawfile.read_bytes()).hexdigest()!=meta["原包SHA256"] or sha256((DATA/"development.json").read_bytes()).hexdigest()!=meta["开发文件SHA256"]:
            raise ValueError("源端归档或开发输入身份改变")
        records=json.loads((DATA/"development.json").read_bytes())
        x=np.array([r["values"] for s in ("source","development") for r in records[s]])
        train_ids=[r["id"] for r in records["source"]]
        ids=[r["id"] for r in records["development"]]
        if set(train_ids+ids)&set(meta["封存编号"]):
            raise ValueError("研究输入占用了盲测暴露")
        seq=[]
        with ZipFile(rawfile) as z:
            names=split_members(z.infolist())
            for s in ("source","development"):
                for (uid,member,_),row in zip(names[s],records[s],strict=True):
                    raw=z.read(member)
                    if uid!=row["id"] or sha256(raw).hexdigest()!=row["原件SHA256"]:
                        raise ValueError("论文当前读取的原件与特征输入错位")
                    seq.append(sampled(raw))
        seq=np.stack(seq)
        y=np.array([r["label"] for r in records["source"]])
        n=len(y); folds=list(StratifiedKFold(n_splits=10,shuffle=True,random_state=17).split(x[:n],y))
        distance=cdist(seq,seq[:n],metric="sqeuclidean")/seq.shape[1]
        bandwidth=np.median(distance[:n][np.triu_indices(n,1)])
        columns,source_predictions,model_table={},{},{}
        candidates=[]
        for gamma in (.25,1,4):
            kernel=np.exp(-gamma*distance/bandwidth)
            for cost in (1,10,100):
                held=np.empty(n)
                for tr,te in folds:
                    held[te]=kernel[np.ix_(te,tr)]@paper(kernel[np.ix_(tr,tr)],y[tr],cost)
                key=(f1_score(y,1+(held>=0),average="macro"),-cost,-gamma)
                candidates.append((key,gamma,cost,held))
        _,g,c,held=max(candidates,key=lambda v:v[0])
        kernel=np.exp(-g*distance/bandwidth)
        dual=paper(kernel[:n],y,c)
        columns["论文全时序抑制核"]=kernel[n:]@dual
        source_predictions["论文全时序抑制核"]=held
        model_table["论文全时序抑制核"]={"gamma":g,"cost":c,"source_waveforms":seq[:n],"bandwidth":bandwidth,"dual":dual}
        for view,idx in (("平均",np.arange(8)),("全部",np.arange(32)),("前后差",np.arange(16,32))):
            for family in ("线性","核"):
                name=view+family
                def fit(indices):
                    estimator=(LogisticRegression(C=1,class_weight="balanced",max_iter=1000) if family=="线性"
                               else SVC(C=4,gamma="scale",class_weight="balanced"))
                    return make_pipeline(StandardScaler(),estimator).fit(x[np.ix_(indices,idx)],y[indices])
                v=np.empty(n)
                for tr,te in folds:
                    v[te]=fit(tr).decision_function(x[np.ix_(te,idx)])
                estimator=fit(np.arange(n))
                columns[name]=estimator.decision_function(x[n:,idx])
                source_predictions[name]=v
                model_table[name]=(idx,estimator)
        names=list(columns)
        z=np.column_stack(list(source_predictions.values()))
        q=np.column_stack(list(columns.values()))
        strength=np.maximum(.1,np.median(abs(z),axis=0))
        z,q=expit(z/strength),expit(q/strength)
        correct=(z>=.5)==(y[:,None]==2)
        fixed_weights=correct.mean(0)+.001
        fixed_weights/=fixed_weights.sum()
        source_margin=np.abs(z-.5)
        threshold=np.median(source_margin,axis=0)
        strong=source_margin>=threshold
        base=correct.mean(0)
        hi=((correct*strong).sum(0)+5*base)/(strong.sum(0)+5)
        lo=((correct*~strong).sum(0)+5*base)/((~strong).sum(0)+5)
        local=np.where(abs(q-.5)>=threshold,hi,lo)
        choices=local.argmax(1)
        pred={k:(1+(v>=0)).tolist() for k,v in columns.items()}
        pred.update({"固定等权":(1+(q.mean(axis=1)>=.5)).tolist(),"固定源端权重":(1+(q@fixed_weights>=.5)).tolist(),
                     "固定报有":np.full(len(ids),2).tolist(),"固定报无":np.ones(len(ids),dtype=int).tolist(),
                     "按当前把握选读法":(1+(q[np.arange(len(q)),choices]>=.5)).tolist(),
                     "按当前把握组合":(1+(np.sum(q*local**4,axis=1)>=.5*np.sum(local**4,axis=1))).tolist()})
        bundle=DEST/"models.joblib"
        joblib.dump({"models":model_table,"order":names,"strength":strength,"fixed_weights":fixed_weights,
                     "threshold":threshold,"hi":hi,"lo":lo},bundle)
        payload={"ids":ids,"预测":pred,"权重SHA256":sha256(bundle.read_bytes()).hexdigest(),"训练数":n}
        key=digest([meta["协议SHA256"],payload])
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        if not receiver.submit(key,payload)["accepted"]:
            raise ValueError("乙烯辨识开发预测未交接")
        sent=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
        if digest([meta["协议SHA256"],sent])!=key:
            raise ValueError("乙烯辨识回执身份改变")
        yy=np.array([r["label"] for r in records["development"]])
        scores={k:{"F1":float(f1_score(yy,v,labels=[1,2],average="macro")),
                   "最差召回":float(recall_score(yy,v,labels=[1,2],average=None).min()),"正确":int(np.sum(yy==v))} for k,v in sent["预测"].items()}
        dynamic=("按当前把握选读法","按当前把握组合")
        best=max(dynamic,key=lambda k:scores[k]["F1"])
        fixed=max((k for k in scores if k not in (*dynamic,"论文全时序抑制核")),key=lambda k:scores[k]["F1"])
        gain=scores[best]["F1"]-max(scores[fixed]["F1"],scores["论文全时序抑制核"]["F1"])
        decision={"候选":best,"固定":fixed,"对两对手的最小领先":gain,"可进独立":gain>=.01 and scores[best]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"request":key,"scores":scores,"decision":decision,"开发":True})
        print(json.dumps({"开发实测":scores,"判定":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()
