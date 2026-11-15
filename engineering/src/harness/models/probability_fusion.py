"""合并完整类别概率；残差校正只学习此前已发放、随后返回答案的预测。

依据 Dirichlet calibration 的 log 概率—仿射映射，扩展到多个专家。
基准几何平均保持显式，正则约束新增系数，避免少量回标直接抹去源模型先验。
"""
from time import process_time

import numpy as np
from scipy.optimize import minimize


def softmax(x):
    exp = np.exp(x - np.max(x,axis=1,keepdims=True))
    return exp / exp.sum(axis=1,keepdims=True)


def evidence(probabilities, hard=False):
    p = np.asarray(probabilities,dtype=float)
    if p.ndim!=3 or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(2)<=0):
        raise ValueError("需要每行、每个专家的完整有限类别概率")
    p = p/p.sum(2,keepdims=True)
    if hard:
        c=p.shape[2]
        chosen=p.argmax(2)
        p=np.full_like(p,.1/(c-1))
        np.put_along_axis(p,chosen[:,:,None],.9,axis=2)
    logs = np.log(np.clip(p,1e-6,1.))
    logs -= logs.mean(2,keepdims=True)
    return logs.reshape(len(p),-1),logs.mean(1)


def observer_probabilities(router, votes):
    """混淆表的归一化分数，平票结果与既有离散入口逐条一致。"""
    counts=router.counts+1+np.eye(router.classes,dtype=int)[None,:,:]
    logp=np.log(counts/counts.sum(2,keepdims=True))
    values=np.stack([votes[k] for k in router.names]).astype(int)-1
    logits=sum(logp[j].T[v] for j,v in enumerate(values))
    p=softmax(logits)
    chosen=router.predict(votes,"uniform")-1
    for j,k in enumerate(chosen):
        if p[j].argmax()!=k:
            p[j,k]=np.nextafter(p[j].max(),np.inf)
    return p


def objective(theta, x, base, y, weights, penalty=.1):
    classes=base.shape[1]
    w=theta.reshape(classes,-1)
    logits=base+x@w[:,:-1].T+w[:,-1]
    p=softmax(logits)
    normal=logits.max(1)+np.log(np.exp(logits-logits.max(1,keepdims=True)).sum(1))
    loss=float(np.sum(weights*(normal-logits[np.arange(len(y)),y]))+.5*penalty*np.sum(w*w))
    p[np.arange(len(y)),y]-=1
    delta=p*weights[:,None]
    gradient=np.column_stack((delta.T@x,delta.sum(0)))+penalty*w
    return loss,gradient.ravel()


class ResidualCalibration:
    def __init__(self, classes, hard=False, state=None):
        self.classes,self.hard=classes,hard
        saved=state or {}
        self.history=saved.get("history",[])
        self.prior=saved.get("prior",[])
        self.answers=saved.get("answers",[])
        if not len(self.history)==len(self.prior)==len(self.answers):
            raise ValueError("概率校正的恢复状态与回标数不同")
        self._fit=None
        self.cpu=0.

    def fit(self, width):
        if self._fit is not None:
            return self._fit
        theta=np.zeros(self.classes*(width+1))
        if self.answers:
            x=np.asarray(self.history);base=np.asarray(self.prior);y=np.asarray(self.answers,dtype=int)-1
            _,inverse,counts=np.unique(y,return_inverse=True,return_counts=True)
            weights=1./(len(counts)*counts[inverse])
            start=process_time()
            result=minimize(objective,theta,args=(x,base,y,weights),jac=True,method="L-BFGS-B",
                            options={"maxiter":200,"ftol":1e-10,"gtol":1e-6})
            self.cpu+=process_time()-start
            if not result.success or not np.isfinite(result.x).all():
                raise ValueError(f"残差概率校正未收敛：{result.message}")
            theta=result.x
        self._fit=theta.reshape(self.classes,width+1)
        return self._fit

    def predict(self, probabilities):
        x,base=evidence(probabilities,self.hard)
        w=self.fit(x.shape[1])
        return softmax(base+x@w[:,:-1].T+w[:,-1])

    def update(self, probabilities, answers):
        """调用者须先确认该批输出，再把同批回标与当时概率一起交入。"""
        x,base=evidence(probabilities,self.hard)
        y=np.asarray(answers,dtype=int)
        if len(y)!=len(x) or np.any((y<1)|(y>self.classes)):
            raise ValueError("回标类别或行数无效")
        self.history.extend(x.tolist());self.prior.extend(base.tolist());self.answers.extend(y.tolist())
        self._fit=None

    def state(self):
        return {"history":self.history,"prior":self.prior,"answers":self.answers}
