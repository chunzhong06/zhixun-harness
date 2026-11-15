"""根据单条目标测量与两份当地反馈的振动距离确认常数参照；不访问目标答案。"""
import numpy as np


def sq_dist(q,bank):
    d=np.sum(q*q,axis=1)[:,None]+np.sum(bank*bank,axis=1)-2*q@bank.T
    return np.maximum(d,0.)


def local_medians(model,median,explore,confirm,serve,whole):
    options={}
    query=model.normalized(np.asarray([r["values"] for r in serve]))
    base=model.predict(np.asarray([r["values"] for r in serve]))
    data=[]
    for rows in (explore,confirm):
        if len(rows)<3:
            return {k:base.copy() for k in ("nearby","distance","both")}
        x=np.asarray([r["values"] for r in rows])
        y=np.asarray([r["label"] for r in rows])
        data.append((model.normalized(x),(median-y)**2-(model.predict(x)-y)**2))
    use_loss=np.ones(len(query),dtype=bool)
    covered=np.ones(len(query),dtype=bool)
    for j,(bank,delta) in enumerate(data):
        in_source=sq_dist(data[1-j][0],bank).min(axis=1)
        limit=np.quantile(in_source,.95,method="higher")
        distance=sq_dist(query,bank)
        near=np.argsort(distance,axis=1,kind="stable")[:,:5]
        covered &= distance.min(axis=1)<=limit+1e-12
        use_loss &= delta[near].mean(axis=1)<-1e-12
    for k,use in (("nearby",use_loss),("distance",covered & whole),("both",covered & use_loss)):
        options[k]=np.where(use,median,base)
    return options
