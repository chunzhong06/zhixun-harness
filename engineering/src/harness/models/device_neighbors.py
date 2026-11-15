"""按各历史设备最接近的观测估计发动机寿命，一台历史设备最多一票。"""
from hashlib import sha256
from pathlib import Path

import numpy as np

from harness.contracts import digest


def nearest(bank,targets,units,query,keys,query_keys):
    obs,distinct=np.empty(len(query)),np.empty(len(query))
    for group in np.unique(query_keys):
        src=np.flatnonzero(keys==group)
        dest=np.flatnonzero(query_keys==group)
        ordered=src[np.argsort(units[src],kind="stable")]
        starts=np.r_[0,1+np.flatnonzero(units[ordered][1:]!=units[ordered][:-1])]
        if len(starts)<5:
            raise ValueError("工况内训练设备不足五台")
        bank_x=bank[ordered]
        norms=(bank_x**2).sum(axis=1)
        for begin in range(0,len(dest),128):
            indices=dest[begin:begin+128]
            dist=norms[None,:]-2*query[indices]@bank_x.T
            # 采用稳定排序以固定等距离观测的选择；按设备聚合时每台只贡献一个测点。
            closest=np.argsort(dist,axis=1,kind="stable")[:,:5]
            obs[indices]=targets[ordered[closest]].mean(axis=1)
            group_dist=np.minimum.reduceat(dist,starts,axis=1)
            chosen=np.argsort(group_dist,axis=1,kind="stable")[:,:5]
            predictions=np.zeros(len(indices))
            for col in range(5):
                for device in np.unique(chosen[:,col]):
                    sel=np.flatnonzero(chosen[:,col]==device)
                    left=starts[device]
                    right=starts[device+1] if device+1<len(starts) else len(ordered)
                    best=left+dist[sel,left:right].argmin(axis=1)
                    predictions[sel]+=targets[ordered[best]]/5
            distinct[indices]=predictions
    return obs,distinct


class DeviceNeighborUpdate:
    """固定训练参照和已交接父版，双重确认通过时启用。接续身份含全部拟合数据。"""
    def __init__(self,model,parent_sha,training):
        x=np.asarray([r.values for r in training])
        if x.ndim!=2 or x.shape[1]!=24 or any(r.role!="fit" or r.label is None for r in training):
            raise ValueError("发动机候选只能使用带标签的官方训练行")
        self.model=model
        self.mean=x[:,:21].mean(0)
        self.scale=np.maximum(x[:,:21].std(0),1e-12)
        self.bank=np.clip((x[:,:21]-self.mean)/self.scale,-10,10)
        self.y=np.array([r.label for r in training])
        self.units=np.array([r.unit_id for r in training])
        if len(set(self.units))<5:
            raise ValueError("参照不足五台不同的训练设备")
        self.identity=digest([sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha,
                              [(r.sample_id,r.unit_id,r.values,r.label) for r in training]])

    def predict(self,rows,use_reference):
        data=np.asarray([r["values"] for r in rows])
        base=self.model.predict(data)
        if not use_reference:
            return base
        query=np.clip((data[:,:21]-self.mean)/self.scale,-10,10)
        _,remaining=nearest(self.bank,self.y,self.units,query,np.zeros(len(self.y)),np.zeros(len(query)))
        return .5*(base+remaining)

    def advance(self,state,batch,explore,confirm,serve):
        if state and state["identity"]!=self.identity:
            raise ValueError("旧决定属于其他父版或训练参照")
        ids=[r["sample_id"] for group in (explore,confirm,serve) for r in group]
        if len(set(ids))!=len(ids) or not serve or any(r["label"] is not None for r in serve):
            raise ValueError("探索、确认、服务编号重叠或服务输入含答案")
        live=state.get("layer")
        event={"batch":batch,"proposed":False,"adopted":False}
        if live is None and explore and confirm:
            event["proposed"]=True
            for name,rows in (("explore",explore),("confirm",confirm)):
                truth=np.array([r["label"] for r in rows])
                a,b=[float(np.sqrt(np.mean((self.predict(rows,enabled)-truth)**2))) for enabled in (False,True)]
                event[name]=[a,b]
                if b>=a-1e-12:
                    break
            else:
                live={"origin":batch,"bank_identity":self.identity}
                event["adopted"]=True
        event["using"]=live["origin"] if live else None
        predictions=dict(zip((r["sample_id"] for r in serve),map(float,self.predict(serve,live is not None)),strict=True))
        return {"identity":self.identity,"layer":live},event,predictions


