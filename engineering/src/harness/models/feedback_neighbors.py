"""将固定的历史设备参照与本轮实际选中的厂模预测组合，权重只由两份已回标资料决定。"""
import numpy as np

from harness.data.cmapss_input import condition_key
from harness.models.device_neighbors import DeviceNeighborUpdate, nearest


def fit_weight(base,delta,truth):
    norm=float(delta@delta)
    return float(np.clip(delta@(truth-base)/norm,0.,.5)) if norm else 0.


def route(explore,confirm,previous):
    """同一候选的开发对照。来自其它窗口的选择只用于约定允许延用的策略。"""
    if not explore or not confirm:
        return {"always":.5,"confirm":0.,"last":previous["last"],"weight":0.,"by-condition":previous["by-condition"]}
    grouped,weights,passes=[],[],[]
    for rows,base,ref in (explore,confirm):
        y=np.array([r["label"] for r in rows])
        change=ref-base
        parent_loss=(base-y)**2
        child_loss=(base+.5*change-y)**2
        weights.append(fit_weight(base,change,y))
        passes.append(child_loss.mean()<parent_loss.mean()-1e-12)
        keys=np.array([condition_key(row["values"][21:]) for row in rows])
        accepted=set()
        for group in np.unique(keys):
            inside=keys==group
            if len({r["unit_id"] for r,k in zip(rows,inside,strict=True) if k})>=5 and child_loss[inside].mean()<parent_loss[inside].mean()-1e-12:
                accepted.add(group)
        grouped.append(accepted)
    weight=.5 if all(passes) else 0.
    return {"always":.5,"confirm":weight,"last":weight,"weight":min(weights),"by-condition":sorted(grouped[0]&grouped[1])}


class CurrentNeighbors(DeviceNeighborUpdate):
    """复用同一训练库，各窗口显式传入在用模型，服务前已冻结该窗的系数。"""
    def observations(self,model,rows):
        if not rows:
            return None
        x=np.asarray([r["values"] for r in rows])
        q=np.clip((x[:,:21]-self.mean)/self.scale,-10,10)
        _,ref=nearest(self.bank,self.y,self.units,q,np.zeros(len(self.y)),np.zeros(len(q)))
        return rows,model.predict(x),ref

    def advance_current(self,model,explore,confirm,serve):
        ids=[r["sample_id"] for group in (explore,confirm,serve) for r in group]
        if not serve or len(set(ids))!=len(ids) or any(r["label"] is not None for r in serve):
            raise ValueError("服务标签或资料分组不正确")
        weights=[]
        if explore and confirm:
            for rows in (explore,confirm):
                _,base,ref=self.observations(model,rows)
                weights.append(fit_weight(base,ref-base,np.array([r["label"] for r in rows])))
        weight=min(weights,default=0.)
        if weight:
            _,base,ref=self.observations(model,serve)
            values=base+weight*(ref-base)
        else:
            base=model.predict(np.array([r["values"] for r in serve]))
            values=base
        if not np.isfinite(values).all():
            raise ValueError("发动机服务预测数值无效")
        ids=[r["sample_id"] for r in serve]
        return dict(zip(ids,map(float,values),strict=True)),dict(zip(ids,map(float,base),strict=True)),{
            "weight":weight,"explore_weight":weights[0] if weights else None,"confirm_weight":weights[1] if weights else None}
