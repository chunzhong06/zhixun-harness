"""由历史整机轨迹预测给定剩余航班数应出现的部件读数，在本次飞行的留出工况验算。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

VALUES=26
PRIMARY="用另一组工况的还原误差试选历史退化轨迹"
ALL="固定六工况同时反推寿命"


class InverseLife:
    def fit(self,x,y,owner):
        self.hypotheses=[];self.lives=[];self.units=np.unique(owner)
        median=np.median(x[:,:,:VALUES],axis=0)
        self.scale=np.median(np.abs(x[:,:,:VALUES]-median),axis=0).clip(1e-8)
        for u in self.units:
            rows=np.flatnonzero(owner==u)
            rows=rows[np.argsort(y[rows])]
            if np.any(np.diff(y[rows])!=1):raise ValueError("源轨迹寿命假设不连续")
            # 允许使用已结束的源设备未来读数平滑参照；推理端没有本机历史或寿命。
            smooth=gaussian_filter1d(x[rows,:,:VALUES]/self.scale,2.,axis=0,mode="nearest")
            self.hypotheses.append(smooth);self.lives.append(y[rows].copy())
        return self

    def one(self,x):
        target=x[:,:VALUES]/self.scale
        lists=[[],[],[]]
        for observed,labels in zip(self.hypotheses,self.lives,strict=True):
            difference=observed-target
            # 源档案里的各通道按实测离散程度归一，限制单一站位越界时的影响。
            dist=np.mean(np.minimum(difference**2,16.),axis=2)
            folds=(dist.mean(1),dist[:,::2].mean(1),dist[:,1::2].mean(1))
            for j,curve in enumerate(folds):
                ix=curve.argmin()
                check=folds[2 if j==1 else 1][ix] if j else curve[ix]
                lists[j].append((labels[ix],curve[ix],check))
        options={}
        def estimate(records,held,count):
            table=np.array(records)
            score=table[:,1] if not held else table[:,2]
            use=np.argsort(score)[:count]
            weights=1/(.1+score[use])
            return float(weights@table[use,0]/weights.sum())
        for count in (1,3,7,15):
            options[f"固定全工况取{count}台"]=estimate(lists[0],False,count)
        options[ALL]=options["固定全工况取7台"]
        options["固定两半各自反推"]=np.mean([estimate(v,False,7) for v in lists[1:]])
        options[PRIMARY]=np.mean([estimate(v,True,7) for v in lists[1:]])
        return options

    def predict(self,x,anchor):
        result=[self.one(row) for row in x]
        cols={k:np.array([r[k] for r in result]) for k in result[0]}
        for k,v in list(cols.items()):
            cols["与联合部件各半/"+k]=(anchor+v)/2
        return cols
