"""从已经结束的窗口对建立提前量读出；当前预测不访问后续波形。"""
import numpy as np
from scipy.linalg import solve

HORIZONS=(0,1,2,3)
PRIMARY="按已回标实测时差选择提前量"


class DelayedTeacher:
    def fit(self,x,delayed,y):
        self.mean=x.mean(0);self.sd=np.maximum(x.std(0),.02)
        design=self.design(x)
        self.grams=[];self.rhs=[];self.online=[];self.outcomes=[]
        self.count=[]
        for lag in HORIZONS:
            # 源端只用已经完整采集的序列，尾部尚无后续读数的配对不参加训练。
            n=len(y)-lag
            z=design[:n]
            target=.5*y[:n]+.5*delayed[lag:]
            self.grams.append(z.T@z/n)
            self.rhs.append(z.T@target/n)
            self.online.append(np.zeros_like(self.grams[-1]))
            self.outcomes.append(np.zeros_like(self.rhs[-1]))
            self.count.append(0)
        self.rows=[];self.issued=[];self.answers={};self.seen=[set() for _ in HORIZONS]
        self.errors=np.full(4,.2)
        return self

    @classmethod
    def from_source(cls,edge,diff,reservoir,y):
        from harness.learning.models.dynamic_gas_readout import StreamRegression
        held=np.empty(len(y))
        blocks=np.arange(len(y))//600
        for k in np.unique(blocks):
            test=blocks==k
            predictions=[StreamRegression().fit(v[~test],y[~test]).outputs(v[test])[:,0] for v in (diff,reservoir)]
            held[test]=np.mean([np.clip(p,0,1) for p in predictions],axis=0)
        return cls().fit(np.column_stack((edge,diff)),held,y)

    def design(self,x):
        return np.column_stack((np.ones(len(x)),(x-self.mean)/self.sd))

    def predict(self,x,anchor):
        z=self.design(x)
        values=[]
        for i in HORIZONS:
            n=max(self.count[i],1)
            g=.2*self.grams[i]+self.online[i]/n
            b=.2*self.rhs[i]+self.outcomes[i]/n
            beta=solve(g+.003*np.eye(g.shape[1]),b,assume_a="pos")
            values.append(z@beta)
        bank=np.column_stack(values)
        w=np.exp(-30*(self.errors-self.errors.min()))
        estimate=bank@(w/w.sum())
        out={f"固定提前{i}窗":p for i,p in enumerate(values)}
        out[PRIMARY]=(estimate+anchor)/2
        out["固定四提前量平均"]=bank.mean(1)
        for i in HORIZONS:out[f"原动态与提前{i}窗各半"]=(bank[:,i]+anchor)/2
        self.rows.extend(z)
        self.issued.extend(anchor)
        return out

    def update(self,indices,y):
        # 已经提交的后续预测只能用于下一批训练，不改写任何先前预测。
        self.answers.update(zip(indices,y,strict=True))
        for h in HORIZONS:
            ready=[k for k in self.answers if k+h<len(self.issued) and k not in self.seen[h]]
            if not ready:continue
            z=np.stack([self.rows[k] for k in ready])
            later=np.clip([self.issued[k+h] for k in ready],0,1)
            labels=np.array([self.answers[k] for k in ready],dtype=float)
            target=(labels+later)/2
            self.online[h]+=z.T@z;self.outcomes[h]+=z.T@target
            self.count[h]+=len(ready);self.seen[h].update(ready)
            err=np.mean(((later>=.5)!=labels)+.2*(later-labels)**2)
            self.errors[h]=.8*self.errors[h]+.2*err
