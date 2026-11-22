"""移动探头的相对气体响应。参考仪只用于装机、已发反馈和接收后的评价。"""
from collections import deque

import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import StandardScaler

SENSORS=("MiCS6814_NH3","MiCS6814_NO2","MiCS5524")
READERS=(*SENSORS,"三通道即时","三通道逆响应","论文储备池","逆响应与论文合训")
PRIMARY="按通道先后变化与已返误差选择当下读法"
POLICIES=("源端保持","累计装机","近期装机")


def voltage_ratio(v):
    if not np.isfinite(v).all() or np.any((v<=0)|(v>=5)):
        raise ValueError("原始慢传感器电压不在电桥范围内")
    # 对数电阻里的固定负载常量在减去启动基线后相消，5524的电桥接法相反。
    return np.log(v/(5-v))*np.array([1,1,-1])


class ProbeEncoder:
    def __init__(self,base,scale):
        self.base=base
        self.scale=scale
        self.past=deque(maxlen=17)
        rng=np.random.default_rng(73)
        w=rng.normal(size=(36,36))*(rng.random((36,36))<.20)
        self.w=.9*w/max(abs(np.linalg.eigvals(w)))
        self.win=rng.normal(0,.24,size=(6,36))
        self.h=np.zeros((2,36))
        self.bias=rng.uniform(-.15,.15,size=36)

    def next(self,volts):
        samples=(voltage_ratio(volts)-self.base)/self.scale
        now=samples.mean(0)
        if not self.past:self.past.append(now)
        prev=np.stack([self.past[max(0,len(self.past)-i)] for i in (1,4,16)])
        delta=now-prev
        recent=samples[-1]-samples[0]
        channel=np.column_stack((now,delta.T,recent))
        self.past.append(now)
        h0=np.tanh(np.r_[now,delta[0]]@self.win+self.h@self.w+self.bias)
        self.h+=np.array([.65,.13])[:,None]*(h0-self.h)
        pooled=channel.ravel()
        reservoir=np.r_[now,self.h.ravel()]
        # 仅以可观测的当前变化描述哪一路先变，固定方也可使用这一信息。
        quick=np.abs(delta[0])+.5*np.abs(recent)
        context=np.r_[np.log1p(quick),quick/(.02+quick.sum()),np.log1p(abs(now))]
        return [*channel,now,pooled,reservoir,np.r_[pooled,self.h.ravel()]],context


class OnlineVoltage:
    def fit(self,x,y):
        self.scale=StandardScaler().fit(x)
        a=self.design(x)
        self.src_gram=a.T@a/len(y)
        self.src_y=a.T@y/len(y)
        self.clear()
        return self

    def design(self,x):
        return np.column_stack((np.ones(len(x)),self.scale.transform(x)))

    def clear(self):
        self.n=np.zeros(2)
        self.gram=np.zeros((2,*self.src_gram.shape))
        self.rhs=np.zeros((2,len(self.src_y)))

    def predict(self,x):
        a=self.design(x)
        ridge=.01*np.eye(a.shape[1])
        source=solve(self.src_gram+ridge,self.src_y,assume_a="pos")
        cols=[a@source]
        for i in range(2):
            mass=max(1.,self.n[i])
            coeff=solve(.18*self.src_gram+self.gram[i]/mass+ridge,.18*self.src_y+self.rhs[i]/mass,assume_a="pos")
            cols.append(a@coeff if self.n[i] else cols[0])
        return np.column_stack(cols)

    def update(self,x,y):
        a=self.design(x)
        for i,decay in enumerate((1.,.90)):
            self.gram[i]=decay*self.gram[i]+a.T@a
            self.rhs[i]=decay*self.rhs[i]+a.T@y
            self.n[i]=decay*self.n[i]+len(y)


class ProbePolicy:
    def fit(self,context,bank,y):
        self.scale=StandardScaler().fit(context)
        self.install=self.scale.transform(context)
        self.site=[]
        self.site_loss=[]
        self.loss=self.losses(bank,y)
        self.base=self.loss.mean(0)
        return self

    @staticmethod
    def losses(bank,y):
        # 浓度单位是以阈值为1的相对参考电压，0到1分类损失与连续偏差分别记录。
        return ((bank>=1)!=(y[:,None]>=1)).astype(float)+.12*np.minimum((bank-y[:,None])**2,4)

    def predict(self,ctx,bank):
        query=self.scale.transform(ctx)
        def local(query,z,loss,strength):
            dist=np.mean((query[:,None]-z[None])**2,axis=2)
            count=min(len(z),80)
            ix=np.argpartition(dist,count-1,axis=1)[:,:count]
            near=np.exp(-np.take_along_axis(dist,ix,axis=1))
            return ((near[:,:,None]*loss[ix]).sum(1)+strength*self.base)/(near.sum(1)[:,None]+strength)
        cost=local(query,self.install,self.loss,8)
        if self.site:
            measured=local(query,np.array(self.site),np.array(self.site_loss),3)
            cost=(cost+2*measured)/3
        weights=np.exp(-12*(cost-cost.min(1)[:,None]))
        return (bank*weights).sum(1)/weights.sum(1)

    def update(self,ctx,bank,y):
        self.site.extend(self.scale.transform(ctx))
        self.site_loss.extend(self.losses(bank,y))
