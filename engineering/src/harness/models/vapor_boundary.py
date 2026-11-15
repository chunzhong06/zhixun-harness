"""按烟羽阈值附近的真实测量学习读出，限制强峰值对低浓度决策的影响。"""
import numpy as np
from scipy.linalg import solve
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

VIEWS=("逆响应","论文储备池","跨通道曲线")
TARGETS=("电压","压缩电压","边界加权电压","越界概率")
PRIMARY="由装机试验和现场回标合用边界模型"


def measured_views(v,idx):
    instant=v[3][idx]
    # 显式均使用当前或已过去采样。两个分量分别表示响应和恢复时的不同斜率。
    d=v[4][idx].reshape(len(idx),3,5)
    s=d[:,:,1]
    inverse=np.column_stack((v[4][idx],np.maximum(s,0),np.minimum(s,0)))
    base=np.column_stack((instant,np.tanh(instant),s))
    cross=(base[:,:,None]*base[:,None,:])[:,np.triu_indices(base.shape[1])[0],np.triu_indices(base.shape[1])[1]]
    return (inverse,v[5][idx],np.column_stack((inverse,np.arcsinh(instant),np.clip(cross,-8,8))))


class LevelFit:
    def fit(self,x,y):
        self.scale=StandardScaler().fit(x)
        self.source_n=len(y)
        a=self.phi(x)
        self.sg,self.sb=self.stats(a,y)
        self.g=[np.zeros_like(v) for v in self.sg];self.b=[np.zeros_like(v) for v in self.sb]
        self.count=0.
        return self

    def phi(self,x):
        return np.column_stack((np.ones(len(x)),self.scale.transform(x)))

    @staticmethod
    def stats(a,y):
        weights=.15+np.exp(-abs(y-1)/.55)
        gram=a.T@a
        return [gram,gram,a.T@(weights[:,None]*a),gram], [a.T@y,a.T@np.arcsinh(y),a.T@(weights*y),a.T@(y>=1)]

    def predict(self,x):
        a=self.phi(x)
        result=[]
        for j in range(4):
            # sg,sb均保留源端有效窗口的总量，现场以归一化的最近回标合并。
            g=self.sg[j]/self.source_n
            b=self.sb[j]/self.source_n
            if self.count:
                g=.15*g+self.g[j]/self.count;b=.15*b+self.b[j]/self.count
            theta=solve(g+.006*np.eye(g.shape[1]),b,assume_a="pos")
            pred=a@theta
            if j==1:pred=np.sinh(np.clip(pred,-4,4))
            if j==3:pred=2*pred
            result.append(pred)
        return np.column_stack(result)

    def update(self,x,y):
        g,b=self.stats(self.phi(x),y)
        self.count=.92*self.count+len(y)
        for j in range(4):
            self.g[j]=.92*self.g[j]+g[j];self.b[j]=.92*self.b[j]+b[j]


class EdgeWeights:
    def fit(self,bank,context,y):
        self.scale=StandardScaler().fit(context)
        self.ref=self.scale.transform(context)
        self.src=self.loss(bank,y)
        self.avg=self.src.mean(0)
        self.context=[];self.hist=[]
        return self

    @staticmethod
    def loss(bank,y):
        return ((bank>=1)!=(y[:,None]>=1)).astype(float)+.30*(expit(4*(bank-1))-(y[:,None]>=1))**2

    def predict(self,bank,context):
        z=self.scale.transform(context)
        def nearest(ref,history,prior,shrink):
            dist=np.mean((z[:,None]-ref[None])**2,axis=2)
            count=min(60,len(ref))
            take=np.argpartition(dist,count-1,axis=1)[:,:count]
            w=np.exp(-np.take_along_axis(dist,take,axis=1))
            return ((history[take]*w[:,:,None]).sum(1)+shrink*prior)/(w.sum(1)[:,None]+shrink)
        expected=nearest(self.ref,self.src,self.avg,5.)
        if self.hist:expected=nearest(np.array(self.context),np.array(self.hist),expected,3.)
        w=np.exp(-25*(expected-expected.min(1)[:,None]))
        return 2*(expit(4*(bank-1))*w).sum(1)/w.sum(1)

    def update(self,bank,c,y):
        self.context.extend(self.scale.transform(c));self.hist.extend(self.loss(bank,y))


def decision_context(q,last):
    med=np.median(q,axis=1)
    return np.column_stack((np.clip(med,-1,4),np.minimum(abs(med-1),2),
        np.log1p(np.std(np.clip(q,-2,8),axis=1)),np.full(len(q),np.clip(last,-1,4)),
        np.arange(1,len(q)+1)/10))
