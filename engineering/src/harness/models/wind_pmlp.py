"""Fiocchi 等 A1 概率网络的三字段适配，源期联合训练后可由本机已返功率微调。"""
from copy import deepcopy
import numpy as np
from scipy.special import expit
from harness.models.wind_source_control import CONTROL, PRIMARY as PRIOR, PAPER as PRIOR_PAPER

PRIMARY = "按本机已返误差与概率网络的置信程度发布"
PAPER = "论文A1全层微调"


class ProbabilityNetwork:
    def __init__(self,inputs=3):
        rng = np.random.default_rng(41)
        sizes = [(inputs,100),(100,80),(80,40),(40,20),(20,1),(40,20),(20,1)]
        self.params = []
        for a,b in sizes:
            w = rng.normal(0,np.sqrt(2/a),(a,b))
            self.params.extend([w,np.zeros(b)])
        self.params[-1][0] = -1.8

    def forward(self,x):
        values = []
        def layer(i,raw,activate):
            h = raw@self.params[2*i]+self.params[2*i+1]
            values.append((raw,h))
            return np.maximum(0,h) if activate else h
        h = x
        for i in range(3):
            h = layer(i,h,True)
        mean = layer(4,layer(3,h,True),False).ravel()
        logit = layer(6,layer(5,h,True),False).ravel()
        sigma = .003+np.logaddexp(0,logit)
        return mean,sigma,logit,values

    def gradients(self,x,y):
        mean,sd,logit,values = self.forward(x)
        residual = (mean-y)/sd
        n = len(x)
        dmean = residual/(n*sd)
        dscale = (1-residual**2)*expit(logit)/(n*sd)
        grads = [None]*len(self.params)
        def reverse(i,delta,activate):
            raw,h = values[i]
            if activate:
                delta = delta*(h>0)
            grads[2*i] = raw.T@delta
            grads[2*i+1] = delta.sum(0)
            return delta@self.params[2*i].T
        shared = reverse(3,reverse(4,dmean[:,None],False),True)+reverse(5,reverse(6,dscale[:,None],False),True)
        for i in (2,1,0):
            shared = reverse(i,shared,True)
        return grads

    def step(self,grads,memory,rate,heads_only=False):
        m,v,number = memory
        number[0] += 1
        scale = max(1.,np.sqrt(sum(np.sum(d*d) for d in grads))/20.)
        for i,d in enumerate(grads):
            if heads_only and i not in (8,9,12,13):
                continue
            d = d/scale
            m[i] = .9*m[i]+.1*d
            v[i] = .999*v[i]+.001*d*d
            self.params[i] -= rate*(m[i]/(1-.9**number[0]))/(np.sqrt(v[i]/(1-.999**number[0]))+1e-8)

    def optimizer(self):
        return ([np.zeros_like(w) for w in self.params],[np.zeros_like(w) for w in self.params],[0])

    def train(self,x,y,t):
        self.center = x.mean(0)
        self.scale = np.maximum(x.std(0),1e-5)
        z = self.standardize(x)
        # 只在源期较晚的时间片选择最佳轮次，然后用所有源样本按选出的轮次重新训练。
        cut = np.quantile(t,.85)
        tr,va = np.where(t<cut)[0],np.where(t>=cut)[0]
        original = deepcopy(self.params)
        best_loss = np.inf
        best_epoch = 1
        for trial,indices in enumerate((tr,np.arange(len(x)))):
            self.params = deepcopy(original)
            opt = self.optimizer()
            rng = np.random.default_rng(41)
            for epoch in range(42 if trial==0 else best_epoch):
                order = rng.permutation(indices)
                for j in range(0,len(order),384):
                    ix = order[j:j+384]
                    self.step(self.gradients(z[ix],y[ix]),opt,.0008 if epoch<20 else .0003)
                if trial==0:
                    mu,sd,_,_ = self.forward(z[va])
                    loss = np.mean(np.log(sd)+.5*((y[va]-mu)/sd)**2)
                    if loss<best_loss:
                        best_loss,best_epoch = float(loss),epoch+1
                    if epoch+1-best_epoch>=7:
                        break
        self.epochs = best_epoch
        return self

    def standardize(self,x):
        return ((x-self.center)/self.scale).clip(-10,10)


class PmlpRelease:
    def __init__(self,previous,network):
        self.previous = previous
        self.network = network

    def predict(self,x):
        return self.previous.predict(x)

    def controller(self):
        return PmlpMaintenance(self.previous.controller(),self.network)


class PmlpMaintenance:
    def __init__(self,previous,source):
        self.old = previous
        self.models = [source,deepcopy(source),deepcopy(source)]
        self.optimizers = [m.optimizer() for m in self.models[1:]]
        self.x = []
        self.y = []
        self.g = np.array([.004]*3)
        self.r = .25*self.g
        self.count = 0

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base)
        out = {n:old[n] for n in (CONTROL,PRIOR,PRIOR_PAPER,"同样工具按全机累计误差分配","三变量提升树源训练")}
        p = out[CONTROL]
        z = self.models[0].standardize(x)
        sd = None
        for model,name in zip(self.models,("论文A1源网络","论文A1只微调输出层",PAPER),strict=True):
            mu,sd,_,_ = model.forward(z)
            out[name] = mu.clip(-.1,1.1)
            for w in (.1,.25,.5,.75):
                out[f"原固定与{name}占{w:g}"] = p+w*(out[name]-p)
        delta = out[PAPER]-p
        zones = np.digitize(sd,[.015,.035])
        out["论文按输出标准差固定合用"] = p+np.choose(zones,[.5,.25,.1])*delta
        out["论文按风况固定合用"] = p+np.choose(np.digitize(x[:,0],[5,10]),[.4,.5,.1])*delta
        out[PRIMARY] = p+(self.r/self.g).clip(0,1)[zones]*delta
        self.pending = z[-1].copy(),p[-1],delta[-1],zones[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,p,d,zone = self.pending
        self.g[zone] = .97*self.g[zone]+d*d+.03*.004
        self.r[zone] = .97*self.r[zone]+d*(y-p)+.03*.001
        self.x.append(z)
        self.y.append(y)
        self.count += 1
        self.x = self.x[-128:]
        self.y = self.y[-128:]
        if self.count%4==0:
            a,b = np.asarray(self.x),np.asarray(self.y)
            for m,opt,heads in zip(self.models[1:],self.optimizers,(True,False),strict=True):
                for _ in range(3):
                    m.step(m.gradients(a,b),opt,.00018 if heads else .00006,heads)
