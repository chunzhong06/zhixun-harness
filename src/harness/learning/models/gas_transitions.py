"""只以本窗口和过去读数推断气体状态是否已经改变。"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PRIMARY = "依据已回标预测损失选择状态转换或直接预测"
TRANS = "递推出现与消失"
BRANCH = "上窗状态分支"


class EdgeEncoder:
    def __init__(self,source):
        self.scale = StandardScaler().fit(source)
        self.past = []

    def encode(self,group):
        z = self.scale.transform(group)
        hist = np.vstack([*self.past,z]) if self.past else z
        left = len(hist)-len(z)
        first = z[:min(3,len(z))].mean(0)
        late = z[-min(3,len(z)):].mean(0)
        prior = hist[max(0,left-3):left].mean(0) if left else first
        slow = hist[max(0,left-15):left].mean(0) if left else first
        v = np.r_[z.mean(0),late,first-prior,late-first,late-slow]
        self.past = [hist[-18:]]
        return v


class TransitionModels:
    def fit(self,x,y):
        self.scale = StandardScaler().fit(x)
        self.src = self.scale.transform(x)
        self.labels = y
        self.site = []
        self.ref = {}
        self.ref_score = np.zeros(5)
        self.n = 0.
        self.state = .5
        self.last_direct = .5
        self.retrain()
        return self

    @staticmethod
    def classifier(x,y,w):
        return LogisticRegression(C=.1,max_iter=350,random_state=17).fit(x,y,sample_weight=w)

    def retrain(self):
        count = len(self.site)
        if count:
            ix = np.array(self.site)
            x = np.vstack((self.src,np.stack([self.ref[i][0] for i in ix])))
            y = np.r_[self.labels,[self.ref[i][1] for i in ix]].astype(int)
            w = np.r_[np.full(len(self.src),min(count,500)/len(self.src)),np.ones(count)]
        else:
            x,y,w = self.src,self.labels,np.ones(len(self.labels))
        self.direct = self.classifier(x,y,w)
        prev = list(self.labels[:-1]); dest = list(self.labels[1:])
        xx = [self.src[1:]];site_prev=[];site_next=[];site_x=[]
        for i in self.site:
            if i-1 in self.ref:
                site_prev.append(self.ref[i-1][1]);site_next.append(self.ref[i][1]);site_x.append(self.ref[i][0])
        ns=len(prev)
        if site_x:
            xx.append(np.stack(site_x));prev.extend(site_prev);dest.extend(site_next)
            w=np.r_[np.full(ns,min(len(site_x),400)/ns),np.ones(len(site_x))]
        else:w=np.ones(ns)
        prev,dest,xx = np.array(prev),np.array(dest),np.vstack(xx)
        self.branches = [self.classifier(xx[prev==v],dest[prev==v],w[prev==v]) for v in (0,1)]

    def predict(self,x,known_probability):
        z = self.scale.transform(x)
        direct = self.direct.predict_proba(z)[:,1]
        a,b = [m.predict_proba(z)[:,1] for m in self.branches]
        path,soft = [],[]
        for p,start,stay in zip(direct,a,b,strict=True):
            self.state = (1-self.state)*start+self.state*stay
            path.append(self.state)
            soft.append((1-self.last_direct)*start+self.last_direct*stay)
            self.last_direct=p
        bank = np.column_stack((known_probability,direct,path,soft,(known_probability+path)/2))
        cost = (self.ref_score+8*.2)/(self.n+8)
        weights=np.exp(-20*(cost-cost.min()));weights/=weights.sum()
        names=("原差分","直接预测当前状态",TRANS,BRANCH,"递推与原差分各半")
        result=dict(zip(names,bank.T,strict=True))
        result[PRIMARY]=bank@weights
        return result,bank,z

    def update(self,indices,x,y,bank):
        for i,row,label in zip(indices,x,y,strict=True):
            if i in self.ref:raise ValueError("标签重复进入状态转移训练")
            self.site.append(i);self.ref[i]=(row,int(label))
        loss=((bank>=.5)!=y[:,None]).astype(float)+.15*(np.clip(bank,0,1)-y[:,None])**2
        self.ref_score=.9*self.ref_score+loss.sum(0);self.n=.9*self.n+len(y)
        self.retrain()
