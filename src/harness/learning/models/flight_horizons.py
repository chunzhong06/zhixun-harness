"""分别预测各寿命门槛能否达到，再用同一航班不同读数的实测分歧合成寿命。"""
import numpy as np
from scipy.special import expit
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


NAMES=("实测通道寿命门槛","部件关系寿命门槛","联合寿命门槛")
DIRECT=("实测通道直接回归","部件关系直接回归","联合直接回归")
GLOBAL="每道门槛采用源端固定配比"
LOCAL="按当前航班门槛分歧选择读法"
PRIMARY="现场门槛预测与原块均值各半"


def views(profile):
    # 同一组源端预置工况的次序固定，取每一工况的中位量和跨工况的离散程度。
    med=profile[:,0,:,:38]
    spread=profile[:,0,:,38:76]
    result=[]
    for ids in (np.arange(14),np.arange(14,38),np.r_[0:14,26:38]):
        val=med[:,:,ids]
        result.append(np.column_stack((val.reshape(len(val),-1),np.median(spread[:,:,ids],axis=1))))
    return result


class HorizonReader:
    def fit(self,x,y,edges):
        self.scale=StandardScaler().fit(x)
        self.kernel=RBFSampler(gamma=.35/x.shape[1],n_components=160,random_state=71)
        self.kernel.fit(self.scale.transform(x))
        design=self.transform(x)
        self.direct=Ridge(alpha=3.).fit(design,y)
        self.classifiers=[]
        for threshold in edges:
            target=y>threshold
            if np.all(target==target[0]):
                self.classifiers.append(float(target[0]))
            else:
                model=LogisticRegression(C=1.,max_iter=400,tol=1e-5)
                self.classifiers.append(model.fit(design,target))
        return self

    def transform(self,x):
        z=np.clip(self.scale.transform(x),-6,6)
        return np.column_stack((z/np.sqrt(z.shape[1]),self.kernel.transform(z)*4))

    def predict(self,x):
        design=self.transform(x)
        bank=[np.full(len(x),m) if isinstance(m,float) else expit(design@m.coef_[0]+m.intercept_[0])
              for m in self.classifiers]
        return np.column_stack(bank),np.maximum(0.,self.direct.predict(design))


class HorizonLife:
    @staticmethod
    def train(x,y,edges):
        return [HorizonReader().fit(view,y,edges) for view in x]

    @staticmethod
    def read(models,x):
        p,d=zip(*(m.predict(view) for m,view in zip(models,x,strict=True)),strict=True)
        return np.stack(p,axis=2),np.column_stack(d)

    @staticmethod
    def context(p):
        mid=p.mean(2)
        # 以结果揭晓前可见的概率水平和通道分歧分组，源端与本次使用同一公式。
        return np.digitize(mid,[.15,.4,.6,.85])*3+np.digitize(np.ptp(p,axis=2),[.08,.25])

    def fit(self,profile,y,groups):
        x=views(profile)
        # 五趟宽度的区间取中点，概率曲线面积近似剩余飞行数的期望。
        self.edges=np.arange(2.5,np.ceil(y.max()/5)*5,5)
        held=np.empty((len(y),len(self.edges),len(NAMES)))
        for group in np.array_split(np.unique(groups),5):
            mask=np.isin(groups,group)
            models=self.train([a[~mask] for a in x],y[~mask],self.edges)
            held[mask],_=self.read(models,[a[mask] for a in x])
        err=(held-(y[:,None,None]>self.edges[None,:,None]))**2
        # 每台源机总权相同，长轨迹不因包含更多容易的窗口而支配取舍。
        counts=dict(zip(*np.unique(groups,return_counts=True),strict=True))
        sw=np.array([1/counts[g] for g in groups]);sw*=len(y)/sw.sum()
        self.global_cost=np.average(err,axis=0,weights=sw)
        ctx=self.context(held)
        self.local_cost=np.zeros((len(self.edges),15,len(NAMES)))
        for j in range(len(self.edges)):
            for k in range(15):
                weights=sw*(ctx[:,j]==k)
                self.local_cost[j,k]=(weights@err[:,j]+40*self.global_cost[j])/(weights.sum()+40)
        self.models=self.train(x,y,self.edges)
        return self

    @staticmethod
    def mix(bank,cost):
        weights=1/(.015+cost)**2
        return (bank*weights).sum(2)/weights.sum(-1)

    def predict(self,profile):
        bank,direct=self.read(self.models,views(profile))
        global_prob=self.mix(bank,self.global_cost)
        local=self.local_cost[np.arange(len(self.edges))[None,:],self.context(bank)]
        local_prob=self.mix(bank,local)
        out={**dict(zip(NAMES,(5*bank.sum(1)).T,strict=True)),**dict(zip(DIRECT,direct.T,strict=True)),
             GLOBAL:5*global_prob.sum(1),LOCAL:5*local_prob.sum(1)}
        return out
