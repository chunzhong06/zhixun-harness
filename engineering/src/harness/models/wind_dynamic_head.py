"""用源期连续片段训练概率网络，新机的合法功率更新同一个均值读出层。"""
from copy import deepcopy
import numpy as np
from harness.models.wind_history import WindHistory,HISTORY_EFFECTS,CONTROL, PRIMARY as HISTORY_SITE
from harness.models.wind_pmlp import ProbabilityNetwork
from harness.models.wind_transition import ANCHOR,PRIMARY as PREVIOUS
from harness.models.wind_history import PAPER as OLD_PAPER

PRIMARY = "根据近期试验选择连续波形网络的微调层级"
PAPER = "论文A1连续风况全层微调"
DECAYS = (.92,.985,1.)


def history_design(x,h):
    v,r,temp = x.T
    return np.column_stack((x,h,r**2/200,v**3/1000,r/np.maximum(v,2)))


def train_network(rows,fields):
    inputs,targets,clocks = [],[],[]
    for _,frame in rows.groupby("turbine_id",sort=True):
        frame = frame.sort_values("clock")
        x,t = frame[fields].to_numpy(),frame.clock.to_numpy()
        inputs.append(history_design(x,WindHistory().transform(t,x)))
        targets.append(frame.power_pu.to_numpy())
        clocks.append(t)
    features = np.vstack(inputs)
    return ProbabilityNetwork(features.shape[1]).train(features,np.concatenate(targets),np.concatenate(clocks))


class DynamicHeadRelease:
    def __init__(self,old,network):
        self.old = old
        self.network = network

    def predict(self,x):
        return self.old.predict(x)

    def controller(self):
        return DynamicHeadMaintenance(self.old.controller(),self.network)


class DynamicHeadMaintenance:
    def __init__(self,previous,source):
        self.old = previous
        self.source = source
        self.tuned = deepcopy(source)
        self.optimizer = self.tuned.optimizer()
        self.history = WindHistory()
        self.gram = np.zeros((len(DECAYS),21,21))
        self.rhs = np.zeros((len(DECAYS),21))
        self.count = 0
        self.examples = []
        self.risk = np.full((3,3),.00035)

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base)
        keep = (ANCHOR,PREVIOUS,HISTORY_SITE,OLD_PAPER,CONTROL,*HISTORY_EFFECTS,"同样工具按全机累计误差分配","三变量提升树源训练")
        out = {n:old[n] for n in keep}
        p = out[ANCHOR]
        z = self.source.standardize(history_design(x,self.history.transform(t,x)))
        mu,sd,_,layers = self.source.forward(z)
        updated = self.tuned.forward(z)[0].clip(-.1,1.1)
        # 均值输出层的输入是最后20个激活量。直接求有源端先验的最小二乘微调，保留相同训练权利。
        latent = np.column_stack((np.ones(len(t)),layers[4][0]))
        features = latent/np.maximum(1.,np.linalg.norm(latent,axis=1,keepdims=True))
        coef = np.linalg.solve(self.gram+.8*np.eye(21),self.rhs[:,:,None])[:,:,0]
        linear = (mu[:,None]+features@coef.T).clip(-.1,1.1)
        options = {"论文A1连续风况冻结":mu.clip(-.1,1.1),PAPER:updated}
        options.update({f"连续风况源先验读出{d:g}":linear[:,i] for i,d in enumerate(DECAYS)})
        for n,values in options.items():
            out[n] = values
            for w in (.1,.25,.5,.75):
                out[f"原固定与{n}占{w:g}"] = p+w*(values-p)
        bank = np.column_stack((p,updated,linear[:,1]))
        zone = np.digitize(x[:,0],[5,10])
        weight = 1/(.00003+self.risk[zone])**3
        adaptive = np.sum(weight*bank,axis=1)/weight.sum(1)
        out[PRIMARY] = (p+adaptive)/2
        out["冻结连续微调初始各半"] = (p+bank.mean(1))/2
        out["论文连续风况按风段固定合用"] = p+np.choose(zone,[.5,.35,.1])*(updated-p)
        out["论文连续风况按标准差固定合用"] = p+np.choose(np.digitize(sd,[.015,.035]),[.5,.25,.1])*(updated-p)
        self.pending = z[-1].copy(),features[-1],mu[-1],zone[-1],bank[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,f,mu,k,bank = self.pending
        d = np.array(DECAYS)
        self.gram = d[:,None,None]*self.gram+np.outer(f,f)
        self.rhs = d[:,None]*self.rhs+f*(y-mu)
        self.risk[k] = .94*self.risk[k]+.06*np.minimum((y-bank)**2,.04)
        self.examples.append((z,y))
        self.examples = self.examples[-128:]
        self.count += 1
        if self.count%4==0:
            xx,yy = zip(*self.examples,strict=True)
            for _ in range(3):
                self.tuned.step(self.tuned.gradients(np.array(xx),np.array(yy)),self.optimizer,.00006)
