"""用已返功率在时间、风速、转速与温度上拟合局部误差，并前向检查相近参考能否迁移。"""
import numpy as np
from scipy.linalg import cho_solve
from harness.models.wind_limit import PRIMARY as PREVIOUS, PAPER as PREVIOUS_PAPER

ANCHOR = "原固定加上限校正0.98占0.25"
PRIMARY = "依过去参考的前向验证更新本机误差曲面"
PAPER = "论文上限读法加时空校正"
SPECS = (("同风同转长期", [2.,1.,10.], 24*90, .45),
         ("同风同转短期", [2.,1.,8.], 48., .55),
         ("同温同风长期", [1.5,3.,2.5], 24*60, .65),
         ("同温同风短期", [3.,4.,4.], 18., .85),
         ("最近天气", [8.,8.,8.], 9., .70),
         ("跨风段转速偏差", [4.,.35,6.], 24*45, .80))


def kernel(x,old,t,stamps,width,hours):
    difference = (x[:,None,:]-old[None,:,:])/width
    dist = np.sum(difference**2,axis=2)
    time = np.abs(t[:,None]-stamps[None,:])/3600
    return np.exp(-.5*dist-time/hours)


class ReferenceRelease:
    def __init__(self,base):
        self.base = base

    def predict(self,x):
        return self.base.predict(x)

    def controller(self):
        return ReferenceMaintenance(self.base.controller())


class ReferenceMaintenance:
    def __init__(self,previous):
        self.old = previous
        self.x = np.empty((0,3))
        self.t = np.empty(0)
        self.errors = np.empty(0)
        self.coefficients = []
        self.risks = np.full((4,len(SPECS)+1),.00035)
        self.last = None

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base)
        keep = (PREVIOUS,PREVIOUS_PAPER,ANCHOR,"同样工具按全机累计误差分配","三变量提升树源训练")
        out = {n:old[n] for n in keep}
        p = out[ANCHOR]
        bank = [p]
        for i,(name,width,hours,noise) in enumerate(SPECS):
            correction = np.zeros(len(t))
            if len(self.t):
                cross = kernel(x,self.x,t,self.t,width,hours)
                correction = cross@self.coefficients[i]
            amended = p+correction.clip(-.12,.12)
            out[name+"误差校正"] = amended
            for amplitude in (.25,.5,.75):
                out[f"原固定加{name}误差占{amplitude:g}"] = p+amplitude*(amended-p)
            bank.append(amended)
        bank = np.column_stack(bank)
        zone = np.digitize(x[:,0],[4.5,8.,11.])
        # 每份证据先在上一次没有见过本条参考时发放，再把其损失计入下一次选择。
        global_risk = self.risks.mean(0)
        weight = 1/(.00005+.5*global_risk+.5*self.risks[zone])**3
        out[PRIMARY] = np.sum(weight*bank,axis=1)/weight.sum(1)
        out["固定多种相近参考等权"] = bank.mean(1)
        out["固定半份参考校正等权"] = (p+bank[:,1:].mean(1))/2
        out[PAPER] = old[PREVIOUS_PAPER]+.5*(out["同风同转短期误差校正"]-p)
        self.last = t[-1],x[-1].copy(),p[-1],bank[-1],zone[-1]
        return out

    def update(self,y):
        self.old.update(y)
        t,x,p,bank,z = self.last
        self.risks[z] = .92*self.risks[z]+.08*np.minimum((y-bank)**2,.04)
        self.t = np.r_[self.t,t][-96:]
        self.x = np.vstack((self.x,x))[-96:]
        self.errors = np.r_[self.errors,np.clip(y-p,-.15,.15)][-96:]
        self.coefficients = []
        for name,width,hours,noise in SPECS:
            k = kernel(self.x,self.x,self.t,self.t,width,hours)
            k.flat[::len(k)+1] += noise
            L = np.linalg.cholesky(k)
            self.coefficients.append(cho_solve((L,True),self.errors,check_finite=False))
