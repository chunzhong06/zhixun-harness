"""由标气成分预测传感器响应，再逆推未知混合物中的目标浓度。"""
import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import StandardScaler


NAMES = ("只拟合目标气体响应","从源配方分离各成分响应","在目标标气重估各成分响应","合用源和目标标气估计响应")
PRIMARY = "用装机留组实测选择是否分解混合响应"


class CompositionModel:
    def fit(self,sx,sy,tx,ty,target_index,all_components,source_mass):
        self.target_index = target_index if all_components else 0
        use = list(range(sy.shape[1])) if all_components else [target_index]
        self.chem = StandardScaler().fit(sy[:,use])
        self.signal = StandardScaler().fit(tx)
        zb = self.signal.transform(tx)
        za = (sx - sx.mean(0))/self.signal.scale_
        ca,cb = self.chem.transform(sy[:,use]), self.chem.transform(ty[:,use])
        self.offset = cb.mean(0)
        ca,cb = ca-ca.mean(0), cb-self.offset
        # 以配方为单位在两个器件分别消除常数偏移；在相同单位下估计各成分响应。
        lhs = source_mass*(ca.T@ca)/len(ca) + (1-source_mass)*(cb.T@cb)/len(cb) + .05*np.eye(len(use))
        rhs = source_mass*(ca.T@za)/len(ca) + (1-source_mass)*(cb.T@zb)/len(cb)
        response = solve(lhs,rhs,assume_a="pos")
        # 标气响应解释不了的共同波动作为观测噪声；用低秩求解避免构造1152方阵。
        error = np.vstack(((za-ca@response)*np.sqrt(source_mass/len(ca)),
                           (zb-cb@response)*np.sqrt((1-source_mass)/len(cb))))
        sigma = max(.03,float(np.mean(np.sum(error**2,axis=0))))
        precision = (response - (response@error.T)@solve(error@error.T+sigma*np.eye(len(error)),error,assume_a="pos"))/sigma
        self.inverse = solve(precision@response.T+np.eye(len(use)),precision,assume_a="pos")
        return self

    def predict(self,x):
        concentrations = self.signal.transform(x)@self.inverse.T + self.offset
        return self.chem.inverse_transform(concentrations)[:,self.target_index]


def fit_tools(sx,sy,tx,ty,idx):
    return [CompositionModel().fit(sx,sy,tx,ty,idx,all_channels,mass) for all_channels,mass in
            ((False,.5),(True,1.),(True,0.),(True,.5))]


class CompositionTrial:
    def fit(self,held,y,cut):
        y = np.asarray(y)
        thresholds = ((held>cut)!=(y[:,None]>cut)).astype(float)
        loss = thresholds + .25*np.minimum(((held-y[:,None])/max(y.std(),1.))**2,4)
        self.means = loss.mean(0)
        self.lr = self.means.argmin()
        self.held = held
        self.loss = loss
        self.scale = max(y.std(),1.)
        return self

    def predict(self,bank):
        diff = (bank[:,None,:]-self.held[None])/self.scale
        similar = np.exp(-np.mean(diff*diff,axis=2))
        risks = (similar@self.loss+4*self.means)/(similar.sum(1)[:,None]+4)
        w = np.exp(-10*(risks-risks.min(1)[:,None]))
        chosen = (w*bank).sum(1)/w.sum(1)
        return {PRIMARY:chosen,"按整组装机平均误差选分解法":bank[:,self.lr]}
