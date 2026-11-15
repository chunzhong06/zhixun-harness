"""SGP40 单秒特征校准、论文两层迁移，以及只从已知标气学习的装机发布策略。"""
import numpy as np
from scipy.linalg import solve
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

NAMES = ("源端论文特征回归","论文两层标气续训","新芯片单独标定","标气校正源端偏差","目标单秒特征岭回归","论文与标气校正各半")
PRIMARY = "用当前各版预测及装机误差决定组合"


def feature_cycles(raw):
    """原始四路周期转换成论文每秒的均值和最小二乘斜率。"""
    v = np.asarray(raw,dtype=float)
    if v.ndim!=3 or v.shape[1:]!=(4,1440) or not np.isfinite(v).all():
        raise ValueError("需要四路完整温度周期")
    sec = v.reshape(-1,4,144,10)
    time = np.arange(10)/10 - .45
    return np.stack((sec.mean(-1),sec @ time/(time@time)),axis=-1).reshape(len(v),1152)


class GasBase:
    def fit(self,x,y):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.mean = y.mean()
        self.sd = max(y.std(),1.)
        norm = (y-self.mean)/self.sd
        # 按岭回归系数逐批剔除弱特征，在本地CPU预算内保留64项后做PLS。
        selected = np.arange(z.shape[1])
        while len(selected)>64:
            local = z[:,selected]
            weight = local.T @ solve(local@local.T + 3*np.eye(len(x)), norm, assume_a="pos")
            selected = selected[np.argsort(-abs(weight),kind="stable")[:max(64,len(selected)//2)]]
        self.selected = selected
        model = PLSRegression(n_components=min(8,len(x)-1,len(selected)),scale=False).fit(z[:,selected],norm)
        self.beta = model.coef_.ravel()
        self.bias = float(model.intercept_.ravel()[0] - self.beta @ z[:,selected].mean(0))
        return self

    def predict(self,x):
        z = self.scaler.transform(x)
        return (z[:,self.selected]@self.beta+self.bias)*self.sd+self.mean

    def transfer(self,x,y):
        """冻结标准化；由真实源模型的选列矩阵和输出系数出发，联合更新两个线性层。"""
        z = self.scaler.transform(x)
        yy = (y-self.mean)/self.sd
        w = np.eye(z.shape[1])[:,self.selected].copy()
        beta,bias = self.beta.copy(),self.bias
        # 每10轮衰减，必要时回退步长以避免小样本上的不稳定；使用本地固定迭代数。
        for epoch in range(60):
            hidden = z@w
            residual = hidden@beta+bias-yy
            e = residual/len(yy)
            dw = np.outer(z.T@e,beta)
            db = hidden.T@e
            dc = e.sum()
            step = .1 * .9**(epoch//10)
            loss = residual@residual
            for _ in range(25):
                new_w = w-step*dw
                new_b = beta-step*db
                new_c = bias-step*dc
                err = z@(new_w@new_b)+new_c-yy
                if err@err <= loss:
                    w,beta,bias = new_w,new_b,new_c
                    break
                step /= 2
        return w@beta,bias


class YearTools:
    def fit(self,base,x,y):
        self.base = base
        self.transfer_w,self.transfer_b = base.transfer(x,y)
        self.target = GasBase().fit(x,y)
        self.scale = StandardScaler().fit(x)
        z = self.scale.transform(x)
        self.local = Ridge(alpha=10).fit(z,y)
        self.correction = Ridge(alpha=10).fit(z,y-base.predict(x))
        return self

    def outputs(self,x):
        original = self.base.predict(x)
        network = (self.base.scaler.transform(x)@self.transfer_w+self.transfer_b)*self.base.sd+self.base.mean
        z = self.scale.transform(x)
        corrected = original+self.correction.predict(z)
        return np.column_stack((original,network,self.target.predict(x),corrected,self.local.predict(z),(network+corrected)/2))


class YearInstallation:
    def fit(self,history,y,cut):
        self.history = history
        self.cut = cut
        self.sd = max(y.std(),1.)
        self.avg_loss = ((history-y[:,None])/self.sd)**2
        # 类别和连续误差都在留出整种标气后计算；相同的装机结果也用于固定全局对照。
        self.cost = ((history>cut)!=(y[:,None]>cut)).astype(float) + .2*np.minimum(self.avg_loss,4)
        self.global_cost = self.cost.mean(0)
        return self

    def predict(self,values):
        # 浓度不同并且各模型分歧大的测量，允许以相似装机试验的结果改变权重。
        diff = (values[:,None,:]-self.history[None,:,:])/self.sd
        distance = np.mean(diff**2,axis=2)
        like = np.exp(-distance/.6)
        local = (like@self.cost + 5*self.global_cost)/(like.sum(1)[:,None]+5)
        weights = np.exp(-8*(local-local.min(1)[:,None]))
        global_weights = np.exp(-8*(self.global_cost-self.global_cost.min()))
        p = dict(zip(NAMES,values.T,strict=True))
        p[PRIMARY] = (weights*values).sum(1)/weights.sum(1)
        p["统一装机试验配比"] = values@(global_weights/global_weights.sum())
        p["统一装机选一版"] = values[:,self.global_cost.argmin()]
        p["全部工具固定平均"] = values.mean(1)
        return p
