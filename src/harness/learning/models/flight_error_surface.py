"""从已结束设备的实际寿命误差学习修正。在每趟服务航班附近重新拟合斜率。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


NAMES = ("已结束轨迹的平均误差", "近六条轨迹的平均误差", "近六条轨迹的局部误差斜率", "近十二条轨迹的局部误差斜率")
PRIMARY = "依相似轨迹留出实试决定采用哪种误差修正"


def evidence(profiles, anchor, paper):
    # 同一航班六个固定工况下的传感器与部件关系，均只来自当前可测输入。
    middle = profiles[:,0,:,:38]
    return np.column_stack((middle.mean(1),middle.std(1),anchor,paper-anchor))


class ErrorSurface:
    def fit(self, x, residual, groups):
        self.scale = StandardScaler().fit(x)
        z = self.scale.transform(x)
        self.axes = PCA(n_components=min(10,len(x),x.shape[1]),whiten=True,svd_solver="full").fit(z)
        self.z = self.axes.transform(z)
        self.residual = residual
        self.units = [np.flatnonzero(groups == u) for u in np.unique(groups)]
        self.bias = np.mean([residual[ix].mean() for ix in self.units])
        return self

    def project(self,x):
        return self.axes.transform(self.scale.transform(x))

    def predict(self,x):
        z = self.project(x)
        d = cdist(z,self.z,"sqeuclidean")
        # 每条退化轨迹最多用三个最相近的航班，不让最长的设备获得最多票。
        nearest = np.stack([ix[np.argsort(d[:,ix],axis=1)[:,:3]] for ix in self.units],axis=1)
        by_unit = np.take_along_axis(d[:,None,:],nearest,axis=2).mean(2)
        ordered = np.argsort(by_unit,axis=1)
        out = np.full((len(x),len(NAMES)),self.bias)
        for i in range(len(x)):
            for width, col in ((6,2),(12,3)):
                ix = nearest[i,ordered[i,:width]].ravel()
                weights = np.exp(-d[i,ix]/max(np.median(d[i,ix]),.01))
                weights /= weights.sum()
                dx = self.z[ix]-z[i]
                target = self.residual[ix]
                center = np.average(dx,axis=0,weights=weights)
                mean = weights @ target
                centered = dx-center
                lhs = centered.T @ (weights[:,None]*centered) + .3*np.eye(dx.shape[1])
                slope = np.linalg.solve(lhs,centered.T @ (weights*(target-mean)))
                out[i,col] = mean - center @ slope
                if width == 6:
                    out[i,1] = mean
        return out


class FlightCorrection:
    def fit(self,x,y,anchor,groups):
        err = y-anchor
        self.full = ErrorSurface().fit(x,err,groups)
        # 安装端留掉整条历史轨迹，比较维持现状或采纳固定幅度的修正。
        withheld = np.zeros((len(y),len(NAMES)))
        for test_groups in np.array_split(np.unique(groups),min(5,len(np.unique(groups)))):
            test = np.isin(groups,test_groups)
            fold = ErrorSurface().fit(x[~test],err[~test],groups[~test])
            withheld[test] = fold.predict(x[test])
        self.candidates = ["沿用", *[f"{n}/{a:g}" for n in NAMES for a in (.5,1.)]]
        delta = self.options(withheld)
        self.loss = (err[:,None]-delta)**2 - err[:,None]**2
        self.global_choice = np.argmin([np.mean([self.loss[ix,j].mean() for ix in self.full.units]) for j in range(delta.shape[1])])
        return self

    @staticmethod
    def options(corrections):
        return np.column_stack((np.zeros(len(corrections)),*[a*col for col in corrections.T for a in (.5,1.)]))

    def predict(self,x,anchor):
        delta = self.options(self.full.predict(x))
        z = self.full.project(x)
        distances = cdist(z,self.full.z,"sqeuclidean")
        errors, near = [], []
        for ix in self.full.units:
            chosen = ix[np.argsort(distances[:,ix],axis=1)[:,:3]]
            errors.append(self.loss[chosen].mean(1))
            near.append(np.take_along_axis(distances,chosen,axis=1).mean(1))
        order = np.argsort(np.column_stack(near),axis=1)[:,:6]
        profile = np.stack(errors,axis=1)
        local = np.take_along_axis(profile,order[:,:,None],axis=1).mean(1)
        general = np.mean([self.loss[ix].mean(0) for ix in self.full.units],axis=0)
        pick = np.argmin(local + general/2,axis=1)
        out = {n:np.maximum(0,anchor+delta[:,i]) for i,n in enumerate(self.candidates)}
        out["按所有留出轨迹统一选修正"] = np.maximum(0,anchor+delta[:,self.global_choice])
        out[PRIMARY] = np.maximum(0,anchor+delta[np.arange(len(x)),pick])
        return out, pick
