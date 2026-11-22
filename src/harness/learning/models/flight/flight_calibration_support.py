"""以整机队为装机试验，比较有限范围内的误差修正及保留原寿命。"""
import numpy as np
from scipy.spatial.distance import cdist

from harness.learning.models.flight.flight_error_surface import ErrorSurface


PRIMARY = "依机队留出误差与当前覆盖范围发布"
NAMES = ("保持原寿命","已知平均偏差各半","三近例中位偏差","局部斜率限制在实测偏差内","近六轨迹实测偏差")


class SupportedTools:
    def fit(self,x,y,anchor,groups):
        self.surface = ErrorSurface().fit(x,y-anchor,groups)
        return self

    def predict(self,x):
        surface = self.surface
        d = cdist(surface.project(x),surface.z,"sqeuclidean")
        # 跨轨迹找参照；服务点到它们的距离，与参照之间的实测距离使用相同尺度。
        by_unit = np.stack([ix[np.argmin(d[:,ix],axis=1)] for ix in surface.units],axis=1)
        ds = np.take_along_axis(d,by_unit,axis=1)
        order = np.argsort(ds,axis=1)[:,:6]
        nearest = np.take_along_axis(by_unit,order,axis=1)
        observed = surface.residual[nearest]
        adjacent = surface.z[nearest]
        separation = np.sum((adjacent[:,:,None]-adjacent[:,None,:])**2,axis=3)
        ii,jj = np.triu_indices(nearest.shape[1],1)
        spread = np.median(separation[:,ii,jj],axis=1)
        depth = np.median(np.take_along_axis(d,nearest,axis=1),axis=1)/np.maximum(.01,spread)
        trust = 1 / np.maximum(1,depth)
        corrections = surface.predict(x)
        low, high = np.quantile(observed,[.2,.8],axis=1)
        bounded = np.clip(corrections[:,2],low,high)
        delta = np.column_stack((np.zeros(len(x)),np.full(len(x),surface.bias/2),
                                 np.median(observed[:,:3],axis=1),bounded,corrections[:,1]))
        signs = np.abs(np.sign(observed).mean(1))
        return delta,trust,np.column_stack((np.log1p(depth),signs,np.median(abs(observed),axis=1)))


class FleetInstallation:
    def fit(self,x,y,anchor,groups,sites):
        self.full = SupportedTools().fit(x,y,anchor,groups)
        self.history = []
        for site in np.unique(sites):
            test = sites == site
            keep = (~test) & (~np.isin(groups,groups[test]))
            trial = SupportedTools().fit(x[keep],y[keep],anchor[keep],groups[keep])
            delta,trust,context = trial.predict(x[test])
            bank = self.outputs(delta,trust)
            err = y[test]-anchor[test]
            cost = (bank-err[:,None])**2-err[:,None]**2
            self.history.append((context,cost))
        self.avg = np.mean([cost.mean(0) for _,cost in self.history],axis=0)
        return self

    @staticmethod
    def outputs(delta, trust):
        return np.column_stack((delta,delta[:,2:]*trust[:,None]/2))

    def predict(self,x,anchor):
        delta,trust,context = self.full.predict(x)
        bank = self.outputs(delta,trust)
        # 各历史机队等权，各机队只用与本次证据最接近的十分之一航班估计真实净收益。
        cost = []
        for ref,loss in self.history:
            scale = np.maximum(np.std(ref,axis=0),(.15,.15,1.))
            d = cdist(context/scale,ref/scale,"sqeuclidean")
            ix = np.argsort(d,axis=1)[:,:max(5,len(ref)//10)]
            cost.append(loss[ix].mean(1))
        score = (np.mean(cost,axis=0)+self.avg)/2
        choices = score.argmin(1)
        cols = [*NAMES,*["按参照覆盖折半/"+n for n in NAMES[2:]]]
        p = dict(zip(cols,np.maximum(0,anchor[:,None]+bank).T,strict=True))
        p["按整机队试验统一选一法"] = p[cols[self.avg.argmin()]]
        p[PRIMARY] = np.maximum(0,anchor+bank[np.arange(len(x)),choices])
        return p
