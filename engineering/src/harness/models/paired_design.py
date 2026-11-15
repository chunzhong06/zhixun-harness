"""根据已购买的共同标气估计新器件的响应，再选择可能改变分档结论的测量。"""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from harness.models.calibration_query import CalibrationQuery


ADAPT = "根据芯片实际响应预测配方池的分档错误"
NO_SITE = "按相同算法只模拟源读数"
GEOMETRY = "根据芯片响应的测点覆盖"
DIRECT = "固定挑靠近分界且多样的配方"
UNIFORM = "标签分层后按源谱覆盖"
RULES = (NO_SITE,GEOMETRY,DIRECT,UNIFORM,ADAPT)
CHOICE = "按真实装机留出误差发布"


class PairedDesign(CalibrationQuery):
    def __init__(self,src,chem,index,base):
        super().__init__(src,chem,index,base)
        self.scaler = StandardScaler().fit(src)
        self.axes = PCA(n_components=12,whiten=True,svd_solver="full").fit(self.scaler.transform(src))
        self.z = self.axes.transform(self.scaler.transform(src))
        spread = np.median(self.distance[self.distance>0])
        self.k = np.exp(-self.distance/spread)
        self.bucket = np.searchsorted(np.quantile(self.y,np.arange(1,9)/9),self.y)

    def next(self,rule,used,measured):
        if len(used)!=len(measured):
            raise ValueError("选中配方与实际反馈数量不同")
        # 共同启动样本不依赖目标设备，用于实测初始响应。
        if len(used)<8:
            if not used:return 0
            score = self.distance[:,used].min(1)
        elif rule in (DIRECT,UNIFORM):
            diversity = self.distance[:,used].min(1)
            if rule==DIRECT:
                score = diversity*(.15+self.edge)
            else:
                counts = np.bincount(self.bucket[used],minlength=9)
                score = diversity/(counts[self.bucket]+1)**2
        else:
            # 从源芯片波形推算目标芯片的对应波形。只在已经测过的配方使用实际目标波形。
            if rule==NO_SITE:
                actual = self.z[used]
            else:
                actual = self.axes.transform(self.scaler.transform(np.stack(measured)))
            offset = (actual-self.z[used]).mean(0)
            fit = solve(self.k[np.ix_(used,used)]+.12*np.eye(len(used)),actual-self.z[used]-offset,assume_a="pos")
            projected = self.z+offset+self.k[:,used]@fit
            mean,sd = actual.mean(0),np.maximum(actual.std(0),.1)
            projected = (projected-mean)/sd
            measured_z = (actual-mean)/sd
            pair = cdist(measured_z,measured_z,"sqeuclidean")/len(sd)
            distance = cdist(projected,measured_z,"sqeuclidean")/len(sd)
            if rule==GEOMETRY:
                score = distance.min(1)
            else:
                kernel = np.exp(-pair/2)
                sim = np.exp(-distance/2)
                baseline = np.mean(self.y[used])
                estimated = baseline+sim@solve(kernel+.1*np.eye(len(used)),self.y[used]-baseline,assume_a="pos")
                # 真实浓度来自现成的源标气说明；只有目标读数是未购、须推算的量。
                disagreement = np.abs(estimated-self.y)/self.band
                confidence = np.exp(-np.minimum(abs(self.y-self.cut),abs(estimated-self.cut))/self.band)
                wrong = (estimated>self.cut)!=(self.y>self.cut)
                uncertainty = np.maximum(.01,1-np.sum(sim*solve(kernel+.1*np.eye(len(used)),sim.T,assume_a="pos").T,axis=1))
                score = (uncertainty+.1)**.5 * (1+disagreement+2*wrong) * (.35+confidence)
        score[used] = -np.inf
        return int(np.argmax(score))


class KernelReadout:
    def fit(self,src,chem,ix,target,j):
        self.source_scale = StandardScaler().fit(src)
        self.axes = PCA(n_components=16,whiten=True,svd_solver="full").fit(self.source_scale.transform(src))
        zs = self.axes.transform(self.source_scale.transform(src[ix]))
        zt = self.axes.transform(self.source_scale.transform(target))
        self.z = zt
        self.delta = zs-zt
        self.y = chem[ix,j]
        self.x = self.axes.transform(self.source_scale.transform(src))
        self.labels = chem[:,j]
        d = cdist(self.x,self.x,"sqeuclidean")/16
        self.source_coef = solve(np.exp(-d/2)+.06*np.eye(len(src)),self.labels-self.labels.mean(),assume_a="pos")
        dist = cdist(zt,zt,"sqeuclidean")/16
        self.scale = max(.01,float(np.median(dist[np.triu_indices(len(zt),1)])))
        self.ref_kernel = np.exp(-dist/self.scale)+.06*np.eye(len(zt))
        self.mapping = solve(self.ref_kernel,self.delta-self.delta.mean(0),assume_a="pos")
        self.direct = solve(self.ref_kernel,self.y-self.y.mean(),assume_a="pos")
        return self

    def predict(self,x):
        z = self.axes.transform(self.source_scale.transform(x))
        k = np.exp(-cdist(z,self.z,"sqeuclidean")/(16*self.scale))
        aligned = z+self.delta.mean(0)+k@self.mapping
        p = np.exp(-cdist(aligned,self.x,"sqeuclidean")/32)@self.source_coef+self.labels.mean()
        return np.column_stack((p,self.y.mean()+k@self.direct))


class QueryReadouts:
    """在装机折中只比较真实可重训练的读法，最后用全部已购买配方重训。"""
    names = ("目标响应逆推","多成分响应逆推","源谱非线性配准","目标谱局部回归","单秒论文续训","目标岭回归","论文与目标响应各半")

    def fit(self,src,chem,indices,measured,j):
        from harness.models.mixture_response import CompositionModel
        from harness.models.year_gas_calibration import GasBase
        from sklearn.linear_model import Ridge
        self.kernel = KernelReadout().fit(src,chem,indices,measured,j)
        self.inverse = [CompositionModel().fit(src,chem,measured,chem[indices],j,many,mass) for many,mass in ((False,.5),(True,0.))]
        self.base = GasBase().fit(src,chem[:,j])
        self.w,self.b = self.base.transfer(measured,chem[indices,j])
        self.scale = StandardScaler().fit(measured)
        self.ridge = Ridge(alpha=10).fit(self.scale.transform(measured),chem[indices,j])
        return self

    def predict(self,x):
        one,multi = [model.predict(x) for model in self.inverse]
        kernels = self.kernel.predict(x)
        paper = (self.base.scaler.transform(x)@self.w+self.b)*self.base.sd+self.base.mean
        return np.column_stack((one,multi,kernels,paper,self.ridge.predict(self.scale.transform(x)),(paper+one)/2))


class QueryDeployment:
    def fit(self,src,chem,ix,measured,j):
        from harness.models.year_sensor_pairs import PairedChoice
        self.cut = float(np.median(chem[:,j]))
        fold = np.arange(len(ix))%4
        held = np.empty((len(ix),len(QueryReadouts.names)))
        for k in range(4):
            test = fold==k
            keep = np.ones(len(src),bool)
            keep[ix[test]] = False
            tr = np.cumsum(keep)[ix[~test]]-1
            trained = QueryReadouts().fit(src[keep],chem[keep],tr,measured[~test],j)
            held[test] = trained.predict(measured[test])
        self.selector = PairedChoice().fit(held,held.std(1)/max(chem[ix,j].std(),1),chem[ix,j],self.cut)
        self.sd = max(chem[ix,j].std(),1)
        self.model = QueryReadouts().fit(src,chem,ix,measured,j)
        return self

    def predict(self,x):
        from harness.models.year_sensor_pairs import PRIMARY
        bank = self.model.predict(x)
        selected = self.selector.predict(bank,bank.std(1)/self.sd)
        p = dict(zip(QueryReadouts.names,bank.T,strict=True))
        p[CHOICE] = selected[PRIMARY]
        p["新工具统一装机配比"] = selected["统一装机验证配对权重"]
        return p
