"""在目标芯片的配对标气上按混合物分折，试验跨器件校准与不同周期训练做法。"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ARMS=("源芯片直接","目标标气单训","源与目标全局校准","装机平均偏移","跨元件直接标准化","同元件分段直接标准化")
MODES=("先合并周期","先分别训练周期")
CV="装机跨混合物试验选一版"
PRIMARY="按装机试验的可靠程度合并通过的训练办法"
PAPER="全局校准按相关性选特征和偏最小二乘"


def mean_inputs(groups):
    return np.stack([g.mean(0) for g in groups])


def fit_rows(groups,y,expand):
    if expand:
        return np.concatenate(groups),np.repeat(y,[len(v) for v in groups])
    return mean_inputs(groups),y


def ridge(x,y):
    return make_pipeline(StandardScaler(),Ridge(alpha=3.)).fit(x,y)


class TransferBank:
    def fit(self,source,labels,target,paired,expand):
        xs,ys=fit_rows(source,labels,expand)
        xt,yt=fit_rows(target,labels[paired],expand)
        self.source=ridge(xs,ys);self.target=ridge(xt,yt)
        self.pool=ridge(np.vstack((xs,xt)),np.r_[ys,yt])
        # 仪器信息和气体标签对所有对手开放；映射按混合物一对一拟合，不假定循环序号必须相同。
        xp=mean_inputs([source[i] for i in paired]);tp=mean_inputs(target)
        self.shift=xp.mean(0)-tp.mean(0)
        self.in_scale=StandardScaler().fit(tp);self.out_scale=StandardScaler().fit(xp)
        a=self.in_scale.transform(tp);b=self.out_scale.transform(xp)
        self.full=Ridge(alpha=2.).fit(a,b)
        self.parts=[Ridge(alpha=2.).fit(x,z) for x,z in zip(np.split(a,4,axis=1),np.split(b,4,axis=1),strict=True)]
        self.expand=expand
        return self

    def predict(self,groups):
        x=np.concatenate(groups) if self.expand else mean_inputs(groups)
        t=self.in_scale.transform(x)
        xb=self.out_scale.inverse_transform(self.full.predict(t))
        parts=self.out_scale.inverse_transform(np.column_stack([m.predict(ti) for m,ti in zip(self.parts,np.split(t,4,axis=1),strict=True)]))
        cols=np.column_stack((self.source.predict(x),self.target.predict(x),self.pool.predict(x),
                              self.source.predict(x+self.shift),self.source.predict(xb),self.source.predict(parts)))
        if self.expand:
            return np.stack([v.mean(0) for v in np.split(cols,np.cumsum([len(g) for g in groups])[:-1])])
        return cols


class CalibrationPaper:
    """Robin 等的全局特征校准本地适配：固定温度段线性特征，按相关性排序，源/装机联合PLSR。"""
    def fit(self,source,y,target,paired,count):
        a,b=mean_inputs(source),mean_inputs(target);x=np.vstack((a,b));label=np.r_[y,y[paired]]
        self.scale=StandardScaler().fit(x);z=self.scale.transform(x)
        self.keep=np.argsort(np.abs(z.T@(label-label.mean())))[-count:]
        self.model=PLSRegression(n_components=min(8,count),scale=False,max_iter=500).fit(z[:,self.keep],label)
        return self

    def predict(self,groups):
        return self.model.predict(self.scale.transform(mean_inputs(groups))[:,self.keep]).ravel()


def balanced_f1(y,pred,threshold):
    return f1_score(y>threshold,pred>threshold,average="macro")


class InstallCycles:
    def fit(self,source,y,target,paired,folds):
        # 折内留出的配方同时从源建模和目标标定中移出，避免仅读到相同源配方就能猜到试验答案。
        self.names=[f"{arm}/{mode}" for mode in MODES for arm in ARMS]
        self.threshold=float(np.median(y))
        scores=np.empty((len(paired),len(self.names)))
        papers=np.empty((len(paired),3))
        for fold in np.unique(folds):
            test=folds==fold
            mask=np.ones(len(y),bool);mask[paired[test]]=False
            lookup=np.cumsum(mask)-1
            s=[g for g,ok in zip(source,mask,strict=True) if ok]
            tx=[g for g,ok in zip(target,~test,strict=True) if ok]
            te=[g for g,ok in zip(target,test,strict=True) if ok]
            ix=lookup[paired[~test]]
            scores[test]=np.column_stack([TransferBank().fit(s,y[mask],tx,ix,mode==MODES[1]).predict(te) for mode in MODES])
            papers[test]=np.column_stack([CalibrationPaper().fit(s,y[mask],tx,ix,n).predict(te) for n in (24,64,128)])
        calibration_y=y[paired]
        self.quality=np.array([balanced_f1(calibration_y,p,self.threshold) for p in scores.T])
        errors=((scores-calibration_y[:,None])**2).mean(0)**.5
        rank=np.lexsort((errors,-self.quality))
        self.best=int(rank[0]);short=rank[self.quality[rank]>=self.quality[self.best]-.035][:4]
        w=np.zeros(len(rank));w[short]=1/np.maximum(errors[short],1.)**2
        self.weight=w/w.sum()
        self.banks=[TransferBank().fit(source,y,target,paired,i==1) for i in (0,1)]
        pick=np.argmin(np.mean((papers-calibration_y[:,None])**2,0))
        self.paper=CalibrationPaper().fit(source,y,target,paired,(24,64,128)[pick])
        self.trial=scores
        return self

    def predict(self,groups):
        bank=np.column_stack([m.predict(groups) for m in self.banks]);paper=self.paper.predict(groups)
        p=dict(zip(self.names,bank.T,strict=True))
        p.update({PAPER:paper,CV:bank[:,self.best],PRIMARY:bank@self.weight,
                  "十二工具固定平均":bank.mean(1),"固定中位模型浓度":np.median(bank,axis=1)})
        for j,name in enumerate(ARMS):
            p[name+"/两种训练顺序各半"]=(bank[:,j]+bank[:,j+len(ARMS)])/2
        for k in self.names:
            p["论文与"+k+"各半"]=(paper+p[k])/2
        return p
