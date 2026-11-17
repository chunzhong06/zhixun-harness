"""以物理释放为训练单位，安装时比较不同片段取法，再学习它们适用的信号范围。"""
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MEAN="整段合并后训练"
WINDOWS="展开所有片段后训练"
PEAK="高响应片段展开训练"
SHAPE="各片段形状展开训练"
DIRECT="完整时间轮廓训练"
FIXED=(MEAN,WINDOWS,PEAK,SHAPE,DIRECT)
CV="装机按全体试验选一种训练办法"
PRIMARY="装机按信号强弱和波动实试训练办法"
VOTE="五种训练办法固定等权"


def read_windows(data):
    a=np.asarray(data,dtype=float)
    if a.ndim!=2 or a.shape[1]!=11 or not np.isfinite(a).all() or np.any(np.diff(a[:,0])<0):
        raise ValueError("需要按原时钟排列的八路风洞传感器和环境记录")
    t=a[:,0];raw=a[:,3:]
    if np.any((raw<=0)|(raw>=3110)) or t[0]>=10 or t[-1]<285:
        raise ValueError("测量覆盖或电阻量程不符")
    z=np.log(raw/(3110-raw))
    base=np.median(z[t<60],axis=0)
    def window(lo,hi):
        subset=z[(t>=lo)&(t<hi)]-base
        if len(subset)<2:raise ValueError("测量缺少完整时间段")
        return np.r_[subset.mean(0),np.quantile(subset,.85,axis=0)-np.quantile(subset,.15,axis=0)]
    return np.stack([window(lo,lo+10) for lo in range(60,240,10)])


def strength(x):
    # 同一次释放的片段可比较幅度；排序只使用现场读数。
    return np.sqrt(np.mean(x[:,:,:8]**2,axis=2))


def views(x):
    amp=strength(x)
    order=np.argsort(amp,axis=1)
    strong=np.take_along_axis(x,order[:,-6:,None],axis=1)
    norm=x/(.015+amp[:,:,None])
    whole=np.concatenate((x.mean(1),np.quantile(x,.8,axis=1)),axis=1)[:,None,:]
    profile=x.reshape(len(x),3,6,16).mean(2).reshape(len(x),1,-1)
    return dict(zip(FIXED,(whole,x,strong,norm,profile),strict=True))


def context(x):
    amp=strength(x)
    mean=x[:,:,:8].mean(1)
    size=np.sqrt((mean**2).mean(1))
    return np.column_stack((np.log(.01+size),np.log(.01+amp.std(1)),mean/(.02+size[:,None])))


class PlumeTools:
    def fit(self,x,y):
        self.models={}
        for name,rows in views(x).items():
            model=make_pipeline(StandardScaler(),SVC(C=3.,gamma="scale",class_weight="balanced"))
            self.models[name]=model.fit(rows.reshape(-1,rows.shape[2]),np.repeat(y,rows.shape[1]))
        return self

    def margins(self,x):
        scores=[]
        for name,rows in views(x).items():
            pred=self.models[name].decision_function(rows.reshape(-1,rows.shape[2])).reshape(rows.shape[:2])
            scores.append(pred.mean(1))
        return np.column_stack(scores)


class PlumeInstallation:
    def fit(self,x,y,group):
        # 两种留出各检验一次：同一配方的另一次释放、两次重复整组未见的配方。
        _,code=np.unique(group,axis=0,return_inverse=True)
        indices=np.arange(len(y))
        if not np.all(np.bincount(code)==2):raise ValueError("装机需每种配方两次独立释放")
        rep=np.zeros(len(y),int)
        for i in set(code):rep[indices[code==i][1]]=1
        trials=[]
        for fold in (rep,code%5):
            held=np.empty((len(y),len(FIXED)))
            for i in np.unique(fold):
                te=fold==i
                held[te]=PlumeTools().fit(x[~te],y[~te]).margins(x[te])
            trials.append(held)
        cuts=(-.5,-.25,0.,.25,.5)
        self.cuts=np.array([max(cuts,key=lambda c:(np.mean([f1_score(y,1+(v[:,j]>=c),average="macro") for v in trials]),-abs(c))) for j in range(len(FIXED))])
        rated=[[f1_score(y,1+(v[:,j]>=self.cuts[j]),average="macro") for j in range(len(FIXED))] for v in trials]
        self.quality=np.asarray(rated).mean(0)
        self.selected=int(np.argmax(self.quality))
        self.trials=np.mean([expit(2*(v-self.cuts)) for v in trials],axis=0)
        self.y=y.copy()
        self.scale=StandardScaler().fit(context(x))
        self.ref=self.scale.transform(context(x))
        self.tools=PlumeTools().fit(x,y)
        return self

    def predict(self,x):
        p=expit(2*(self.tools.margins(x)-self.cuts))
        cols=dict(zip(FIXED,p.T,strict=True))
        cols[CV]=p[:,self.selected]
        cols[VOTE]=p.mean(1)
        cols["按装机整体验证分固定加权"]=p@np.exp(8*self.quality)/np.exp(8*self.quality).sum()
        cols["训练候选固定中位"]=np.median(p,axis=1)
        # 本次可见幅度、波动、通道形状决定哪些已带标签的装机试验更相似，评价仍按完整释放。
        z=self.scale.transform(context(x))
        near=np.exp(-.5*np.mean((z[:,None]-self.ref[None])**2,axis=2))**3
        local=[]
        for w in near:
            loss=np.empty(len(FIXED))
            for j in range(len(FIXED)):
                err=(self.trials[:,j]-(self.y==2))**2
                loss[j]=np.mean([np.average(err[self.y==c],weights=w[self.y==c]+.1) for c in (1,2)])
            a=np.exp(-16*(loss-loss.min()))
            local.append(a/a.sum())
        cols[PRIMARY]=(np.sum(p*np.array(local),axis=1)+cols[CV])/2
        return {k:1+(v>=.5).astype(int) for k,v in cols.items()},cols
