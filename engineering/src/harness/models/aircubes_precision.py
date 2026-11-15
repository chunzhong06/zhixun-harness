"""由夏季按时间前向试验估计校准后各时刻的误差协方差，再用真实现场回标更新。"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from harness.models.aircubes import AirCalibration
from harness.models.aircubes_retrain import ElectrodeRelease
from harness.models.aircubes_innovation import TransferRelease
from harness.models.aircubes_joint import JointRelease
from harness.models.aircubes_changes import ChangeRelease,ChangeMaintenance,INCUMBENT

CHANGE="单步浓度变化现场重训"
PRIMARY="根据已返参考的时距与现场误差估计变化量的可信度"


def evidence(clock,p,q,last,x):
    if last is None:
        h=np.full(len(clock),6.);m=np.zeros(len(clock))
    else:
        t,y=last;h=(clock-t)/3600;m=np.abs(p-y)
    return np.column_stack((np.minimum(h,18),np.log1p(m),p/15,np.log1p(abs(q-p)),x[:,2]/12))


class PrecisionRelease:
    def fit(self,t,x,y):
        ts=np.linspace(t[0],t[-1]+1,5)
        ctx=[];banks=[];truth=[]
        # 三次源期前向安装，允许从密集源答案评估每个年龄段；服务期不会得到密集参考。
        for i in (1,2,3):
            tr=t<ts[i];te=(t>=ts[i])&(t<ts[i+1])
            tick,src,yy=t[tr],x[tr],y[tr]
            if not np.any(te):continue
            paper=AirCalibration().fit(src[:,:5],yy)
            ele=ElectrodeRelease().fit(src[:,:5],yy)
            joint=JointRelease().fit(tick,src,yy)
            tools={"paper":paper,"electrode":ele,"joint":joint,
                   "transfer":TransferRelease().fit(tick,src[:,:5],yy,ele),
                   "changes":ChangeRelease().fit(tick,src,yy)}
            # 源试验用当时可安装的物理校正与漂移工具，避免借用看过该段标签的源端配比。
            from harness.models.aircubes_bias import DriftMaintenance
            class SourceBaseline(DriftMaintenance):
                def predict(self,time,raw,b):
                    out=super().predict(time,raw,b)
                    out[INCUMBENT]=out["论文强校正与合用当前及上一窗变化重训0.82各半"]
                    return out
            controller=ChangeMaintenance(tools,SourceBaseline(tools))
            time,raw,target=t[te],x[te],y[te];base=paper.predict(raw[:,:5]);last=None
            bounds=np.r_[0,np.flatnonzero(np.diff(time//21600))+1,len(time)]
            for a,b in zip(bounds[:-1],bounds[1:],strict=True):
                out=controller.predict(time[a:b],raw[a:b],base[a:b]);p=out[INCUMBENT];q=out[CHANGE]
                ctx.append(evidence(time[a:b],p,q,last,raw[a:b]))
                banks.append(np.column_stack((p,q)));truth.extend(target[a:b])
                if b<len(time) and (time[b-1]+600)%21600==0:
                    controller.update(target[b-1]);last=time[b-1],target[b-1]
        raw=np.vstack(ctx);bank=np.vstack(banks);target=np.array(truth)
        self.scale=StandardScaler().fit(raw)
        design=self.phi(raw);d=bank[:,1]-bank[:,0];err=target-bank[:,0]
        a=design*d[:,None]
        self.gram=a.T@a/len(a);self.rhs=a.T@err/len(a)
        self.global_w=np.clip(d@err/max(1.,d@d),0.,1.)
        return self

    def phi(self,raw):
        z=self.scale.transform(raw).clip(-3,3)
        # 加入时距与浓度变化之间的交互，同一时距可以覆盖非常不同的浓度变化。
        return np.column_stack((np.ones(len(z)),z,z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,3],
                                 np.maximum(0,z[:,2]-1.)))


class PrecisionMaintenance:
    def __init__(self,tools):
        self.old=ChangeMaintenance(tools);self.q=tools["precision"]
        self.g=np.zeros_like(self.q.gram);self.r=np.zeros_like(self.q.rhs)
        self.last=None

    def predict(self,t,x,base):
        out=self.old.predict(t,x,base);p=out[INCUMBENT];q=out[CHANGE];delta=q-p
        cov=evidence(t,p,q,self.last,x);f=self.q.phi(cov)
        target_prior=np.r_[self.q.global_w,np.zeros(f.shape[1]-1)]
        for reg in (.25,1.,4.):
            prior_g=self.q.gram+np.eye(f.shape[1])*reg
            prior_r=self.q.rhs+target_prior*reg
            fixed=np.linalg.solve(prior_g,prior_r)
            out[f"源端校准时距配比/{reg:g}"]=p+(f@fixed).clip(0,1)*delta
            updated=np.linalg.solve(40*prior_g+self.g,40*prior_r+self.r)
            out[f"回标后校准时距配比/{reg:g}"]=p+(f@updated).clip(0,1)*delta
        out["源端统一差分量"]=p+self.q.global_w*delta
        for k in (.1,.2,.3,.4):
            out[f"同工具论文固定差分量{k:g}"]=p+k*delta
        out[PRIMARY]=.5*(out["源端校准时距配比/1"]+out["回标后校准时距配比/1"])
        self.pending=t[-1],f[-1]*delta[-1],p[-1]
        return out

    def update(self,y):
        self.old.update(y);t,a,p=self.pending;self.last=t,y
        self.g=.97*self.g+np.outer(a,a)
        self.r=.97*self.r+a*(y-p)
