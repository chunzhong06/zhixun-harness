"""为一趟航班选择误差校正范围，只用以往已知答案的其他退化轨迹试验。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

WIDTHS=(.5,1.,2.,4.)
PRIMARY="在相近历史整机上试验后校正寿命"
NONE="保持联合部件模型"
GLOBAL="历史设备等权平均修正"


def context(x,bank):
    profile=x[:,0,:,:38]
    # 工况内中位偏差比整趟时间顺序更容易跨航线比较。分别保留水平和跨工况离散程度。
    return np.column_stack((np.median(profile,axis=1),np.std(profile,axis=1),bank[:,3],bank[:,:3]-bank[:,3:4]))


class LocalTrials:
    def fit(self,x,bank,truth,group):
        self.scale=RobustScaler().fit(x)
        self.pca=PCA(n_components=8,svd_solver="full").fit(self.scale.transform(x).clip(-6,6))
        pc=self.pca.transform(self.scale.transform(x).clip(-6,6))
        self.metric=RobustScaler().fit(pc)
        self.refs=self.metric.transform(pc)
        self.group=group.copy()
        self.errors=truth-bank[:,3]
        self.counts={g:int(np.sum(group==g)) for g in np.unique(group)}
        self.groups=np.array(list(self.counts))
        self.prior=float(np.mean([self.errors[group==g].mean() for g in self.groups]))
        # 每个历史航班自身的校正必须用另外整台设备的已知误差来试，不能用自己的寿命。
        offsets=self.corrections(self.refs,leave_group=group)
        self.loss=np.column_stack((self.errors**2,(self.errors[:,None]-offsets)**2))
        return self

    def neighbors(self,z,leave_group=None):
        d=cdist(z,self.refs,"sqeuclidean")/self.refs.shape[1]
        if leave_group is not None:d=np.where(leave_group[:,None]==self.group,np.inf,d)
        # 每台最相近的已完成航班贡献一次；不会让某台拥有更多采样点就支配试用。
        idx=np.column_stack([np.flatnonzero(self.group==g)[np.argmin(d[:,self.group==g],axis=1)] for g in self.groups])
        return idx,np.take_along_axis(d,idx,1)

    def corrections(self,z,leave_group=None):
        idx,d=self.neighbors(z,leave_group)
        outputs=[]
        for width in WIDTHS:
            w=np.exp(-.5*d/width**2)
            offsets=(w*self.errors[idx]).sum(1)/(w.sum(1)+2.)
            outputs.append(offsets)
        return np.column_stack(outputs)

    def predict(self,x,bank):
        pc=self.pca.transform(self.scale.transform(x).clip(-6,6))
        z=self.metric.transform(pc)
        idx,d=self.neighbors(z)
        offsets=self.corrections(z)
        anchor=bank[:,3]
        out={NONE:anchor,GLOBAL:np.maximum(0,anchor+self.prior)}
        for j,v in enumerate(WIDTHS):
            out[f"固定邻域宽度{v:g}"]=np.maximum(0,anchor+offsets[:,j])
        # 候选在各台历史临近点留下实际试用误差。优于保持原版才允许应用该尺度。
        w=np.exp(-.5*d)
        relative=self.loss[idx]-self.loss[idx][:,:,:1]
        risk=(w[:,:,None]*relative).sum(1)/(1+w.sum(1)[:,None])
        selected=np.argmin(risk,axis=1)
        chosen=np.column_stack((np.zeros(len(x)),offsets))[np.arange(len(x)),selected]
        out[PRIMARY]=np.maximum(0,anchor+chosen)
        out["邻域现场修正半量"]=np.maximum(0,anchor+.5*chosen)
        return out,selected
