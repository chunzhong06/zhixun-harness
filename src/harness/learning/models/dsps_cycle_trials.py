"""在一次真实暴露的五次稳态循环上实试，比较跨循环训练与先合并再训练。"""
import numpy as np
from harness.learning.models.dsps_dose_contrast import InstalledDoseRelease, EQUAL, PAPER

MEAN="五次循环先求平均再训练"
EXPANDED="展开五个循环训练后固定平均概率"
PRIMARY="在当前暴露中按各循环留出读数的复测偏差发放"


def cycles(stable):
    values=np.asarray(stable,dtype=float)
    if values.shape!=(500,8,16) or not np.isfinite(values).all():
        raise ValueError("稳态输入须含五个完整加热周期")
    return np.median(values[:,:,7].reshape(5,20,5,8),axis=2).reshape(5,160)


def features(rows):
    # 当前这组读法只用稳态分量，第二槽保留既有装机模型的输入形状。
    return np.stack((rows,np.zeros_like(rows)),axis=1)


class CycleTrials:
    def fit(self,a,b,y,g,level):
        self.readers=[]
        for source,target,repeat in ((np.median(a,1),np.median(b,1),1),(a.mean(1),b.mean(1),1),
                                     (a.reshape(-1,160),b.reshape(-1,160),5)):
            m=InstalledDoseRelease().fit(features(source),features(target),np.repeat(y,repeat),np.repeat(g,repeat),np.repeat(level,repeat))
            self.readers.append(m)
        self.scale=np.std(b.reshape(-1,160),axis=0).clip(1e-5)
        return self

    def predict(self,b):
        n=len(b)
        probs=[]
        for m,values in zip(self.readers,(np.median(b,1),b.mean(1),b.reshape(-1,160)),strict=True):
            p,_=m.probabilities(features(values));probs.append(p)
        single=probs[2][EQUAL].reshape(n,5,3)
        out={EQUAL:probs[0][EQUAL],PAPER:probs[0][PAPER],MEAN:probs[1][EQUAL],EXPANDED:single.mean(1),
             "直接标准化用循环平均":probs[1][PAPER],"直接标准化展开循环":probs[2][PAPER].reshape(n,5,3).mean(1)}
        for i in range(5):out[f"固定发第{i+1}循环"]=single[:,i]
        # 每个循环与其余四次真实测量的典型形状比较，给稳定的循环更大的发放权重。
        held=np.stack([np.median(np.abs((b[:,i]-np.median(np.delete(b,i,axis=1),axis=1))/self.scale),axis=1) for i in range(5)],axis=1)
        w=(.03+held)**-2
        field=(w[:,:,None]*single).sum(1)/w.sum(1)[:,None]
        out["只按循环复测试选"]=field
        out[PRIMARY]=(field+out[EQUAL])/2
        out["五次固定平均与原版各半"]=(out[EXPANDED]+out[EQUAL])/2
        out["三读法固定平均"]=(out[EQUAL]+out[MEAN]+out[EXPANDED])/3
        return {k:1+p.argmax(1) for k,p in out.items()},held
