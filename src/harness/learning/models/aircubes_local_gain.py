"""按已到达的浓度范围与电极响应，为强校准器估计局部斜率。"""
import numpy as np
from harness.learning.models.aircubes_stack import StackMaintenance

INCUMBENT="现场统一重估六工具配比"
PRIMARY="按本机回标覆盖重新拟合高浓度响应"
WIDTHS=(.45,1.,2.)


class LocalGain:
    def __init__(self,tools):
        self.old=StackMaintenance(tools)
        self.scaler=tools["joint"].scale
        self.records=[]
        self.risk=np.ones(5)*5.

    def predict(self,t,x,base):
        out=self.old.predict(t,x,base)
        p=out[INCUMBENT]
        z=np.clip(self.scaler.transform(x),-7,7)
        context=np.column_stack((p/10,(z[:,0]-z[:,1])/2,z[:,2]/2,(z[:,5]+z[:,6])/4))
        cols=[p];support=np.zeros_like(p)
        if self.records:
            locations,errors,stamps=zip(*self.records,strict=True)
            locations=np.stack(locations);errors=np.array(errors)
            age=(t[:,None]-np.array(stamps)[None,:])/86400
            dist=(context[:,None,:]-locations[None,:,:])**2
            gap=locations[None,:,0]-context[:,None,0]
            for width in WIDTHS:
                weights=np.exp(-age/20-dist[:,:,0]/(2*width**2)-dist[:,:,1:].mean(2)/2)
                mass=weights.sum(1);slope_mass=(weights*gap).sum(1)
                second=(weights*gap**2).sum(1)+8
                # 带收缩的局部一次回归，向原装机解收缩；截距给出当前浓度的改动。
                rhs=weights@errors;rhs_slope=(weights*gap)@errors
                correction=(rhs-slope_mass*rhs_slope/second)/(mass+5-slope_mass**2/second)
                cols.append(np.maximum(0,p+correction))
                if width==1.:support=mass
            take_high=np.clip((p-20)/15,0,1)
            cols.append(p+take_high*(cols[2]-p))
        else:
            cols.extend([p]*4)
        bank=np.column_stack(cols)
        names=["保持原全局读法",*[f"固定局部电极响应带宽{v:g}" for v in WIDTHS],"只在高读数修正斜率"]
        out.update(dict(zip(names,bank.T,strict=True)))
        w=(.5+self.risk)**-2
        # 窄带训练受附近真值数量限制；支持不够时退回已装机的全局读法。
        share=support/(support+6)
        out[PRIMARY]=p+share*((bank@w/w.sum())-p)
        out["相同覆盖但冻结选法"]=p+share*(bank.mean(1)-p)
        self.pending=(context[-1].copy(),t[-1],bank[-1].copy())
        return out

    def update(self,y):
        self.old.update(y)
        ctx,t,bank=self.pending
        self.records.append((ctx,np.clip(y-bank[0],-14,14),t))
        self.records=self.records[-256:]
        self.risk=.95*self.risk+.05*np.minimum((y-bank)**2,400)
