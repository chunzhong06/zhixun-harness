"""给不同装机询问顺序使用同一个已验证的气体分类和论文比较模型。"""
import numpy as np
from harness.learning.models.mixture_response import CompositionModel
from harness.learning.models.year_gas_calibration import GasBase,YearTools
from harness.learning.models.year_sensor_pairs import PairedYears,PairedChoice


def table(old,paired):
    return np.column_stack(((old[:,3]+old[:,4])/2,paired,(old[:,5]+old[:,4])/2))


class InstallReadout:
    def fit(self,base,sx,chem,ix,measured,gas_index,gas):
        y = chem[:,gas_index]
        self.cut = float(np.median(y))
        self.gas = gas
        self.size = len(ix)
        self.paired = PairedYears().fit(base,sx[ix],measured,y[ix])
        self.comp = CompositionModel().fit(sx,chem,measured,chem[ix],gas_index,gas=="toluene",0. if gas=="toluene" else .5)
        self.old = YearTools().fit(base,measured,y[ix])
        if self.size==20 and gas=="acetone":
            # 强固定20瓶方案以真实整配方留出确定权重，每折从源端也移除相同配方。
            held = np.empty((self.size,7))
            dist = np.empty(self.size)
            fold = np.arange(self.size)%4
            for j in range(4):
                test = fold==j
                keep = np.ones(len(sx),bool)
                keep[ix[test]] = False
                b = GasBase().fit(sx[keep],y[keep])
                maps = PairedYears().fit(b,sx[ix[~test]],measured[~test],y[ix[~test]])
                pairs,dist[test] = maps.predict(measured[test])
                old = YearTools().fit(b,measured[~test],y[ix[~test]]).outputs(measured[test])
                held[test] = table(old,pairs)
            self.selector = PairedChoice().fit(held,dist,y[ix],self.cut)
        return self

    def predict(self,x):
        pairs,dist = self.paired.predict(x)
        curve = pairs[:,3]
        comp = self.comp.predict(x)
        old = self.old.outputs(x)
        bank = table(old,pairs)
        if self.gas=="acetone" and self.size==20:
            uni = self.selector.predict(bank,dist)["统一装机验证配对权重"]
            strong = (uni+comp)/2
            paper = bank[:,-1]
        elif self.gas=="acetone":
            strong = comp
            paper = (curve+comp)/2
        elif self.gas=="toluene":
            paper = (curve+comp)/2
            strong = paper if self.size==20 else comp
        else:
            raise ValueError("发布规则未支持该气体")
        return {"按气体与预算固定":strong,"按气体与预算论文":paper,"单用论文续训":old[:,1],"全部周期均值/论文与校正各半":old[:,5],
                "单用成分响应":comp,"全曲线回源":curve,"目标单训":old[:,4]}
