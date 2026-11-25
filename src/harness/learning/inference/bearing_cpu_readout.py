"""在本轮训练行上拟合固定的径向基函数读出，保存逐行 CPU 推理所需的数值。"""
from io import BytesIO
import numpy as np

class BearingCPUReadout:
    def __init__(self, payload, width):
        with np.load(BytesIO(payload.tobytes()), allow_pickle=False) as saved:
            self.quantiles = saved["quantiles"]
            self.centers = saved["centers"]
            self.coef = saved["coef"]
            self.offset = float(saved["offset"].item())
        if (self.quantiles.ndim != 2 or self.quantiles.shape[1] != width
                or not 2 <= len(self.quantiles) <= 128 or np.any(np.diff(self.quantiles, axis=0) < 0)
                or self.centers.ndim != 2 or self.centers.shape[1] != width
                or not 1 <= len(self.centers) <= 512 or self.coef.shape != (len(self.centers),)
                or not all(np.isfinite(a).all() for a in (self.quantiles,self.centers,self.coef,self.offset))):
            raise ValueError("轴承 CPU 读出维度或数值无效")
        self.references = np.linspace(0.,1.,len(self.quantiles))
        self.norms = np.einsum("ij,ij->i",self.centers,self.centers)

    def transform(self, raw):
        from scipy.special import ndtri
        table = self.quantiles
        ranks = np.empty(raw.shape, dtype=np.float64)
        for j, grid in enumerate(table.T):
            x = raw[:,j]
            # 重复分位数的左右秩取中值，边界外的传感器读数截到训练分布两端。
            ranks[:,j] = (np.interp(x,grid,self.references)
                          - np.interp(-x,-grid[::-1],-self.references[::-1]))*.5
            ranks[x>=grid[-1],j] = 1.
            ranks[x<=grid[0],j] = 0.
        return ndtri(np.clip(ranks,1e-7,1.-1e-7))

    def basis(self, x):
        dots = np.einsum("ik,jk->ij",x,self.centers) if len(x)==1 else x @ self.centers.T
        distance = np.einsum("ij,ij->i",x,x)[:,None]+self.norms[None,:]-2.*dots
        return np.exp(-.015*np.maximum(distance,0.))

    def predict(self, raw):
        return np.maximum(0.,self.basis(self.transform(raw)) @ self.coef+self.offset)
