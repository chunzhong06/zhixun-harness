"""在单行输入上重建训练侧保存的 TabM 回归器，作者实现与训练编码保持一致。"""
import io

import numpy as np


def architecture(bins):
    import rtdl_num_embeddings as embeddings
    import tabm
    embedded = embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=8, activation=False, version="B")
    return tabm.TabM.make(n_num_features=len(bins), cat_cardinalities=[], d_out=1, num_embeddings=embedded,
                          k=8, n_blocks=2, d_block=96)


class TabMReadout:
    def __init__(self, payload, width):
        import torch
        from sklearn.preprocessing import QuantileTransformer
        if payload.dtype != np.uint8 or payload.ndim != 1 or not 0 < payload.size <= 4*1024**2:
            raise ValueError("TabM 读出须为有界的数值资产")
        with np.load(io.BytesIO(payload.tobytes()), allow_pickle=False) as saved:
            self.keep = saved["keep"]
            if self.keep.dtype != bool or self.keep.shape != (width,) or not self.keep.any():
                raise ValueError("TabM 特征列与父模型不一致")
            bins = [torch.from_numpy(saved["bin."+str(i)]) for i in range(int(self.keep.sum()))]
            self.transform = QuantileTransformer(output_distribution="normal")
            self.transform.quantiles_ = saved["quantiles"]
            self.transform.references_ = saved["references"]
            self.transform.n_quantiles_, self.transform.n_features_in_ = self.transform.quantiles_.shape
            if self.transform.n_features_in_ != width or self.transform.references_.shape != (self.transform.n_quantiles_,):
                raise ValueError("TabM 源分位数形状不符")
            self.center, self.scale = float(saved["label_mean"]), float(saved["label_scale"])
            if not np.isfinite([self.center,self.scale]).all() or self.scale <= 0:
                raise ValueError("TabM 回归标签尺度无效")
            self.model = architecture(bins)
            self.model.load_state_dict({k[6:]:torch.from_numpy(saved[k]) for k in saved.files if k.startswith("state.")})
        self.model.eval()
        # 分位数来自这个检查点。缓存只随读出层存活，切换父版时重新建立。
        from scipy.special import ndtri
        from sklearn.preprocessing._data import BOUNDS_THRESHOLD
        self._threshold = BOUNDS_THRESHOLD
        self._knots = np.ascontiguousarray(self.transform.quantiles_[:, self.keep].T)
        self._reverse_knots = np.ascontiguousarray(-self._knots[:, ::-1])
        self._reverse_references = -self.transform.references_[::-1]
        tail = BOUNDS_THRESHOLD - np.spacing(1.)
        self._normal_limits = ndtri([tail, 1. - tail])

    def _encode(self, raw):
        if raw.shape != (1, len(self.keep)) or not np.isfinite(raw).all():
            return self.transform.transform(raw)[:, self.keep].copy()
        from scipy.special import ndtri
        values = raw[0, self.keep]
        # 同值分位点保留 sklearn 的正反向插值平均。先按输入精度舍入，再一起做正态逆变换。
        ranks = np.empty_like(values)
        for index, value in enumerate(values):
            forward = np.interp(value, self._knots[index], self.transform.references_)
            reverse = np.interp(-value, self._reverse_knots[index], self._reverse_references)
            ranks[index] = .5 * (forward - reverse)
        ranks[values + self._threshold > self._knots[:, -1]] = 1.
        ranks[values - self._threshold < self._knots[:, 0]] = 0.
        encoded = np.clip(ndtri(ranks.astype(np.float64)), *self._normal_limits)
        return encoded.astype(np.float32)[None, :]

    def _heads(self, rows):
        import torch
        raw = np.asarray(rows, dtype=np.float32)
        encoded = self._encode(raw)
        with torch.inference_mode():
            return torch.cat([self.model(batch).squeeze(-1) for batch in torch.from_numpy(encoded).split(256)])

    def _mean(self, heads):
        result = heads.mean(1).numpy() * self.scale + self.center
        if not np.isfinite(result).all():
            raise ValueError("TabM 读出出现非有限预测")
        return np.maximum(0., result.astype(np.float64))

    def predict(self, rows):
        return self._mean(self._heads(rows))

    def predict_with_spread(self, rows):
        heads = self._heads(rows)
        spread = (heads.std(1,unbiased=False)*self.scale).numpy()
        if not np.isfinite(spread).all():
            raise ValueError("TabM 多头分歧不是有限值")
        return self._mean(heads),spread
