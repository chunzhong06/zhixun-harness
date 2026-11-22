"""按有界作者配置拟合当轮授权训练行；序列化纯数值状态供独立预测。"""
import io

import numpy as np

from harness.learning.inference.tabm_readout import architecture, TabMReadout


def fit_readout(x, y):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    from rtdl_num_embeddings import compute_bins
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 16 or np.ptp(y) == 0:
        raise ValueError("TabM 拟合需要足够的训练行和变化的回归标签")
    jitter = np.random.default_rng(0).normal(0,1e-5,x.shape).astype(np.float32)
    transform = QuantileTransformer(n_quantiles=min(len(x), max(min(len(x)//30,1000),10)),
                                    output_distribution="normal", subsample=10**9, random_state=42)
    transformed = transform.fit(x+jitter).transform(x)
    keep = np.ptp(transformed, axis=0) > 0
    if not keep.any():
        raise ValueError("当轮全部特征恒定")
    xx = torch.from_numpy(transformed[:,keep].copy())
    center, scale = y.mean(), y.std()
    yy = torch.tensor((y-center)/scale, dtype=torch.float32)
    bins = compute_bins(xx, n_bins=12)
    # 恢复调用方随机状态，线程数沿用比赛进程配置。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = architecture(bins)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
        model.train()
        for _ in range(6):
            for batch in torch.randperm(len(xx)).split(256):
                optimizer.zero_grad()
                (model(xx[batch]).squeeze(-1)-yy[batch,None]).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
    arrays = {"state."+key:t.detach().cpu().numpy() for key,t in model.state_dict().items()}
    arrays.update(keep=keep, quantiles=transform.quantiles_, references=transform.references_,
                  label_mean=center, label_scale=scale)
    arrays.update({"bin."+str(i):b.numpy() for i,b in enumerate(bins)})
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    fitted = TabMReadout(payload, x.shape[1])
    return payload, fitted
