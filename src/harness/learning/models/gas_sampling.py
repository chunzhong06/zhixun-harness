"""在官方每遍更新预算内重排训练样本，只读取训练标签；类别编号在此为模型内部的零起点。"""


def focused_indices(targets, label, seed, epoch):
    import numpy as np

    target = np.flatnonzero(targets == label)
    if not len(target):
        raise ValueError("指定类别缺少训练样本")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, label, 613]))
    # 每个其他类别至少留一项；抽出的空位最多用于再看一遍目标类。
    available = []
    for other in np.unique(targets):
        if other != label:
            available.extend(rng.permutation(np.flatnonzero(targets == other))[1:])
    count = min(len(target), len(available))
    order = np.arange(len(targets))
    slots = rng.permutation(available)[:count].astype(int)
    order[slots] = rng.permutation(target)[:count]
    return order


def focused_model(base, label, history):
    """随包加载器用子类恢复同一 NPZ，仍由原 fit 执行打乱、Adam 更新和保存。"""
    class FocusedModel(base):
        def fit(self, x, y, epochs):
            losses = []
            for _ in range(epochs):
                indices = focused_indices(y, label, self.meta["seed"], self.meta["epochs"])
                history.append({"epoch": self.meta["epochs"] + 1, "draws": len(indices),
                                "target_draws": int((y[indices] == label).sum())})
                losses.extend(super().fit(x[indices], y[indices], 1))
            return losses
    return FocusedModel
