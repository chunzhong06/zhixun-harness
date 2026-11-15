"""Robin 等温度循环卷积迁移的本地 CPU 实现。只从装机留出配方决定是否采用。"""

from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


from harness.models.sgp_cycle_network import CycleNetwork, normalize


def observations(path):
    with np.load(path, allow_pickle=False) as d:
        # 论文的逐子传感器标准化，一次测量内部完成，不借用其他测点统计。
        x = normalize(d["cycles"])
        return x, d["cycle_ids"], d["mixture_ids"]


def train(model, x, y, epochs, rate, head_only=False):
    model.requires_grad_(True)
    if head_only:
        model.encoder.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=.9)
    for _ in range(epochs):
        model.train()
        if head_only:
            model.encoder.eval()
        for idx in torch.randperm(len(x)).split(64):
            opt.zero_grad()
            nn.functional.mse_loss(model(x[idx]), y[idx]).backward()
            opt.step()
        scheduler.step()
    return model.eval()


def group_outputs(model, x, codes, ids, center, scale):
    with torch.no_grad():
        values = torch.cat([model(batch) for batch in x.split(128)]).numpy() * scale + center
    return np.array([values[codes == uid].mean(dtype=np.float64) for uid in ids])


def main(work, dest):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sx, source_codes, source_ids = observations(work / "source.npz")
    with np.load(work / "source.npz") as raw:
        truth = dict(zip(source_ids, raw["concentrations"], strict=True))
    with np.load(work / "B-80-install.npz") as raw:
        order = raw["mixture_ids"]
    if len(order) != 80:
        raise ValueError("源与装机划分异常")
    # 两种预算取相同序列的前20/80个。每四种留一，从源端同步去除所有验证配方。
    excluded = order[::4]
    bases = {}
    for mode, keep in (("trial", ~np.isin(source_codes, excluded)), ("service", np.ones(len(source_codes), dtype=bool))):
        original = np.asarray([truth[i] for i in source_codes[keep]], dtype=np.float32)
        center, scale = float(original.mean()), float(original.std())
        torch.manual_seed(1031)
        model = train(CycleNetwork(), sx[keep], torch.tensor((original-center)/scale), 16, .001)
        bases[mode] = model, center, scale
        torch.save(model.state_dict(), dest / (mode + "-source.pt"))

    arms = {"全网小步": (.001 * .9 ** 8, False), "全网中步": (.001 * .9 ** 4, False),
            "全网原步": (.001, False), "只更新末层": (.001 * .9 ** 4, True)}
    results = {}
    for chip in "BC":
        service_x, service_codes, service_ids = observations(work / (chip + "-service.npz"))
        for size in (20, 80):
            key = f"{chip}-{size}"
            install_x, codes, ids = observations(work / (key + "-install.npz"))
            if not np.array_equal(ids, order[:size]):
                raise ValueError("装机样本顺序改变")
            results[key] = {}
            for mode, (base, center, scale) in bases.items():
                pick = ~np.isin(codes, excluded) if mode == "trial" else np.ones(len(codes), dtype=bool)
                if mode == "trial":
                    out_x, out_codes, out_ids = install_x[~pick], codes[~pick], ids[::4]
                else:
                    out_x, out_codes, out_ids = service_x, service_codes, service_ids
                y = np.asarray([truth[i] for i in codes[pick]], dtype=np.float32)
                predictions = {"直接用源网络": group_outputs(base, out_x, out_codes, out_ids, center, scale).tolist()}
                for name, (rate, head_only) in arms.items():
                    torch.manual_seed(1032)
                    model = train(deepcopy(base), install_x[pick], torch.tensor((y-center)/scale), 20, rate, head_only)
                    if mode == "service":
                        weight = dest / (key + "-" + name + ".pt")
                        torch.save({"weights": model.state_dict(), "center": center, "scale": scale}, weight)
                        pack = torch.load(weight, map_location="cpu", weights_only=True)
                        model = CycleNetwork().eval()
                        model.load_state_dict(pack["weights"])
                    predictions[name] = group_outputs(model, out_x, out_codes, out_ids, center, scale).tolist()
                results[key][mode] = {"ids": out_ids.tolist(), "values": predictions}
    (work / "outputs.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("dest", type=Path)
    args = cli.parse_args()
    main(args.work, args.dest)
