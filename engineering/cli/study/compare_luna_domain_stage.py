"""在官方寿命模型上单独迁移阶段选样；两组各自续训，同轮使用同样多的训练行。"""
from argparse import ArgumentParser
from contextlib import ExitStack
from math import floor, sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engineering.cli.study.run_luna_rul_comparison import Replay, LocalStore, frozen, read, rmse, np
from harness.contracts import digest

BASE = ROOT / "results/experiments/luna-comparison/domain-stage"
ARMS = ("stage", "size_control")


def choose(replay, parent, exploration):
    """归一化寿命只用有答案的训练轨迹；当轮要识别的阶段只读取探索特征和父版输出。"""
    fit = replay.fit
    frac = np.array([r.label/max(r.time_end+r.label-1, 1) for r in fit])
    if replay.task == "engine":
        # EviAdapt 的作者实现将源域分为 0/1/2 三段，不完整目标只与 1/2 两段匹配。
        src = np.where(frac <= .15, 0, np.where(frac > .66, 2, 1))
        pred = replay.predict(parent, exploration)
        target = np.where(pred <= np.quantile(pred,.30), 1, 2)
    else:
        # 借用“分阶段迁移”。三等分和最近中心是本地可执行替代，不冒充论文的阶段网络。
        src = np.digitize(frac,[1/3,2/3])
        x = np.array([r.values for r in fit])
        scale = np.std(x, axis=0)
        scale[scale==0] = 1
        centers = np.array([np.mean(x[src==stage]/scale,axis=0) for stage in range(3)])
        xt = np.array([r.values for r in exploration])/scale
        target = ((xt[:,None,:]-centers[None,:,:])**2).mean(axis=2).argmin(axis=1)
    src_counts = np.bincount(src,minlength=3)
    dst_counts = np.bincount(target,minlength=3)
    proportion = dst_counts / dst_counts.sum()
    max_rows = min(src_counts[i]/p for i,p in enumerate(proportion) if p)
    counts = [min(src_counts[i],floor(max_rows*p+1e-9)) for i,p in enumerate(proportion)]
    priority = lambda row: digest([17,row.sample_id])
    subset = []
    for i,count in enumerate(counts):
        subset.extend(sorted((r for r,s in zip(fit,src,strict=True) if s==i),key=priority)[:count])
    if not subset:
        raise ValueError("训练集没有相应阶段的样本")
    control = sorted(fit,key=priority)[:len(subset)]
    return dict(zip(ARMS,(subset,control),strict=True)), {"source":src_counts.tolist(),
        "target":dst_counts.tolist(), "selected":[int(c) for c in counts]}


def run(task):
    with ExitStack() as stack:
        stack.enter_context(LocalStore(BASE/task).exclusive())
        replay = Replay(task,BASE/task/"runtime",stack)
        parent = {a:replay.original for a in ARMS}
        states = {a:[] for a in ARMS}
        for n in range(1,11):
            roles = replay.roles(n)
            exp, cfm, service = (roles[k] for k in ("explore","adopt","serve"))
            sets,counts = choose(replay,parent["stage"],exp)
            for a in ARMS:
                path = BASE/task/"rounds"/a/f"{n:02d}.json"
                if path.exists():
                    st = read(path)
                    if st["parent"] != parent[a] or st["fit_ids"] != digest(sorted(r.sample_id for r in sets[a])):
                        raise ValueError("阶段选样恢复点不一致")
                else:
                    trial = replay.train(a,n,1,{"kind":a},sets[a],parent[a],exp)
                    k = trial["candidate"]
                    old,new = [rmse(replay.errors(v,exp)) for v in (parent[a],k)]
                    confirm = [rmse(replay.errors(v,cfm)) for v in (parent[a],k)] if new<old-1e-12 else None
                    selected = k if confirm and confirm[1] < confirm[0]-1e-12 else parent[a]
                    if any(r.label is not None for r in service):
                        raise ValueError("服务答案意外提前可见")
                    st = {"round":n,"parent":parent[a],"selected":selected,"candidate":k,
                          "fit_ids":digest(sorted(r.sample_id for r in sets[a])),"rows":len(sets[a]),
                          "stage_counts":counts,"explore":[old,new],"confirm":confirm,
                          "training_cpu_seconds":trial["training_cpu_seconds"],
                          "predictions":dict(zip([r.sample_id for r in service],replay.predict(selected,service).tolist(),strict=True))}
                    frozen(path,st)
                parent[a] = st["selected"]
                states[a].append(st)
        # 揭示官方服务答案；共同测试单位及同轮预测按身份对齐。
        truth = {r.sample_id:r for r in replay.package.validation}
        result = {}
        for a,seq in states.items():
            errors, group, windows = [], {}, []
            for n,st in enumerate(seq,1):
                if set(st["predictions"]) != {r.sample_id for r in replay.roles(n)["serve"]}:
                    raise ValueError("服务测点有出入")
                obs = []
                for sid,v in st["predictions"].items():
                    err = v-truth[sid].label
                    group.setdefault(truth[sid].unit_id,[]).append(err)
                    obs.append(err)
                windows.append(rmse(obs))
                errors.extend(obs)
            result[a] = {"rmse":rmse(errors),"per_device":{k:rmse(v) for k,v in group.items()},
                         "rounds":windows,"rows":len(errors),"adoptions":sum(s["parent"]!=s["selected"] for s in seq),
                         "train_rows":sum(s["rows"] for s in seq),"training_cpu_seconds":sum(s["training_cpu_seconds"] for s in seq)}
        frozen(BASE/task/"scores.json",result)
        if task == "engine":
            replay.endpoints(parent)
        print(task,[(a,v["rmse"]) for a,v in result.items()])


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("engine","bearing"))
    run(p.parse_args().task)
