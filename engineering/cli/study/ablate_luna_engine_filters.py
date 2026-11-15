"""冻结成熟组已提出的候选个数与训练行数，改变行身份后独立走完十轮。"""
from pathlib import Path
from contextlib import ExitStack
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from engineering.cli.study.run_luna_rul_comparison import Replay, LocalStore, read, frozen, rmse
from harness.contracts import digest

BASE = ROOT/"results/experiments/luna-comparison"
SEEDS = (17,29,41)


def run():
    output = BASE/"engine-filter-ablation"
    with ExitStack() as stack:
        stack.enter_context(LocalStore(output).exclusive())
        replay=Replay("engine",output/"runtime",stack)
        parents = {str(seed):replay.original for seed in SEEDS}
        frozen(output/"design.json",{"seeds":list(SEEDS),"design":"按已冻结成熟组各轮实际候选数及行数，"
            "用预定散列种子选取等量训练样本；每条支线独立使用自己的先前在用权重及相同提名、确认和服务规则。",
            "input":"rul/issued/engine/mature；不使用该组服务标签或事后收益选种子"})
        for n in range(1,11):
            original=read(BASE/"rul/issued/engine/mature"/f"{n:02d}.json")
            roles=replay.roles(n)
            exp,conf,service = (roles[k] for k in ("explore","adopt","serve"))
            if any(r.label is not None for r in service):
                raise ValueError("服务标签提前可见")
            for key,parent in list(parents.items()):
                path = output/"rounds"/key/f"{n:02d}.json"
                if path.exists():
                    st=read(path)
                    if st["parent"]!=parent:
                        raise ValueError("无法接续既有结果")
                else:
                    trials=[]
                    for i,t in enumerate(original["trials"],1):
                        rows=sorted(replay.fit,key=lambda r:digest([int(key),n,i,r.sample_id]))[:t["rows"]]
                        if len(rows)!=t["rows"]:
                            raise ValueError("样本数必须与参照试验一致")
                        fit=replay.train(key,n,i,{"kind":"equal_size","seed":int(key)},rows,parent,exp)
                        cand=fit["candidate"]
                        trials.append({"key":cand,"rows":len(rows),"explore":rmse(replay.errors(cand,exp)),
                                       "cpu":fit["training_cpu_seconds"]})
                    best=min(trials,key=lambda z:z["explore"],default=None)
                    incumbent=rmse(replay.errors(parent,exp))
                    check = [rmse(replay.errors(v,conf)) for v in (parent,best["key"])] if best and best["explore"]<incumbent-1e-12 else None
                    chosen=best["key"] if check and check[1]<check[0]-1e-12 else parent
                    st={"parent":parent,"selected":chosen,"round":n,"trials":trials,"parent_explore":incumbent,
                        "confirm":check,"predictions":dict(zip([r.sample_id for r in service],replay.predict(chosen,service).tolist(),strict=True))}
                    frozen(path,st)
                parents[key]=st["selected"]
        targets={r.sample_id:r for r in replay.package.validation}
        result={}
        for key in parents:
            residuals,units,rounds,sizes=[],{},[],[]
            for n in range(1,11):
                st=read(output/"rounds"/key/f"{n:02d}.json")
                sizes.extend(t["rows"] for t in st["trials"])
                vals=[]
                if set(st["predictions"])!={r.sample_id for r in replay.roles(n)["serve"]}:
                    raise ValueError("评分的服务编号不齐")
                for sid,val in st["predictions"].items():
                    err=val-targets[sid].label
                    vals.append(err)
                    units.setdefault(targets[sid].unit_id,[]).append(err)
                rounds.append(rmse(vals)); residuals.extend(vals)
            result[key]={"rmse":rmse(residuals),"worst_round":max(rounds),"worst_device":max(rmse(v) for v in units.values()),
                         "rows":len(residuals),"fits":len(sizes),"fit_rows":sum(sizes)}
        frozen(output/"scores.json",result)
        replay.endpoints(parents)
        print(result)


if __name__=="__main__":
    run()
