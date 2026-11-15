"""四组预测全部封存后，核对分数并追踪首个改变预测的决定及同量抽样效果。"""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack
from math import fsum, isclose, sqrt

from run_unseen_agents import PRIMARY as ARMS, ExternalReplay, LocalStore, RESULT, digest, frozen, np, read, rmse


def per_unit(rows,preds):
    squared=defaultdict(list)
    for k,p in preds.items():
        r=rows[k]
        squared[r.unit_id].append((p-r.label)**2)
    return {k:(fsum(v),len(v)) for k,v in sorted(squared.items())}


def total(stats):
    return sqrt(fsum(s for s,_ in stats.values())/sum(n for _,n in stats.values()))


def balanced_bootstrap(a,b):
    """每次整台重抽，保留样本在一条寿命轨迹上的依赖。区间只针对这次固定决策路径。"""
    if len(a)<10:return None
    if a.keys()!=b.keys():raise ValueError("配对设备不齐")
    keys=sorted(a)
    v=np.asarray([(a[k][0],b[k][0],a[k][1]) for k in keys])
    rng=np.random.default_rng(17)
    index=rng.integers(0,len(keys),size=(4000,len(keys)))
    s=v[index].sum(axis=1)
    # 正值表示第一组误差较低。
    diff=np.sqrt(s[:,1]/s[:,2])-np.sqrt(s[:,0]/s[:,2])
    return np.quantile(diff,[.025,.975]).tolist()


def same_size(run,leader,seed):
    parent=run.original
    issued={}; fits=[]; adopted=[]
    folder=RESULT/"winner-check"/run.task/leader/f"sample-{seed}"
    for n in range(1,11):
        reference=read(run.state_path(leader,n))
        role=run.roles(n)
        exp,confirm,service=(role[k] for k in ("explore","adopt","serve"))
        path=folder/f"{n:02d}.json"
        if path.exists():
            record=read(path)
            if record["parent"]!=parent or record["leader"]!=leader:
                raise ValueError("消融恢复与胜者或父版不符")
        else:
            attempts=[]
            for j,old in enumerate(reference["trials"],1):
                rows=sorted(run.fit,key=lambda r:digest([seed,n,j,r.sample_id]))[:old["rows"]]
                if len(rows)!=old["rows"]:raise ValueError("训练资源未匹配")
                job=run.train(f"posthoc-{leader}-random-{seed}",n,j,{"seed":seed},rows,parent,exp)
                attempts.append({"model":job["candidate"],"rows":len(rows),
                                 "score":rmse(run.errors(job["candidate"],exp))})
            best=min(attempts,key=lambda x:x["score"],default=None)
            selected=parent
            if best and best["score"]<rmse(run.errors(parent,exp))-1e-12:
                if rmse(run.errors(best["model"],confirm))<rmse(run.errors(parent,confirm))-1e-12:
                    selected=best["model"]
            record={"leader":leader,"parent":parent,"selected":selected,"fits":attempts,
                    "issued":dict(zip([r.sample_id for r in service],run.predict(selected,service).tolist(),strict=True))}
            frozen(path,record)
        if [x["rows"] for x in reference["trials"]]!=[x["rows"] for x in record["fits"]]:
            raise ValueError("消融的逐候选行数不匹配")
        if parent!=record["selected"]:adopted.append(n)
        parent=record["selected"]
        fits.extend(x["rows"] for x in record["fits"])
        issued.update(record["issued"])
    if issued.keys()!=run.truth.keys():raise ValueError("消融预测不齐")
    return {"rmse":total(per_unit(run.truth,issued)),"adoptions":adopted,"fits":len(fits),"train_rows":sum(fits)}


def audit(task,common=False):
    with ExitStack() as stack:
        stack.enter_context(LocalStore(RESULT/"issued"/task).exclusive())
        run=ExternalReplay(task,stack)
        arms=tuple("project" if a=="project-frozen" else a for a in ARMS) if common else ARMS
        summary=run.score_external(arms,"common-scores" if common else "primary-scores")
        predictions={}; groups={}; sequences={}
        for a in arms:
            states=[read(run.state_path(a,n)) for n in range(1,11)]
            preds={k:v for s in states for k,v in s["predictions"].items()}
            groups[a]=per_unit(run.truth,preds)
            if not isclose(total(groups[a]),summary[a]["rmse"],rel_tol=1e-12) or preds.keys()!=run.truth.keys():
                raise ValueError("逐条预测与声明分数不同")
            predictions[a]=preds;sequences[a]=states
        order=sorted((a for a in arms if a!="static"),key=lambda a:(summary[a]["rmse"],summary[a]["fits"],a))
        top=order[0]
        tied=[a for a in order if abs(summary[a]["rmse"]-summary[top]["rmse"])<1e-10]
        rivals={};changes=[]
        for a in arms:
            if a==top:continue
            seq=sequences[a]
            delta=[groups[a][u][0]-groups[top][u][0] for u in groups[top]]
            round_deltas=[]
            for n,(x,y) in enumerate(zip(sequences[top],seq,strict=True),1):
                keys=x["predictions"].keys()
                gain=fsum((y["predictions"][k]-run.truth[k].label)**2-
                          (x["predictions"][k]-run.truth[k].label)**2 for k in keys)
                round_deltas.append({"round":n,"rows":len(keys),"rival_minus_winner_squared_error":gain})
            rivals[a]={"rmse_advantage_for_leader":summary[a]["rmse"]-summary[top]["rmse"],
                "paired_devices_better_tied_worse":[sum(i>1e-8 for i in delta),sum(abs(i)<=1e-8 for i in delta),sum(i< -1e-8 for i in delta)],
                "changed_predictions":sum(abs(v-predictions[a][k])>1e-9 for k,v in predictions[top].items()),
                "first_difference":next((n for n,(x,y) in enumerate(zip(sequences[top],seq),1)
                        if any(abs(p-y["predictions"][k])>1e-9 for k,p in x["predictions"].items())),None),
                "device_bootstrap_interval":balanced_bootstrap(groups[top],groups[a]),
                "round_differences":round_deltas}
        for n,s in enumerate(sequences[top],1):
            if not s["adopted"]:continue
            service=run.roles(n)["serve"]
            old=dict(zip([r.sample_id for r in service],run.predict(s["parent"],service).tolist(),strict=True))
            winning=next(t for t in s["trials"] if t["candidate"]==s["selected"])
            changes.append({"round":n,"action":winning["action"],"training_rows":winning["rows"],
                            "confirmation":s["confirmation"],"same_parent_service_rmse":total(per_unit(run.truth,old)),
                            "candidate_service_rmse":summary[top]["rounds"][n-1]})
        sizes=[t["rows"] for s in sequences[top] for t in s["trials"]]
        controls=({str(seed):same_size(run,top,seed) for seed in (17,29,41)}
                  if any(n<len(run.fit) for n in sizes) else {})
        result={"numerical_order":order,"ties":tied,"representative":top,"paired":rivals,
                "winner_changes":changes,"same_size_controls":controls}
        frozen(RESULT/"winner-check"/task/("common.json" if common else "summary.json"),result)
        print(task,order,{a:summary[a]["rmse"] for a in arms},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("bearing","engine"));p.add_argument("--common",action="store_true")
    args=p.parse_args();audit(args.task,args.common)
