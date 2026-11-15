"""在已见批次比较历史排名的交付口径与训练时距；选择仅用开场前资料。"""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from harness.models.sparse_reference import SETTINGS,correct,name,window_scores
from harness.models.station_residual import Rule,error,gate
from harness.runtime.storage import LocalStore,atomic_json


def history_rankings(tools,rules,y,fit,parent_setting,gap):
    """三个前向窗口，分别比较原模型误差和同反馈机制下的实际预测误差。"""
    end=np.flatnonzero(fit)[-1]+1
    clock=np.arange(len(y));raw_scores=[];stream_scores=[];parents=[];stream_parents=[];folds=[]
    for fold in range(3):
        first=end-(3-fold)*21*24
        train=fit&(clock<first-gap*24)
        test=fit&(clock>=first)&(clock<first+21*24)
        if train.sum()<80 or test.sum()<16:raise ValueError("前向历史样本不足")
        bank=scene.base_bank(tools,rules,train,y)
        parent=tools.train(tools.static,train,y)
        matrix=np.column_stack([bank[r] for r in rules])
        raw_scores.append(np.sqrt(np.mean((matrix[test]-y[test,None])**2,axis=0)))
        parents.append(error(y[test],tools.issue(parent,test)))
        # 与实际发放相同，使用检验开始前七日的合法参考恢复反馈状态。
        ix=clock[(clock>=first-7*24)&(clock<first+21*24)]
        measured=test[ix]
        values=[]
        for setting in SETTINGS:
            issued=correct(matrix[ix],y[ix],ix,setting)
            values.append(np.sqrt(np.mean((issued[measured]-y[ix[measured],None])**2,axis=0)))
        stream_scores.append(np.asarray(values))
        old=correct(tools.issue(parent,ix)[:,None],y[ix],ix,parent_setting)[:,0]
        stream_parents.append(error(y[ix[measured]],old[measured]))
        folds.append({"train_last":int(clock[train][-1]),"test_first":int(clock[test][0]),
                      "train_count":int(train.sum()),"test_count":int(test.sum()),"gap_days":gap})
    raw=np.asarray(raw_scores);stream=np.asarray(stream_scores)
    raw_risk=(raw/np.maximum(parents,1e-8)[:,None]).mean(axis=0)
    joint_risk=(stream/np.maximum(stream_parents,1e-8)[:,None,None]).mean(axis=0)
    # 对每个配方的反馈也只凭早期回放选择，未使用探索、确认或未来答案。
    best_setting=joint_risk.argmin(axis=0)
    joint=joint_risk[best_setting,np.arange(len(rules))]
    return {"raw":{"risk":raw_risk,"order":np.argsort(raw_risk,kind="stable")},
            "feedback":{"risk":joint,"order":np.argsort(joint,kind="stable"),"settings":best_setting},
            "folds":folds,"raw_rmse":raw,"feedback_rmse":stream}


def numeric_candidate(tools,rules,order,base,parent,parent_setting,y,windows,service,returned):
    top=int(order[0])
    signature=lambda r:(r.background,r.correction,r.features,r.learner)
    second=next(int(i) for i in order if signature(rules[i])!=signature(rules[top]))
    losses={}
    settings={}
    for i in (top,second):
        for s in SETTINGS:
            losses[(i,s)]=window_scores(base[rules[i]][:,None],windows,y,s)[:,0]
        settings[i]=min(SETTINGS,key=lambda s:np.mean(losses[(i,s)][:2]**2))
    best=min((top,second),key=lambda i:np.mean(losses[(i,settings[i])][:2]**2))
    parent_base=tools.issue(parent,np.arange(len(y)))[:,None]
    old=window_scores(parent_base,windows,y,parent_setting)[:,0]
    initial=(1 if best==top else 2) if gate(old,losses[(best,settings[best])]) else 0
    models=[(parent,parent_setting)]+[(tools.train(rules[i],np.isfinite(y)&(np.arange(len(y))<np.flatnonzero(windows[0])[0]),y),settings[i]) for i in (top,second)]
    models += [(parent,settings[i]) for i in (top,second)]
    packet={"models":models,"initial":initial,"memory_days":28}
    dynamic,_=scene.emit_deployment(tools,packet,y,windows,service,returned)
    model,setting=models[initial]
    frozen=scene.fixed_stream(tools.issue(model,np.arange(len(y)))[:,None],y,windows,service,returned,setting,True)[:,0]
    return frozen,dynamic,{"ranked_ids":[top,second],"selected_id":best,"adopted":bool(initial),
                           "settings":{str(i):list(settings[i]) for i in (top,second)}}


def main():
    ap=ArgumentParser();ap.add_argument("--original",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    source=json.loads((args.original/"scores.json").read_bytes());scene.configure(source["start"])
    tools=scene.prepare(args.archive,True,True);tools.static=Rule(**source["initial_spec"]["recipe"])
    f=source["initial_spec"]["feedback"];parent_setting=(f["gain"],f["hours"])
    rules=[Rule(**r) for r in source["recipes"]]
    winner=int(source["best_fixed"].split("-")[1]);scores={};details={}
    with LocalStore(args.output).exclusive():
        if (args.output/"scores.json").exists():raise ValueError("不可覆盖已见诊断")
        for station in source["counts"]:
            truth=scene.readings(args.archive,station).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
            base=scene.base_bank(tools,rules,fit,y);parent=tools.train(tools.static,fit,y)
            old=json.loads((args.original/station/"history.json").read_bytes())
            details[station]={};emitted={};valid=np.isfinite(truth[service])
            for gap in (0,21):
                hist=history_rankings(tools,rules,y,fit,parent_setting,gap)
                if gap==0 and not np.allclose(hist["raw_rmse"].T,old["rmse"],rtol=0,atol=1e-10):
                    raise ValueError("原历史试训不能精确复现")
                for view in ("raw","feedback"):
                    key=f"gap{gap}-{view}"
                    frozen,dynamic,record=numeric_candidate(tools,rules,hist[view]["order"],base,parent,parent_setting,y,windows,service,returned)
                    if key=="gap0-raw":
                        for variant,pred in (("numeric",frozen),("numeric-adaptive",dynamic)):
                            if abs(error(truth[service][valid],pred[valid])-source["station_rmse"][variant][station])>1e-10:
                                raise ValueError("原数值路径不能精确复现")
                    for label,pred in (("fixed",frozen),("adaptive",dynamic)):
                        emitted[key+"-"+label]=pred
                        scores.setdefault(key+"-"+label,{})[station]=error(truth[service][valid],pred[valid])
                    record.update(fixed_winner_history_rank=int(np.flatnonzero(hist[view]["order"]==winner)[0])+1,
                                  leading_recipe=asdict(rules[int(hist[view]["order"][0])]),
                                  ordered_ids=hist[view]["order"].tolist(),folds=hist["folds"])
                    details[station][key]=record
            dest=args.output/station;dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/"issued.npz",**emitted)
            print(station+" 历史排序与完整预测完成",flush=True)
        means={k:float(np.mean(list(v.values()))) for k,v in scores.items()}
        atomic_json(args.output/"scores.json",{"scope":"已见2025开发诊断，四种排名只使用开场前答案", "station_rmse":scores,
                    "macro_rmse":means,"sites":details,"original_project":source["macro_rmse"]["project"],
                    "original_fixed":source["macro_rmse"][source["best_fixed"]]})
        print(json.dumps({"完整误差":means},ensure_ascii=False))


if __name__=="__main__":main()
