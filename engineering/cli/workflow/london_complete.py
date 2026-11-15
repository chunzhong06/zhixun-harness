"""从原始小时资料冷启动，真实试改、选案、补训、交付并比较完整固定策略。"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_midstream import identity, propose
from harness.integrations.candidate_edit import named_scores
from harness.integrations.generated_regressor import IMPORTS,NUMPY
from harness.integrations.proposal_inbox import ProposalInbox
from harness.integrations.recipe_regressor import code_inputs
from harness.models.sparse_reference import SETTINGS
from harness.models.station_residual import error,gate
from harness.models.stream_plan import StreamPlanBank
from harness.runtime.storage import LocalStore,atomic_json


def read(path):return json.loads(Path(path).read_bytes())


def maintenance_ready(windows):
    """反馈不足只禁止需要验证的维护，不取消该站后续交付和评分。"""
    return all(int(window.sum())>=8 for window in windows)


def masks(labels, at):
    hours=np.arange(len(labels));fit=(hours<at-21*24)&np.isfinite(labels)
    windows=[(hours>=at+a*24)&(hours<at+(a+7)*24)&np.isfinite(labels) for a in (-21,-14,-7)]
    return fit,windows,np.arange(at-21*24-6,at)


def main(argv):
    ap=ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--start',required=True);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fixed-only',action='store_true')
    ap.add_argument('--verify-existing',type=Path,help='仅开发等价：比较已有发布记录，不作为运行输入')
    ap.add_argument('--reuse-components-from',type=Path,help='仅固定开发复验：训练数组与源码完全相同时复用已核验权重')
    args=ap.parse_args(argv);config=read(args.library);scene.configure(args.start)
    if args.reuse_components_from and not args.fixed_only:raise ValueError('独立现场运行从原始数据训练，不使用开发权重复验模式')
    tools=scene.prepare(args.archive,True,True);features,names=code_inputs(tools)
    reference=config['initial_plan'];registry=dict(config['plans']);reference_id=identity(reference)
    registry.setdefault(reference_id,reference)
    frozen_plans=dict(registry)
    with LocalStore(args.output).exclusive():
        if (args.output/'sites.json').exists():raise ValueError('不可覆盖冷启动运行')
        metadata={'start':args.start,'archive_sha256':sha256(args.archive.read_bytes()).hexdigest(),
            'library_sha256':sha256(args.library.read_bytes()).hexdigest(),'feature_names':names,
            'fixed_only':args.fixed_only,'default_plan_id':config['default_plan_id'],'reference_plan_id':reference_id}
        atomic_json(args.output/'input-identity.json',metadata)
        sites={};excluded={};inbox=ProposalInbox(args.output/'queue');counts={}
        for site in scene.TARGETS:
            truth=scene.readings(args.archive,site).to_numpy()
            if np.isfinite(truth[scene.CLOCK<scene.START]).mean()<.8:
                excluded[site]='开场前覆盖不足80%';continue
            try:old_y,fit0,windows0,service,returned=scene.roles(truth)
            except ValueError as exc:excluded[site]=str(exc);continue
            start=int(service[0]);at=start+28*24;dest=args.output/site;dest.mkdir()
            bank=StreamPlanBank(dest/'components',features,names,
                args.reuse_components_from/site/'components' if args.reuse_components_from else None)
            labels=np.r_[old_y[:start],returned]
            initial_history=np.arange(start-21*24-6,start)
            initial=bank.train(reference,dest/'initial',features,names,fit0,old_y,initial_history,old_y[initial_history],windows0)
            prefix=service[service<at];fallback=np.maximum(0.,features[service,4])
            first=bank.issue(reference,initial,initial_history,prefix,labels,fallback[:len(prefix)])
            np.savez_compressed(dest/'first-issued.npz',ids=prefix,prediction=first)
            fit,windows,history=masks(labels,at);known=labels[history]
            ready=maintenance_ready(windows)
            parent_errors=[error(labels[m],first[np.flatnonzero(m)-start]) if m.any() else None for m in windows]
            candidates=[];trial_failures={}
            # 冻结方案库只用截至维护开始21日前的标签试训；确认分数不进入驱动。
            for key,plan in frozen_plans.items():
                folder=dest/('refit' if key==reference_id else 'library/'+key)
                try:
                    training=bank.train(plan,folder,features,names,fit,labels,history,known,windows)
                    candidates.append({'folder':str(folder.relative_to(dest)),'plan':plan,'training':training})
                except (ValueError,KeyError,TypeError,SyntaxError,subprocess.TimeoutExpired) as exc:trial_failures[key]=str(exc)
            if not candidates:raise ValueError('没有可用的合法试训候选')
            refit=next(c['training'] for c in candidates if identity(c['plan'])==reference_id)
            ranked=(sorted(candidates,key=lambda c:np.mean(np.square(c['training']['history_rmse'][:2]))) if ready else candidates)
            observed=np.flatnonzero((np.arange(len(labels))>=start)&(np.arange(len(labels))<at-7*24)&np.isfinite(labels))
            # 返回实际已发预测。完整时钟不因缺标签压缩，确认周不进入这些错例。
            summaries=[{'hour':int(h),'raw_inputs':features[h,:9].round(5).tolist(),'answer':float(labels[h]),
                        'issued_prediction':float(first[h-start])} for h in observed]
            context={}
            if ready:
                context={'feature_names':names,'raw_input_names':names[:9],'reference_plan':reference,'reference_identity':reference_id,
                    'parent_scores':named_scores(parent_errors[:2],windows[:2],history,prediction_object='deployed_parent_ensemble_after_feedback'),
                    'refit_observation':{'scores':named_scores(refit['history_rmse'][:2],windows[:2],history,prediction_object='complete_refit_ensemble_after_feedback'),
                        'errors':refit['exploration'],'component_ablations':refit['component_ablations']},
                    'legal_service_observations':summaries,'shared_candidates':[
                        {'identity':identity(c['plan']),'scores':named_scores(c['training']['history_rmse'][:2],windows[:2],history,prediction_object='shared_candidate_after_feedback')}
                        for c in ranked],
                    'top_shared_plans':[{'identity':identity(c['plan']),'plan':c['plan']} for c in ranked[:3]],
                    'shared_candidates_policy':'共享候选来自此前开发冻结的源码，以上评分来自本次合法探索；停止新增试验后仍可采用共享候选。旧hypothesis只说明来源，不是本批观察。',
                    'training_contract':{'last_training_hour':int(np.flatnonzero(fit)[-1]),'rows':int(fit.sum()),'maintenance_hour':at,
                        'exploration_hours':[[at-21*24,at-14*24],[at-14*24,at-7*24]],'time_column':'elapsed_hour真实连续小时，缺答案不压缩',
                        'confirmation':'最后七日仅在提名后确认，不向驱动提供答案和分数','publication':'选案结束后只用严格早于维护时刻的到期参考重新拟合；新训练内成绩不重新选案'},
                    'allowed_imports':{k:sorted(v) if v else '纯计算' for k,v in IMPORTS.items()},'allowed_numpy':sorted(NUMPY),
                    'allowed_feedback':[{'gain':g,'hours':h} for g,h in SETTINGS]}
            atomic_json(dest/'trial-failures.json',trial_failures)
            if args.fixed_only or not ready:
                arms={}
                for arm in ('project','paper'):
                    best=ranked[0];decision={'trials':[],'nominated':best['folder'],'adopted':ready and gate(np.asarray(parent_errors),np.asarray(best['training']['history_rmse'])),
                        'parent_errors':parent_errors,'nominated_errors':best['training']['history_rmse'],'fixed_only':args.fixed_only,
                        'maintenance_skipped':not ready,'feedback_counts':[int(w.sum()) for w in windows]}
                    atomic_json(dest/(arm+'-decision.json'),decision);arms[arm]=(candidates,decision)
            else:
                with ThreadPoolExecutor(2) as pool:
                    pending={arm:pool.submit(propose,arm,site,dest,reference,context,parent_errors,features,names,fit,labels,history,known,windows,inbox,True,
                        train_callback=bank.train,initial_candidates=candidates,follow_best=True) for arm in ('project','paper')}
                    arms={arm:job.result() for arm,job in pending.items()}
            # 新提案暂存到全局固定清单，其他站的驱动仍只取得事前冻结清单。
            for arm,(tried,_) in arms.items():
                for candidate in tried:registry.setdefault(identity(candidate['plan']),candidate['plan'])
            # 先保存所有发布选择，再补训；确认后的训练内数据不回流给选案逻辑。
            future=service[service>=at]
            original=bank.issue(reference,initial,initial_history,service,labels,fallback)
            if not np.allclose(original[:len(prefix)],first,rtol=0,atol=1e-9):raise ValueError('旧预测发生变化')
            issued={'baseline':original};publication_fit=(np.arange(len(labels))<at)&np.isfinite(labels)
            published={};published_training={};all_candidates={identity(c['plan']):c for c in candidates}
            for tried,_ in arms.values():all_candidates.update({identity(c['plan']):c for c in tried})
            def publish(key,plan):
                if key not in published:
                    record=bank.train(plan,dest/'publication'/key,features,names,publication_fit,labels,history,known,windows)
                    published[key]=bank.issue(plan,record,history,future,labels,original[len(prefix):])
                    published_training[key]=record
                return published[key]
            for arm,(tried,decision) in arms.items():
                best=next(c for c in tried if c['folder']==decision['nominated']);key=identity(best['plan'])
                issued[arm]=original.copy()
                if decision['adopted']:
                    try:issued[arm][len(prefix):]=publish(key,best['plan'])
                    except Exception as exc:decision['publication_failed']=str(exc);decision['adopted']=False
                decision['published_plan_id']=key if decision['adopted'] else reference_id
                atomic_json(dest/(arm+'-decision.json'),decision)
            chosen=ranked[0];numeric=original.copy()
            numeric_adopted=ready and gate(np.asarray(parent_errors),np.asarray(chosen['training']['history_rmse']))
            if numeric_adopted:
                try:numeric[len(prefix):]=publish(identity(chosen['plan']),chosen['plan'])
                except Exception as exc:
                    numeric_adopted=False;atomic_json(dest/'numeric-publication-failed.json',{'error':str(exc)})
            atomic_json(dest/'shared-decision.json',{'nominated':identity(chosen['plan']),'adopted':numeric_adopted,
                'published':identity(chosen['plan']) if numeric_adopted else reference_id})
            issued['shared-only']=numeric
            sites[site]={'bank':bank,'labels':labels,'service':service,'returned':returned,'at':at,'history':history,'windows':windows,'fit':fit,
                'parent_errors':parent_errors,'maintenance_ready':ready,'initial':initial,'initial_history':initial_history,'publication_fit':publication_fit,
                'candidates':all_candidates,'published':published,'published_training':published_training,'issued':issued}
            np.savez_compressed(dest/'field-issued.npz',ids=service,returned=returned,**issued)
            counts[site]={'issued':len(service),'history_labels':int(np.isfinite(old_y).sum())}
            print(site+' 冷启动、现场选择与补训交付完成',flush=True)
        if not sites:raise ValueError('没有通过事前历史资格的站点')
        atomic_json(args.output/'sites.json',{'eligible':list(sites),'excluded':excluded})
        atomic_json(args.output/'driver-complete.json',{'done':True})
        atomic_json(args.output/'plan-bank.json',registry)
        # 全部现场停止后，才把本次新提案固定跨站；每份策略使用同一时点及资料。
        for site,state in sites.items():
            dest=args.output/site;bank=state['bank'];labels=state['labels'];service=state['service'];at=state['at'];history=state['history'];windows=state['windows']
            issued=state['issued'];original=issued['baseline'];future=service[service>=at];prefix=len(service)-len(future);failures={}
            for key,plan in registry.items():
                try:
                    if key not in state['candidates']:
                        tr=bank.train(plan,dest/'fixed-trial'/key,features,names,state['fit'],labels,history,labels[history],windows)
                        state['candidates'][key]={'plan':plan,'training':tr}
                    trial=state['candidates'][key]['training']
                    # 保留以前使用试训权重直接交接的固定对手，不强迫所有对手补训。
                    trial_future=bank.issue(plan,trial,history,future,labels,original[prefix:])
                    trial_direct=np.r_[original[:prefix],trial_future]
                    accepted=state['maintenance_ready'] and gate(np.asarray(state['parent_errors']),np.asarray(trial['history_rmse']))
                    issued['trial-direct-'+key]=trial_direct
                    issued['trial-gated-'+key]=trial_direct if accepted else original.copy()
                    if key not in state['published']:
                        record=bank.train(plan,dest/'publication'/key,features,names,state['publication_fit'],labels,history,labels[history],windows)
                        state['published'][key]=bank.issue(plan,record,history,future,labels,original[prefix:]);state['published_training'][key]=record
                    direct=np.r_[original[:prefix],state['published'][key]]
                    issued['refit-direct-'+key]=direct
                    issued['refit-gated-'+key]=direct if accepted else original.copy()
                except (KeyError,ValueError,TypeError,SyntaxError,subprocess.TimeoutExpired) as exc:
                    failures[key]=str(exc)
                    # 补训失败不能抹掉此前已成功构建的试训权重对照。
                    for mode in ('trial-direct-','trial-gated-','refit-direct-','refit-gated-'):issued.setdefault(mode+key,original.copy())
            issued['default-policy']=issued['refit-gated-'+config['default_plan_id']]
            atomic_json(dest/'fixed-failures.json',failures)
            atomic_json(dest/'timing.json',{'start_hour':int(service[0]),'maintenance_hour':at,'parent_errors':state['parent_errors'],
                'maintenance_ready':state['maintenance_ready'],'maintenance_feedback_counts':[int(w.sum()) for w in windows],
                'initial_history':state['initial_history'].tolist(),'trial_history':history.tolist(),'windows':[np.flatnonzero(w).tolist() for w in windows],
                'initial_training_last_hour':state['initial']['last_training_hour'],'trial_training_last_hour':int(np.flatnonzero(state['fit'])[-1]),
                'publication_training_last_hour':int(np.flatnonzero(state['publication_fit'])[-1]),'actual_unique_fits':bank.fit_attempts,
                'reused_component_fits':bank.reused_fits,'logical_component_slots':bank.logical_slots})
            np.savez_compressed(dest/'issued.npz',ids=service,returned=state['returned'],**issued)
        atomic_json(args.output/'predictions-complete.json',{'done':True})
        scores={};equivalence=[]
        for site,state in sites.items():
            truth=scene.readings(args.archive,site).to_numpy();service=state['service'];valid=np.isfinite(truth[service])
            if not valid.any():raise ValueError(f'预列合格站{site}没有可评分服务答案，不删除该站宣称获胜')
            counts[site]['scored']=int(valid.sum())
            for key,value in state['issued'].items():scores.setdefault(key,{})[site]=error(truth[service][valid],value[valid])
            if args.verify_existing:
                with np.load(args.verify_existing/site/'issued.npz',allow_pickle=False) as z:
                    compared=[k for k in state['issued'] if k=='baseline' or k.startswith(('refit-direct-','refit-gated-'))]
                    maximum=max(float(np.max(np.abs(state['issued'][key]-z[key]))) for key in compared)
                    if maximum>1e-8:raise ValueError(f'{site}冷启动没有复现原完整固定交付：{maximum}')
                    equivalence.append({'site':site,'strategies':len(compared),'maximum_difference':maximum})
        means={k:float(np.mean(list(v.values()))) for k,v in scores.items()}
        fixed=[k for k in means if k.startswith(('trial-direct-','trial-gated-','refit-direct-','refit-gated-')) or k=='baseline'];best=min(fixed,key=means.get)
        protected=max(scores['project'].values())<=1.01*max(scores[best].values())
        result={'scope':'已见开发的冷启动等价' if args.fixed_only else '事前冻结完整采集期的冷启动现场比较',
            'start':args.start,'fixed_only':args.fixed_only,'counts':counts,'excluded':excluded,'macro_rmse':means,'station_rmse':scores,
            'best_fixed':best,'unique_fits':sum(s['bank'].fit_attempts for s in sites.values()),'plans':len(registry),
            'reused_component_fits':sum(s['bank'].reused_fits for s in sites.values()),
            'requests':len(list((args.output/'queue/requests').glob('*/*.json'))),'development_equivalence':equivalence,
            'acceptance':{'beats_fixed_one_percent':means['project']<=.99*means[best],'beats_paper_one_percent':means['project']<=.99*means['paper'],
                'worst_station_protected':protected,'passed':not args.fixed_only and means['project']<=.99*means[best] and means['project']<=.99*means['paper'] and protected}}
        atomic_json(args.output/'scores.json',result)
        print(json.dumps({'自主完整误差':means['project'],'成熟机制完整误差':means['paper'],'无新提案':means['shared-only'],
            '既有最强完整策略':means['default-policy'],'同工具最强固定':means[best],'验收':result['acceptance'],'实际独立拟合':result['unique_fits']},ensure_ascii=False))
