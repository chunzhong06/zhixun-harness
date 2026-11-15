"""独立重载完整维护的实际权重，核对信息截止、试训、交接、固定策略与完整计分。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.study.verify_london_feedback import replay
from engineering.cli.study.verify_generated_probe import predict
from harness.contracts import digest
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs
from harness.runtime.storage import atomic_json


def read(path):return json.loads(Path(path).read_bytes())
def plan_id(plan):return sha256(json.dumps({k:plan[k] for k in ('components','filters')},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def loss(y,p):return float(np.sqrt(np.mean((y-p)**2))) if len(y) else None

def same_scores(a,b):
    return len(a)==len(b) and all((x is None and y is None) or (x is not None and y is not None and abs(x-y)<=1e-8) for x,y in zip(a,b,strict=True))
def adopt(old,new):return bool(np.mean(np.square(new[:2]))<np.mean(np.square(old[:2])) and new[2]<old[2])


def main():
    ap=ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--library',type=Path,required=True);ap.add_argument('--rebuild-from',type=Path)
    ap.add_argument('--compare-publication',type=Path);args=ap.parse_args()
    if args.rebuild_from:
        source=read(args.rebuild_from/'scores.json')
        command=[sys.executable,'-B',str(ROOT/'engineering/cli/workflow/run_workflow.py'),'london-complete',
            '--archive',str(args.archive),'--start',source['start'],'--library',str(args.library),'--output',str(args.run),
            '--fixed-only','--reuse-components-from',str(args.rebuild_from)]
        if args.compare_publication:command+=['--verify-existing',str(args.compare_publication)]
        subprocess.run(command,check=True)
    if (args.run/'audit.json').exists():raise ValueError('不覆盖已有独立核验')
    result=read(args.run/'scores.json');meta=read(args.run/'input-identity.json');config=read(args.library)
    if sha256(args.archive.read_bytes()).hexdigest()!=meta['archive_sha256'] or sha256(args.library.read_bytes()).hexdigest()!=meta['library_sha256']:
        raise ValueError('数据或冻结源码库身份改变')
    scene.configure(result['start']);tools=scene.prepare(args.archive,True,True);features,names=code_inputs(tools)
    if names!=meta['feature_names']:raise ValueError('特征顺序不同')
    plans=read(args.run/'plan-bank.json');frozen=set(config['plans'])|{plan_id(config['initial_plan'])}
    checked_components=set();maximum=0.;request_count=0;all_scores={};checks=[]
    for site in result['counts']:
        dest=args.run/site;truth=scene.readings(args.archive,site).to_numpy();old_y,_,initial_windows,service,returned=scene.roles(truth)
        labels=np.r_[old_y[:service[0]],returned];timing=read(dest/'timing.json');start,at=int(service[0]),timing['maintenance_hour']
        history=np.asarray(timing['trial_history']);initial_history=np.asarray(timing['initial_history']);future=service[service>=at];prefix=len(service)-len(future)
        windows=[]
        for hours in timing['windows']:
            mask=np.zeros(len(labels),dtype=bool);mask[hours]=True;windows.append(mask)
        ready=all(int(w.sum())>=8 for w in windows)
        if timing.get('maintenance_ready',True)!=ready:raise ValueError('维护反馈资格不同')
        cache={};trainings={}
        def base(component,hours,fallback,strict=False):
            hours=np.asarray(hours,dtype=int);key=(component['id'],sha256(hours.tobytes()).hexdigest())
            if key in cache:return cache[key].copy()
            folder=Path(component['folder']).resolve()
            if not folder.is_relative_to(args.run.resolve()):raise ValueError('模型记录越出本次运行目录')
            weights=folder/'fit/model.joblib'
            if sha256(weights.read_bytes()).hexdigest()!=component['weights_sha256']:raise ValueError('发布权重身份不同')
            load_module(folder/'fit/source.py');model=joblib.load(weights)
            _,values,bad=predict(model,features[hours],component['prediction_limit'],fallback)
            if strict and bad:raise ValueError('合法历史重载失败')
            if not bad:cache[key]=values.copy()
            checked_components.add(str(folder));return values
        def issue(plan,record,past,hours,fallback):
            clock=np.r_[past,hours];out=np.zeros(len(hours))
            for spec,component in zip(plan['components'],record['components'],strict=True):
                raw=np.r_[base(component,past,np.zeros(len(past)),True),base(component,hours,fallback)]
                out+=spec['weight']*replay(raw,clock,labels[clock],(spec['feedback']['gain'],spec['feedback']['hours']))[len(past):]
            return out
        for path in dest.rglob('training.json'):
            record=read(path);plan=read(path.with_name('plan.json'));key=plan_id(plan)
            if key!=record['identity']:raise ValueError('训练计划身份不符')
            kind=path.relative_to(dest).parts[0];cutoff=start-21*24 if kind=='initial' else (at if kind=='publication' else at-21*24)
            selected=(np.arange(len(labels))<cutoff)&np.isfinite(labels)
            for f in plan['filters']:
                col=features[:,names.index(f['feature'])];selected&=(col<=f['threshold']) if f['operator']=='<=' else (col>f['threshold'])
            expected_ids=np.flatnonzero(selected)
            for spec,component in zip(plan['components'],record['components'],strict=True):
                folder=Path(component['folder'])
                if (folder/'fit/source.py').read_text(encoding='utf-8')!=spec['python_source']:raise ValueError('权重源码不是该方案')
                with np.load(folder/'input.npz',allow_pickle=False) as z:
                    if (not np.array_equal(z['train_ids'],expected_ids) or not np.array_equal(z['train_x'],features[expected_ids])
                        or not np.array_equal(z['train_y'],labels[expected_ids]) or np.any(z['train_ids']>=cutoff)):
                        raise ValueError('训练使用了不同或未到期标签')
            h=initial_history if kind=='initial' else history;ws=initial_windows if kind=='initial' else windows
            out=np.zeros(len(h))
            for spec,component in zip(plan['components'],record['components'],strict=True):
                raw=base(component,h,np.zeros(len(h)),True)
                out+=spec['weight']*replay(raw,h,labels[h],(spec['feedback']['gain'],spec['feedback']['hours']))
            measured=[loss(labels[h][w[h]],out[w[h]]) for w in ws]
            if not same_scores(measured,record['history_rmse']):raise ValueError('试训评分不同')
            trainings[str(path.parent.relative_to(dest))]=(plan,record)
        reference,initial=trainings['initial'];original=issue(reference,initial,initial_history,service,np.maximum(0.,features[service,4]))
        with np.load(dest/'first-issued.npz',allow_pickle=False) as z:
            if not np.array_equal(z['ids'],service[:prefix]) or not np.allclose(original[:prefix],z['prediction'],rtol=0,atol=1e-8):raise ValueError('前半段被回写')
        parent_errors=[loss(labels[w],original[np.flatnonzero(w)-start]) for w in windows]
        if not same_scores(parent_errors,timing['parent_errors']):raise ValueError('已发父版的验证分数不同')
        reconstructed={'baseline':original};trial_records={};failures=read(dest/'fixed-failures.json')
        for location,(plan,tr) in trainings.items():
            if location.split('/')[0] not in ('initial','publication') and location.split('\\')[0] not in ('initial','publication'):
                trial_records[plan_id(plan)]=(plan,tr)
        for key,plan in plans.items():
            if key in failures:
                # 已失败的分支另核对存在的成功产物，不能默认丢掉试训权重对照。
                if key not in trial_records:
                    for mode in ('trial-direct-','trial-gated-','refit-direct-','refit-gated-'):reconstructed[mode+key]=original.copy()
                    continue
            _,trial=trial_records[key];accepted=ready and adopt(timing['parent_errors'],trial['history_rmse'])
            pred=issue(plan,trial,history,future,original[prefix:]);direct=np.r_[original[:prefix],pred]
            reconstructed['trial-direct-'+key]=direct;reconstructed['trial-gated-'+key]=direct if accepted else original.copy()
            pub=dest/'publication'/key/'training.json'
            if pub.is_file():
                record=read(pub);value=issue(plan,record,history,future,original[prefix:]);direct=np.r_[original[:prefix],value]
                reconstructed['refit-direct-'+key]=direct;reconstructed['refit-gated-'+key]=direct if accepted else original.copy()
            else:
                reconstructed['refit-direct-'+key]=original.copy();reconstructed['refit-gated-'+key]=original.copy()
        for arm in ('project','paper'):
            decision=read(dest/(arm+'-decision.json'));plan,trial=trainings[decision['nominated']]
            if not same_scores(decision['parent_errors'],parent_errors) or not same_scores(decision['nominated_errors'],trial['history_rmse']):raise ValueError('采用比较对象不同')
            expected=ready and adopt(decision['parent_errors'],decision['nominated_errors']) and 'publication_failed' not in decision
            if expected!=decision['adopted']:raise ValueError('发布采用决定不同')
            key=plan_id(plan);reconstructed[arm]=reconstructed['refit-direct-'+key] if expected else original.copy()
        eligible=[(key,tr) for key,(_,tr) in trial_records.items() if key in frozen]
        chosen=read(dest/'shared-decision.json');key=chosen['nominated'];tr=trial_records[key][1]
        best_loss=min(np.mean(np.square(v[1]['history_rmse'][:2])) for v in eligible) if ready else None
        if key not in frozen or (ready and abs(np.mean(np.square(tr['history_rmse'][:2]))-best_loss)>1e-8):raise ValueError('共享选择不是合法探索最优')
        if chosen['adopted'] and not ready:raise ValueError('反馈不足却采用数值选案')
        reconstructed['shared-only']=reconstructed['refit-direct-'+key] if chosen['adopted'] else original.copy()
        reconstructed['default-policy']=reconstructed['refit-gated-'+config['default_plan_id']]
        with np.load(dest/'issued.npz',allow_pickle=False) as z:
            names_checked=[k for k in z.files if k not in ('ids','returned')]
            for key in names_checked:
                delta=float(np.max(np.abs(z[key]-reconstructed[key])));maximum=max(maximum,delta)
                if delta>1e-8:raise ValueError('完整逐条发放不同：'+site+'/'+key)
                valid=np.isfinite(truth[service]);all_scores.setdefault(key,{})[site]=loss(truth[service][valid],reconstructed[key][valid])
        for arm in ('project','paper'):
            decision=read(dest/(arm+'-decision.json'))
            for i,trial in enumerate(decision['trials'],1):
                matches=list((args.run/'queue/requests'/('mid-'+arm)).glob(f'{site}-trial-{i}-*.json'))
                if len(matches)!=1:raise ValueError('缺少实际驱动请求')
                path=matches[0];q=read(path);c=q['context'];body={'model':'gpt-6-luna','messages':[{'role':'system','content':q['instruction']},
                    {'role':'user','content':json.dumps(c,ensure_ascii=False,allow_nan=False)}]}
                if digest(body)!=q['request_id'] or c['reference_identity']!=plan_id(c['reference_plan']):raise ValueError('驱动或工作版身份不符')
                if any(r['hour']>=at-7*24 for r in c['legal_service_observations']):raise ValueError('向驱动泄露确认或未来答案')
                if any(r['identity'] not in frozen for r in c['shared_candidates']):raise ValueError('提前共享了别站的新提案')
                if not trial.get('request_failed'):
                    a=read(args.run/'queue/responses'/('mid-'+arm)/path.name)
                    if a['request_id']!=q['request_id'] or a['plan']!=trial['proposal']:raise ValueError('原提案改变')
                request_count+=1
        checks.append({'site':site,'strategies':len(names_checked),'training_plans':len(trainings),'cutoffs_verified':True,'first_half_preserved':True})
    for key,stations in all_scores.items():
        if abs(float(np.mean(list(stations.values())))-result['macro_rmse'][key])>1e-8:raise ValueError('站点等权总分不同')
    audit={'models_reloaded':len(checked_components),'requests_verified':request_count,'maximum_prediction_difference':maximum,
        'all_training_cutoffs_verified':True,'confirmation_hidden_from_driver':True,'all_fixed_and_scores_verified':True,'sites':checks}
    atomic_json(args.run/'audit.json',audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
