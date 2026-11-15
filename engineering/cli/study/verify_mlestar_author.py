"""独立重载已执行的作者候选与发布模型，恢复封存并比较同起点固定交付。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.study.verify_london_feedback import replay
from engineering.cli.study.verify_generated_probe import predict
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs
from harness.runtime.storage import atomic_json


def read(path):return json.loads(Path(path).read_bytes())
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))


def main():
    parser=ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--library',required=True,type=Path);parser.add_argument('--fixed-run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();source=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():raise ValueError('不可覆盖独立核验')
    recovery_path=next((p for p in (source/'recovery.json',source/'refinement-recovery.json') if p.is_file()),None)
    recovery=read(recovery_path) if recovery_path else {}
    old=Path(recovery.get('original_run',str(source)));meta=read(source/'task-identity.json');config=read(args.library)
    source_summary=read(source/'scores.json') if (source/'scores.json').is_file() else {}
    arm=source_summary.get('controller','paper')
    if sha256(args.archive.read_bytes()).hexdigest()!=meta['archive_sha256'] or sha256(args.library.read_bytes()).hexdigest()!=meta['library_sha256']:raise ValueError('恢复对象输入改变')
    scene.configure(meta['start']);features,names=code_inputs(scene.prepare(args.archive,True,True));truth=scene.readings(args.archive,meta['site']).to_numpy()
    historical,fit0,windows0,service,returned=scene.roles(truth);start=int(service[0]);at=start+28*24;labels=np.r_[historical[:start],returned]
    history=np.arange(at-21*24-6,at);validation=history[history<at-7*24];confirmation=history[history>=at-7*24];future=service[service>=at]
    windows=[(history>=at+d*24)&(history<at+(d+7)*24)&np.isfinite(labels[history]) for d in (-21,-14,-7)]
    if not np.array_equal(validation,meta['validation_ids']) or not np.array_equal(confirmation,meta['confirmation_ids']):raise ValueError('时间身份不同')
    initial=read(old/'initial/training.json');initial_history=np.arange(start-21*24-6,start);clock=np.r_[initial_history,service];baseline=np.zeros(len(service));models=0
    for spec,record in zip(config['initial_plan']['components'],initial['components'],strict=True):
        folder=Path(record['folder']);weight=folder/'fit/model.joblib'
        if sha256(weight.read_bytes()).hexdigest()!=record['weights_sha256']:raise ValueError('开场权重变化')
        with np.load(folder/'input.npz',allow_pickle=False) as z:
            if not np.array_equal(z['train_ids'],np.flatnonzero(fit0)) or not np.array_equal(z['train_y'],historical[fit0]):raise ValueError('开场训练标签改变')
        load_module(folder/'fit/source.py');model=joblib.load(weight)
        _,raw,failures=predict(model,features[clock],record['prediction_limit'],np.maximum(0.,features[clock,4]))
        if failures:raise ValueError('开场独立预测出现回退')
        value=replay(raw,clock,labels[clock],(spec['feedback']['gain'],spec['feedback']['hours']))[len(initial_history):]
        baseline+=spec['weight']*value;models+=1
    for p in (old/'baseline-issued.npz',source/'baseline-issued.npz'):
        with np.load(p,allow_pickle=False) as z:
            if not np.array_equal(z['ids'],service) or not np.allclose(z['prediction'],baseline,rtol=0,atol=1e-8):raise ValueError('原发放预测未复现')
    worker=out/'model-replay';(worker/'models').mkdir(parents=True)
    for name,ids in [('validation',validation),('confirmation',confirmation),('service',future)]:pd.DataFrame(features[ids],columns=names).to_csv(worker/(name+'.csv'),index=False)
    cases=[];owners={};all_records=[]
    for origin in ([old,source] if old!=source else [source]):
        records=read(origin/'executions/index.json')
        for execution in records:
            if execution['returncode']:continue
            folder=Path(execution['directory']);training=pd.read_csv(folder/'input/train.csv');y=training.pop('target').to_numpy()
            cutoff=at if execution['phase']=='publication' else at-21*24
            allowed=(np.arange(len(labels))<cutoff)&np.isfinite(labels)
            if list(training.columns)!=names or not np.allclose(training.to_numpy(),features[allowed],rtol=0,atol=1e-10) or not np.allclose(y,labels[allowed],rtol=0,atol=1e-10):raise ValueError('执行输入违反训练截止')
            for candidate in execution['candidates']:
                p=Path(candidate['path']);weight=p.with_suffix('.pkl');checksum=sha256(weight.read_bytes()).hexdigest()
                if checksum!=candidate['weights_sha256']:raise ValueError('候选权重变化')
                key=f'model-{len(cases):03d}';shutil.copy2(weight,worker/'models'/(key+'.pkl'))
                cases.append(key);owners[key]=(execution,candidate,p);all_records.append((execution,candidate))
    atomic_json(worker/'cases.json',cases)
    script=worker/'replay.py';script.write_text('''from pathlib import Path
import json
import cloudpickle
import numpy as np
import pandas as pd
datasets={name:pd.read_csv(name+'.csv') for name in ('validation','confirmation','service')}
for key in json.loads(Path('cases.json').read_text(encoding='utf-8')):
    model=cloudpickle.loads((Path('models')/(key+'.pkl')).read_bytes())
    output={}
    for name,rows in datasets.items():
        values=[]
        for i in range(len(rows)):
            p=np.asarray(model.predict(rows.iloc[[i]]),dtype=float)
            if p.shape!=(1,) or not np.isfinite(p).all():raise ValueError('独立单行预测无效')
            values.append(max(0.,float(p[0])))
        output[name]=np.asarray(values)
    np.savez_compressed(key+'.npz',**output)
''',encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/execute_author_program.py'),'--script',str(script)],cwd=worker,env=env,capture_output=True,text=True,encoding='utf-8',timeout=200)
    (worker/'stdout.log').write_text(result.stdout,encoding='utf-8');(worker/'stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise ValueError('独立模型重载失败：'+result.stderr[-700:])
    maximum=0.;trial_fixed={};published=None;selected=None;decision=read(source/'decision.json')
    for key,(execution,candidate,p) in owners.items():
        with np.load(worker/(key+'.npz'),allow_pickle=False) as z:values={n:z[n] for n in z.files}
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as saved:
            for name in ('validation','confirmation'):
                delta=float(np.max(np.abs(values[name]-saved[name])));maximum=max(maximum,delta)
                if delta>1e-8:raise ValueError('模型实际预测未复现')
        feedback=(candidate['feedback']['gain'],candidate['feedback']['hours']);past=np.r_[values['validation'],values['confirmation']]
        prior_prediction=replay(past,history,labels[history],feedback)
        errors=[rmse(labels[history][w],prior_prediction[w]) for w in windows]
        if not np.allclose(errors[:2],candidate['exploration_rmse'],rtol=0,atol=1e-8):raise ValueError('探索成绩不同')
        issued=replay(np.r_[past,values['service']],np.r_[history,future],labels[np.r_[history,future]],feedback)[len(history):]
        full=np.r_[baseline[:len(service)-len(future)],issued]
        if execution['phase']=='publication' and str(p).startswith(str(source)):
            published=(full,candidate,values)
        elif str(p).startswith(str(source)) and execution['sequence']==decision['selected_execution']:
            selected=(errors,candidate)
        else:
            trial_fixed[key]=full
        models+=1
    if published is None or selected is None:raise ValueError('缺少真实最终及发布模型')
    parent_errors=[rmse(labels[history][w],baseline[history[w]-start]) for w in windows]
    errors,candidate=selected
    adopted=bool(np.mean(np.square(errors[:2]))<np.mean(np.square(parent_errors[:2])) and errors[2]<parent_errors[2])
    if adopted!=decision['adopted'] or not np.allclose(errors,decision['candidate_errors'],rtol=0,atol=1e-8):raise ValueError('确认决定不同')
    direct=published[0];actual=direct if adopted else baseline.copy()
    if not np.allclose(np.load(source/'serving/prediction.npy',allow_pickle=False),published[2]['service'],rtol=0,atol=1e-8):raise ValueError('实际发布服务预测不同')
    if (source/'issued.npz').is_file():
        with np.load(source/'issued.npz',allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'],service) or not np.allclose(saved[arm],actual,rtol=0,atol=1e-8):raise ValueError('完整实际发放未复现')
    strategies={'baseline':baseline,arm:actual,arm+'_direct':direct,**{arm+'-trial-'+k:v for k,v in trial_fixed.items()}}
    with np.load(args.fixed_run/meta['site']/'issued.npz',allow_pickle=False) as fixed:
        if not np.allclose(fixed['baseline'],baseline,rtol=0,atol=1e-8):raise ValueError('已有固定对手起点不同')
        for key in fixed.files:
            if key.startswith(('trial-direct-','trial-gated-','refit-direct-','refit-gated-')):strategies['frozen-'+key]=fixed[key]
    np.savez_compressed(out/'issued.npz',ids=service,returned=returned,**strategies)
    atomic_json(out/'predictions-complete.json',{'done':True})
    valid=np.isfinite(truth[service]);scores={k:rmse(truth[service][valid],v[valid]) for k,v in strategies.items()}
    fixed_keys=[k for k in scores if k.startswith(('frozen-',arm+'-trial-'))]+[arm+'_direct','baseline'];best=min(fixed_keys,key=scores.get)
    final={'scope':meta['start']+'-'+meta['site']+'完整源码研究交付的独立重载核验','controller':arm,'rmse':scores,'best_fixed':best,'adopted':adopted,
        'issued':len(service),'scored':int(valid.sum()),'model_invocations':source_summary.get('total_model_invocations',recovery.get('prior_actual_model_invocations')),'new_model_invocations':0,
        'same_budget_project_comparison':False,'independent_acquisition':False,'source_run':str(source)}
    audit={'models_reloaded':models,'maximum_prediction_difference':maximum,'training_cutoffs_verified':True,'initial_predictions_preserved':True,
        'actual_publication_verified':True,'decisions_and_scores_verified':True,'original_final_program_ast_unchanged':recovery.get('python_ast_unchanged'),
        'counterparts':'冻结库全部旧策略、作者实际试训权重直接交付及作者实际发布；尚未逐项补训所有新源码，也未新跑同预算项目组'}
    atomic_json(out/'scores.json',final);atomic_json(out/'audit.json',audit)
    print(json.dumps({'控制器':arm,'完整误差':scores[arm],'原始完整误差':scores['baseline'],'已实跑最强固定':scores[best],
        '最强固定身份':best,'实际采用':adopted,'独立权重重载':models,'最大预测差':maximum},ensure_ascii=False))


if __name__=='__main__':main()
