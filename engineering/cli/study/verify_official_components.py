"""独立重载官方部件比较的实际权重，重算全部发放、门槛和无新提案消融。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from harness.evaluation.metrics import macro_f1
from harness.runtime.storage import atomic_json
from engineering.cli.workflow import public_maintenance as public


def read(p):return json.loads(Path(p).read_bytes())


def score(task,y,p,groups):
    y=np.asarray(y);p=np.asarray(p);g=np.asarray(groups)
    if task=='gas':return {'score':macro_f1(list(zip(y,p))),'worst':min(float(np.mean(p[y==c]==c)) for c in np.unique(y))}
    error=(p-y)**2
    return {'score':float(np.sqrt(error.mean())),'worst':max(float(np.sqrt(error[g==k].mean())) for k in np.unique(g))}


def gate(task,old,new):
    if task=='gas':return new['score']-old['score']>1e-12
    return new['score']<old['score'] and (task!='engine' or new['worst']<=old['worst']*1.01)


def verify(run):
    run=Path(run).resolve()
    if (run/'audit.json').exists():raise ValueError('不覆盖独立核验')
    report=read(run/'scores.json');task=report['task'];origin=read(run/'input-identity.json');bundle=Path(origin['bundle'])
    if sha256((bundle/'task-identity.json').read_bytes()).hexdigest()!=origin['task_identity_sha256']:raise ValueError('原研究任务身份改变')
    identity=read(bundle/'task-identity.json');oracle=read(bundle/'oracle.json')
    if task=='gas':
        from harness.integrations.published_parent import bind,augment
        parent_run=Path(identity['parent_run']);timeline=read(parent_run/'run.json');stage=identity['stage']
        receipt=timeline['accepted'][stage-1];returned=set(receipt['returned'])
        if identity.get('feedback_receipt')!=receipt or identity.get('reference_parent_ordinal')!=stage:
            raise ValueError('气体任务没有绑定已完成查询与查询前父版')
        if set(oracle['explore_ids'])|set(oracle['confirm_ids'])!=returned or len(returned)>timeline['budget']:
            raise ValueError('气体评价使用了未实际回标的标签')
        samples=read(parent_run/'regular/state.json')['samples'];binding=bind(parent_run,'gas',stage)
        for ids_key,label_key,filename,role in [('explore_ids','explore_y','validation.csv','explore'),('confirm_ids','confirm_y','test.csv','adopt')]:
            rows=[samples[k] for k in oracle[ids_key]]
            if any(r['available_at']!=stage-1 or r['role']!=role for r in rows):raise ValueError('气体真实回标时点或角色错误')
            if [r['label'] for r in rows]!=oracle[label_key]:raise ValueError('气体真实回标标签不同')
            actual=augment(binding,np.asarray([r['values'] for r in rows]),ROOT/'data/official')
            if not np.allclose(actual,pd.read_csv(bundle/'task'/filename).to_numpy(),rtol=0,atol=1e-8):
                raise ValueError('气体反馈评价没有使用查询前的实际父版')
    with np.load(bundle/'outer-inputs.npz',allow_pickle=False) as z:future=z['future'];baseline=z['baseline'];service=z['service_x'][future]
    records=[]
    for arm in ('mle_star','iris'):
        for r in read(run/arm/'executions/index.json'):
            if r['returncode'] or not r['candidates']:continue
            for name,checksum in identity['task_files'].items():
                if sha256((Path(r['directory'])/'input'/name).read_bytes()).hexdigest()!=checksum:raise ValueError('程序训练与可见输入改变')
            if sha256(Path(r['source']).read_text(encoding='utf-8').encode()).hexdigest()!=r['source_sha256']:raise ValueError('实际程序源码改变')
            c=r['candidates'][-1];p=Path(c['path']).with_suffix('.pkl')
            if sha256(p.read_bytes()).hexdigest()!=c['weights_sha256']:raise ValueError('实际模型改变')
            records.append((arm,r,c,p))
    with TemporaryDirectory(dir=run,prefix='temporary-audit-') as temporary:
        work=Path(temporary);(work/'models').mkdir();columns=read(bundle/'task/task.json')['feature_columns']
        for name,filename in [('validation','validation.csv'),('confirmation','test.csv')]:shutil.copy2(bundle/'task'/filename,work/(name+'.csv'))
        pd.DataFrame(service,columns=columns).to_csv(work/'service.csv',index=False)
        keys=[]
        for arm,r,c,p in records:
            key=arm+'-'+str(r['sequence']);keys.append(key);shutil.copy2(p,work/'models'/(key+'.pkl'))
        atomic_json(work/'cases.json',keys)
        script=work/'replay.py';script.write_text('''from pathlib import Path
import json
import cloudpickle
import numpy as np
import pandas as pd
frames={n:pd.read_csv(n+'.csv') for n in ('validation','confirmation','service')}
for key in json.loads(Path('cases.json').read_text(encoding='utf-8')):
    model=cloudpickle.loads((Path('models')/(key+'.pkl')).read_bytes())
    values={}
    for name,frame in frames.items():
        output=[]
        for i in range(len(frame)):
            value=np.asarray(model.predict(frame.iloc[[i]]),dtype=float)
            if value.shape!=(1,) or not np.isfinite(value).all():raise ValueError('独立实际预测无效')
            output.append(max(0.,float(value[0])))
        values[name]=np.asarray(output)
    np.savez_compressed(key+'.npz',**values)
''',encoding='utf-8')
        env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/execute_author_program.py'),'--script',str(script)],cwd=work,env=env,capture_output=True,text=True,encoding='utf-8',timeout=300)
        if result.returncode:raise ValueError('独立权重预测失败：'+result.stderr[-1000:])
        past=[pd.read_csv(bundle/'task'/name).iloc[:,-1].to_numpy() for name in ('validation.csv','test.csv')]
        parent=[score(task,oracle[label],p,oracle[group]) for label,p,group in zip(('explore_y','confirm_y'),past,('explore_groups','confirm_groups'),strict=True)]
        restored={'baseline':baseline};exploration={};maximum=0.;candidates={}
        for (arm,r,c,p),key in zip(records,keys,strict=True):
            with np.load(work/(key+'.npz'),allow_pickle=False) as z:values={k:z[k] for k in z.files}
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as saved:
                for name in ('validation','confirmation'):
                    delta=float(np.max(np.abs(saved[name]-values[name])));maximum=max(maximum,delta)
                    if delta>1e-8:raise ValueError('保存模型不能复现探索或确认预测')
            checks=[score(task,oracle[label],values[name],oracle[group]) for label,name,group in
                    zip(('explore_y','confirm_y'),('validation','confirmation'),('explore_groups','confirm_groups'),strict=True)]
            loss=1-checks[0]['score'] if task=='gas' else checks[0]['score']
            if abs(loss-c['score'])>1e-8:raise ValueError('原探索评分不符')
            exploration[key]=loss
            full=baseline.copy();full[future]=values['service'];restored['fixed-'+key]=full
            approved=all(gate(task,a,b) for a,b in zip(parent,checks,strict=True))
            restored['gated-'+key]=full if approved else baseline.copy();candidates[key]=(approved,checks)
        for arm in ('mle_star','iris'):
            decision=read(run/arm/'decision.json');key=arm+'-'+str(decision['selected_execution'])
            if candidates[key][0]!=decision['adopted']:raise ValueError('实际确认决定不符')
            restored[arm]=restored['gated-'+key]
            shared=[key for key in keys if key.startswith(arm+'-') and int(key.rsplit('-',1)[1])<=(3 if task=='gas' else 4)]
            selected=min(shared,key=exploration.get);restored[arm+'-without-proposals']=restored['gated-'+selected]
        with np.load(run/'issued.npz',allow_pickle=False) as saved:
            if saved['ids'].tolist()!=oracle['service_ids']:raise ValueError('完整交付样本身份不符')
            for key in saved.files:
                if key=='ids':continue
                if not np.allclose(saved[key],restored[key],rtol=0,atol=1e-8):raise ValueError('实际完整预测没有复现：'+key)
        for arm in ('mle_star','iris'):
            delivered=read(run/arm/'delivery.json')['predictions']
            if not np.allclose([delivered[k] for k in oracle['service_ids']],restored[arm],rtol=0,atol=1e-8):raise ValueError('实际交接与评分不符')
        package=public.read_official_gas(Path(identity['archive'])) if task=='gas' else public.PACKAGES[task][2](Path(identity['archive']))
        answers={r.sample_id:r.label for r in package.validation};truth=[answers[k] for k in oracle['service_ids']]
        verified={k:score(task,truth,p,oracle['service_groups']) for k,p in restored.items()}
        for key,expected in report['scores'].items():
            if any(abs(verified[key][field]-expected[field])>1e-8 for field in ('score','worst')):raise ValueError('完整指标不符')
        audit={'models_reloaded':len(records),'maximum_prediction_difference':maximum,'inputs_and_training_labels_verified':True,
            'actual_query_timing_verified':True if task=='gas' else None,
            'confirmation_and_issued_verified':True,'scores_verified':True,'without_proposals':{a:verified[a+'-without-proposals'] for a in ('mle_star','iris')},
            'new_proposals_changed_delivery':{a:not np.allclose(restored[a],restored[a+'-without-proposals'],rtol=0,atol=1e-8) for a in ('mle_star','iris')}}
        parent_run=Path(identity['parent_run']);state_path=parent_run/'regular/state.json'
        if not state_path.exists():state_path=parent_run/'state.json'
        samples=read(state_path)['samples'];rounds=np.asarray([samples[k]['available_at'] for k in oracle['service_ids']])
        # 只用于解释已完成交付的退步时段，不能回流到本轮控制器改变决定。
        audit['delivery_by_round']={str(r+1):{key:score(task,np.asarray(truth)[rounds==r],restored[key][rounds==r],
            np.asarray(oracle['service_groups'])[rounds==r]) for key in ('baseline','mle_star','iris')} for r in np.unique(rounds)}
        if task=='engine':
            earlier=ROOT/'results/experiments/local-defaults/whole-maintenance/engine-field-maintenance-complete'
            established=read(earlier/'scores.json');name=established['best_fixed']
            with np.load(earlier/'issued.npz',allow_pickle=False) as z:
                names=z['names'].tolist();ids=z['ids'].tolist();matrix=z['predictions']
                if matrix.shape!=(len(ids),len(names)):raise ValueError('先前固定参照矩阵身份不符')
                p=matrix[:,names.index(name)]
                index={k:i for i,k in enumerate(ids)};p=p[[index[k] for k in oracle['service_ids']]]
            if not np.allclose(p[~future],baseline[~future],rtol=0,atol=1e-8):raise ValueError('先前固定参照前五轮起点不同')
            audit['established_strong_fixed']={'name':name,'score':score(task,truth,p,oracle['service_groups']),
                'first_five_rounds_identical':True,'source':str(earlier),
                'tool_coverage':'先前含native刻度及锚点残差Ridge；本次共同预置列表未覆盖，必须另列更强已知参照，不能把本轮池冠军当全局最强'}
        atomic_json(run/'audit.json',audit)
        print(json.dumps({task:{k:v for k,v in audit.items() if k!='delivery_by_round'}},ensure_ascii=False))


def main():
    ap=ArgumentParser();ap.add_argument('--run',required=True,type=Path,nargs='+');args=ap.parse_args()
    for run in args.run:verify(run)


if __name__=='__main__':main()
