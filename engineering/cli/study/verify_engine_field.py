"""独立重载发动机全部固定与现场权重，复算发布时序、共享候选增量及候选区间上界。"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import public_maintenance as public
from harness.integrations.engine_parent_view import transform
from harness.integrations.generated_regressor import load_module
from harness.integrations.published_parent import read
from harness.runtime.storage import atomic_json


def metrics(y,p,groups):
    e=(np.asarray(y)-p)**2;groups=np.asarray(groups)
    return {'error':float(np.sqrt(e.mean())),'worst':max(float(np.sqrt(e[groups==g].mean())) for g in np.unique(groups))}


def main():
    ap=ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--original',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,default=ROOT/'data/official');args=ap.parse_args()
    with TemporaryDirectory(dir=args.run,prefix='temporary-') as temporary:
        os.environ['TEMP']=temporary;os.environ['TMP']=temporary
        import tempfile
        tempfile.tempdir=temporary
        report=read(args.run/'scores.json')
        with np.load(args.run/'issued.npz',allow_pickle=False) as z:ids=z['ids'].tolist();names=z['names'].tolist();matrix=z['predictions']
        bank={p.name:{'folder':p,'available':0} for p in (args.run/'shared-programs').iterdir() if p.is_dir()}
        for name in names:
            match=re.fullmatch(r'fixed-direct-(project|paper)-r(1|6)-t([12])',name)
            if match:
                arm,stage,trial=match.groups();key=f'{arm}-r{stage}-t{trial}'
                bank[key]={'folder':args.run/arm/f'round-{stage}'/f'trial-{trial}'/'program','available':int(stage)-1}
        first=next(iter(bank.values()))['folder'];binding=read(first/'program.json')['binding']
        samples=read(ROOT/binding['run']/'regular/state.json')['samples']
        serve=[samples[k] for k in ids];ordered=list(serve);indices={}
        for stage in (0,5):
            for role in ('explore','adopt'):
                rows=sorted((r for r in samples.values() if r['role']==role and r['available_at']==stage),key=lambda r:r['sample_id'])
                indices[stage,role]=np.arange(len(ordered),len(ordered)+len(rows));ordered.extend(rows)
        raw=np.asarray([r['values'] for r in ordered]);native,parent=transform(binding,raw,args.data_root)
        x={'raw':np.column_stack((raw,parent)),'native':np.column_stack((native,parent))};predictions={'parent':parent}
        for name,record in bank.items():
            folder=record['folder'];manifest=read(folder/'program.json')
            for file,key in (('source.py','source_sha256'),('model.joblib','weights_sha256')):
                if sha256((folder/file).read_bytes()).hexdigest()!=manifest[key]:raise ValueError('发布模型身份变化')
            load_module(folder/'source.py');model=joblib.load(folder/'model.joblib')
            predictions[name]=np.asarray([max(0.,float(model.predict(row[None,:])[0])) for row in x[manifest['feature_view']]])
            if not np.isfinite(predictions[name]).all():raise ValueError('重载出现非有限输出')
        reconstructed={'parent':parent[:len(serve)]};times=np.asarray([r['available_at'] for r in serve]);changes=[]
        def known_score(name,stage,role,weight=1.):
            use=indices[stage,role];rows=[ordered[i] for i in use]
            values=weight*predictions[name][use]+(1-weight)*parent[use]
            return metrics([r['label'] for r in rows],values,[r['unit_id'] for r in rows])
        for name,record in bank.items():
            for weight in (1.,.25,.5):
                suffix=name if weight==1 else name+f'-mix-{weight}'
                p=weight*predictions[name][:len(serve)]+(1-weight)*parent[:len(serve)]
                reconstructed['fixed-direct-'+suffix]=np.where(times>=record['available'],p,parent[:len(serve)])
                active=False;out=parent[:len(serve)].copy()
                for stage in (0,5):
                    a=known_score(name if active else 'parent',stage,'adopt',weight if active else 1.)
                    b=known_score(name,stage,'adopt',weight)
                    if stage>=record['available'] and b['error']<a['error'] and b['worst']<=1.01*a['worst']:active=True
                    if active:out[(times>=stage)&(times<(5 if stage==0 else 10))]=p[(times>=stage)&(times<(5 if stage==0 else 10))]
                reconstructed['fixed-gated-'+suffix]=out
        shared_names=['parent']+[name for name in bank if not name.startswith(('project-','paper-'))]
        numeric=parent[:len(serve)].copy();active='parent';numeric_events=[]
        for stage in (0,5):
            nominated=min(shared_names,key=lambda k:known_score(k,stage,'explore')['error'])
            a,b=known_score(active,stage,'adopt'),known_score(nominated,stage,'adopt')
            accepted=b['error']<a['error'] and b['worst']<=1.01*a['worst']
            if accepted:active=nominated
            mask=(times>=stage)&(times<(5 if stage==0 else 10));numeric[mask]=predictions[active][:len(serve)][mask]
            numeric_events.append({'stage':stage,'nomination':nominated,'accepted':accepted,'published':active})
        for arm in ('project','paper'):
            out=parent[:len(serve)].copy()
            for stage in (0,5):
                event=read(args.run/arm/f'round-{stage+1}'/'decision.json');chosen=event['published']['name'] if event['published'] else 'parent'
                mask=(times>=stage)&(times<(5 if stage==0 else 10));out[mask]=predictions[chosen][:len(serve)][mask]
            reconstructed[arm]=out
            with np.load(args.original/arm/'round-1/issued.npz',allow_pickle=False) as z:
                if z['ids'].tolist()!=ids[:3500] or not np.allclose(out[:3500],z['prediction'],rtol=0,atol=1e-9):raise ValueError('恢复改变了前五轮预测')
        if set(names)!=set(reconstructed):raise ValueError('对照清单不完整')
        difference=max(float(np.max(np.abs(reconstructed[name]-matrix[:,j]))) for j,name in enumerate(names))
        if difference>1e-9:raise ValueError('独立重载与实际交付不同')
        package=public.PACKAGES['engine'][2](args.data_root/'engine'/public.PACKAGES['engine'][0]);answers={r.sample_id:r.label for r in (*package.train,*package.validation)}
        y=np.asarray([answers[k] for k in ids]);groups=[r['unit_id'] for r in serve]
        rescored={name:metrics(y,p,groups) for name,p in reconstructed.items()}
        if any(abs(rescored[k][field]-report['scores'][k][field])>1e-9 for k in names for field in ('error','worst')):raise ValueError('分数不一致')
        # 事后分别挑两个区间的模型是乐观上界，不能用作现场选择或陌生验收。
        losses=(matrix-y[:,None])**2;oracle=[];squared=0.
        for stage in (0,5):
            mask=(times>=stage)&(times<(5 if stage==0 else 10));j=int(np.argmin(losses[mask].sum(axis=0)))
            squared+=float(losses[mask,j].sum());oracle.append({'stage':stage,'strategy':names[j],'rows':int(mask.sum())})
        result={'models_reloaded':len(bank),'strategies_recomputed':len(names),'maximum_prediction_difference':difference,
                'first_five_rounds_unchanged':True,'without_new_proposals':{'events':numeric_events,'scores':metrics(y,numeric,groups),
                    'equal_project':bool(np.allclose(numeric,reconstructed['project'],rtol=0,atol=1e-9)),
                    'equal_paper':bool(np.allclose(numeric,reconstructed['paper'],rtol=0,atol=1e-9))},
                'two_interval_oracle_diagnostic':{'rmse':float(np.sqrt(squared/len(y))),'choices':oracle,
                    'scope':'终局答案仅作现有预测空间诊断，未考虑改变前段选择后重新确认的限制，不是合法实际成绩'}}
        atomic_json(args.run/'audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
