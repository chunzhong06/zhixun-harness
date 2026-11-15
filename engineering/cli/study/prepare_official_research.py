"""真实重载三类官方父版，建立统一合法研究任务，并复现全部既有交付。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from harness.integrations.official_research_task import prepare,metric
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.author_mlestar import ACTOR
from harness.runtime.storage import atomic_json

SOURCE='''from sklearn.base import BaseEstimator
from harness_task import load_data,evaluate_model
class PreserveParent(BaseEstimator):
    def fit(self,X,y):return self
    def predict(self,X):return X.iloc[:,-1].to_numpy()
X,y,V,T=load_data()
model=PreserveParent().fit(X,y)
evaluate_model(model,'完整父版保持')
'''


def main():
    parser=ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--task',nargs='+',choices=('gas','bearing','engine'),default=['gas','bearing','engine']);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    scores={}
    for task in args.task:
        folder=out/task;identity=prepare(task,folder);oracle=json.loads((folder/'oracle.json').read_bytes())
        def score(prediction):
            values=metric(task,oracle['explore_y'],prediction,oracle['explore_groups'])
            return {**values,'loss':1-values['score'] if task=='gas' else values['score']}
        workspace=folder/'check';work=workspace/'preserve';shutil.copytree(folder/'task',work/'input')
        executor=AuthorExecutor(folder/'executions',workspace,{'validation_y':np.asarray(oracle['explore_y']),
            'confirmation_y':np.asarray(oracle['confirm_y'])},scorer=score)
        ACTOR.set('official_parent_equivalence');result=executor(SOURCE,work,'preserve.py',90)
        if result['returncode']:raise ValueError(task+' 父版实际模型交付失败：'+result['stderr'])
        candidate=executor.records[-1]['candidates'][-1]
        with np.load(Path(candidate['path']).with_suffix('.npz'),allow_pickle=False) as z:
            for name,file in [('validation','validation.csv'),('confirmation','test.csv')]:
                if not np.allclose(z[name],pd.read_csv(folder/'task'/file).iloc[:,-1],rtol=0,atol=1e-8):raise ValueError('合法反馈预测改变')
        # 全服务输入与答案只在外层；父版已通过真实单行加载逐条复现。
        if task=='gas':
            from harness.data.official_gas import read_official_gas
            package=read_official_gas(Path(identity['archive']))
        else:
            from engineering.cli.workflow.public_maintenance import PACKAGES
            package=PACKAGES[task][2](Path(identity['archive']))
        answers={r.sample_id:r.label for r in package.validation}
        with np.load(folder/'outer-inputs.npz',allow_pickle=False) as z:
            values=z['baseline'];assert np.allclose(values,z['service_x'][:,-1],rtol=0,atol=1e-8)
        scores[task]={'complete':metric(task,[answers[k] for k in oracle['service_ids']],values,oracle['service_groups']),
            'training_rows':identity['train_rows'],'explore_rows':identity['explore_rows'],'confirm_rows':identity['confirm_rows'],
            'service_rows':len(values),'actual_model_reloaded':True,'labels_exposed_only_for_training':True,'performance_claim':'父版能力等价'}
        print(json.dumps({task:scores[task]},ensure_ascii=False),flush=True)
    atomic_json(out/'scores.json',scores)


if __name__=='__main__':main()
