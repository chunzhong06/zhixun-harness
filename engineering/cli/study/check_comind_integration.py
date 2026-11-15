"""合成小任务检查完整作者流程及内核边界；固定响应仅用于工程验证。"""
import asyncio
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from harness.integrations.author_comind import run_comind
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.frontier_task import DELIVERY
from harness.runtime.storage import atomic_json

CODE='''from harness_task import load_data,evaluate_model
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
X,y,V,T=load_data()
model=Ridge(alpha=0.01).fit(X,y)
stem=evaluate_model(model,'engineering only')
with np.load('artifacts/'+stem+'.npz',allow_pickle=False) as z:
    pd.DataFrame({'prediction':z['validation']}).to_csv('validation_submission.csv',index=False)
    pd.DataFrame({'prediction':z['confirmation']}).to_csv('submission.csv',index=False)
'''

class FixtureBroker:
    """固定可核对响应，不访问模型服务，不产生论文性能结论。"""
    max_calls=100;timeout=60;search_calls=0;calls=0;round=0
    async def exchange(self,kind,payload):
        self.calls+=1
        messages=payload['request']['messages'];text=messages[0]['content'];last=messages[-1]['content']
        if 'determine whether' in text:answer='<is_lower_better>True</is_lower_better>'
        elif 'holistic overview' in text:answer='<summary>Ridge baseline</summary><suggestions>Check fit</suggestions><code>'+CODE+'</code><submission>None</submission>'
        elif 'alternative and highly novel' in text:answer=''.join('<solution><description>Plan '+str(i)+'</description><idea>Fit Ridge</idea></solution>' for i in range(4))
        elif 'rephrase and decouple' in text:answer='<idea>Fit and compare Ridge variants</idea>'
        elif 'generate a list of pipelines' in text:
            self.round+=1
            answer=''.join('<pipeline><title>Round '+str(self.round)+' Model '+str(i)+'</title><description>Fit Ridge</description><datasets></datasets><codebase>None</codebase><code>'+CODE+'</code></pipeline>' for i in range(2))
        elif 'converting a Jupyter notebook' in text:answer='<code>'+CODE+'</code>'
        elif 'Please summarize the full code' in last:answer='<description>Fitted actual Ridge</description><code>'+CODE+'</code><suggestions>Only an engineering fixture</suggestions>'
        else:answer='<goal>Fit actual synthetic training data</goal><code>'+CODE+'</code><validation_submission>validation_submission.csv</validation_submission><submission>submission.csv</submission>'
        return {'text':answer}

async def main(out):
    if out.exists():raise ValueError('不覆盖工程检查')
    task=out/'task';task.mkdir(parents=True)
    x=np.arange(20.)[:,None];y=x[:,0]*2+3
    pd.DataFrame({'x':x[:,0],'target':y}).to_csv(task/'train.csv',index=False)
    for name,values in [('validation',np.arange(20.,24.)),('test',np.arange(24.,26.))]:
        pd.DataFrame({'x':values}).to_csv(task/(name+'.csv'),index=False)
    def score(v):
        value=float(np.sqrt(np.mean((v-(np.arange(20.,24.)*2+3))**2)))
        return {'score':value,'loss':value}
    work=out/'work';work.mkdir()
    executor=AuthorExecutor(out/'executions',work,{'validation_y':np.arange(4.),'confirmation_y':np.arange(2.)},scorer=score)
    broker=FixtureBroker()
    selected=await run_comind('Synthetic regression, RMSE lower better. '+DELIVERY,task,work,executor,broker,CODE,iterations=2,cells=1)
    assert len(executor.records)==4
    assert all(r['returncode']==0 and r['candidates'] for r in executor.records)
    assert len(json.loads((work/'author-selection.json').read_bytes())['reports'])==5
    report={'synthetic_fixture_only':True,'author_iterations':2,'coding_branches':4,'actual_model_delivered':bool(selected['candidates']),
            'loss':selected['candidates'][-1]['score'],'fixture_calls':broker.calls,'real_model_calls':0}
    atomic_json(out/'checks.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':asyncio.run(main(Path(sys.argv[1]).resolve()))
