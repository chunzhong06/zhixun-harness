"""正式Qwen驱动的完整控制器对照；双方从同一官方父版出发并只共享合法资料。"""
from pathlib import Path
from hashlib import sha256
from argparse import ArgumentParser
import asyncio
import json
import os
import shutil
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
import cloudpickle
import numpy as np
import pandas as pd
from engineering.cli.workflow.native_research import credential
from engineering.cli.study.evaluate_native_transfer import dataset
from harness.integrations.native_research import prepare, INSTRUCTION as NATIVE_INSTRUCTION
from harness.integrations.free_research_driver import run_free_research
from harness.integrations.author_comind import run_comind
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.frontier_task import DELIVERY
from harness.integrations.service_research_broker import ServiceResearchBroker
from harness.integrations.service_transport import HttpChatTransport
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.integrations.service_budget import ServiceBudget
from harness.runtime.storage import atomic_json

PROGRAM='def train(session):\n    session.build_regression_components((21,22,23),(0,2,0),epochs=60,head_epochs=12)\n'
SEED='from harness_task import train_candidate,evaluate_model\nmodel=train_candidate('+repr(PROGRAM)+')\nstem=evaluate_model(model,"source condition and residual")\n'
SEED+='import numpy as np\nimport pandas as pd\nwith np.load("artifacts/"+stem+".npz",allow_pickle=False) as z:\n    pd.DataFrame({"prediction":z["validation"]}).to_csv("validation_submission.csv",index=False)\n    pd.DataFrame({"prediction":z["confirmation"]}).to_csv("submission.csv",index=False)\n'


async def run(args):
    out=args.output.resolve()
    task,oracle=prepare('engine',out)
    with np.load(task/'train.npz') as z:x,y=z['x'],z['y']
    cols=[f'input_{i}' for i in range(x.shape[1])]
    frame=pd.DataFrame(x,columns=cols);frame['target']=y;frame.to_csv(task/'train.csv',index=False)
    with np.load(task/'evaluation-inputs.npz') as z:
        pd.DataFrame(z['explore'],columns=cols).to_csv(task/'validation.csv',index=False)
        pd.DataFrame(z['confirm'],columns=cols).to_csv(task/'test.csv',index=False)
    # task.json仅含任务元数据；不把archive复制给模型，原包内有验证标签。
    atomic_json(task/'native-contract.json',{'files':{name:sha256((task/name).read_bytes()).hexdigest() for name in ('initial.npz','baseline-model.py','train.npz')}})
    atomic_json(task/'task.json',{'task':'engine','feature_columns':cols,'interface':'native-continuation'})
    rmse=lambda a,b:float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
    def measured(prediction):return {'score':rmse(oracle['explore']['y'],prediction),'loss':rmse(oracle['explore']['y'],prediction)}
    work=out/'workspace';work.mkdir()
    executor=AuthorExecutor(out/'executions',work,{'validation_y':np.asarray(oracle['explore']['y']),'confirmation_y':np.asarray(oracle['adopt']['y'])},scorer=measured)
    config=PyTrioConfig(model='Qwen/Qwen3.8-27B',max_tokens=8192,timeout_seconds=600,raw_completion=True)
    broker=ServiceResearchBroker(out/'queue',config,HttpChatTransport(config,credential(),allow_network=True),max_calls=48)
    desc='Engine remaining useful life. RMSE lower is better. Official CPU weights MUST actually be continued. '+DELIVERY
    desc+='\nThe generic sklearn fitting instructions above are superseded by this contest contract: use from harness_task import train_candidate; model=train_candidate(program). program is a string defining train(session), using only numpy/math. evaluate_model rejects all models not returned by this tool. You may edit this training program. Source-only training, maximum 60 parent epochs and 12 residual epochs; actual parent, Adam and scalers are inherited.\n'+NATIVE_INSTRUCTION
    desc+='\nShared executable seed:\n'+SEED
    (out/'task-description.txt').write_text(desc,encoding='utf8')
    error=None;selected=None
    try:
        if args.arm=='comind':selected=await run_comind(desc,task,work,executor,broker,SEED)
        else:selected=await run_free_research(desc,task,work,executor,broker,reference_sources={'complete-source-continuation':SEED})
    except Exception as e:error={'type':type(e).__name__,'message':str(e)}
    atomic_json(out/'search-closed.json',{'calls':broker.calls,'usage':broker.usage,'error':error,'selected':selected})
    if selected is None:
        print(json.dumps({'arm':args.arm,'error':error,'calls':broker.calls},ensure_ascii=False),flush=True)
        raise RuntimeError('完整搜索没有交付选择，详见search-closed.json')
    candidate=selected['candidates'][-1]
    with np.load(Path(candidate['path']).with_suffix('.npz')) as z:
        confirm=z['confirmation'].copy();explore=z['validation'].copy()
    with np.load(out/'initial-predictions.npz') as z:initial={k:z[k].copy() for k in z.files}
    accepted=rmse(oracle['adopt']['y'],confirm)<rmse(oracle['adopt']['y'],initial['adopt']) and measured(explore)['loss']<measured(initial['explore'])['loss']
    # 确认后封存决定，再预测原先不可见的服务和迁移输入。
    atomic_json(out/'decision.json',{'adopted':accepted,'candidate':candidate})
    model=cloudpickle.loads(Path(candidate['path']).with_suffix('.pkl').read_bytes())
    with np.load(out/'service-inputs.npz') as z:sx=z['x'].copy()
    tx,ids,groups,truth=dataset('engine')
    def predict(x):return np.asarray([model.predict(pd.DataFrame(row[None,:],columns=cols))[0] for row in x])
    service=predict(sx);transfer=predict(tx)
    np.savez_compressed(out/'predictions.npz',service=service,transfer=transfer,transfer_ids=np.asarray(ids))
    release=out/'release';release.mkdir()
    (release/'checkpoint.npz').write_bytes(model.checkpoint)
    atomic_json(release/'training.json',model.report)
    result={'arm':args.arm,'driver':'Qwen/Qwen3.8-27B','author_search_complete':error is None,'adopted':accepted,
        'official_service_rows':len(sx),'official_service_rmse':rmse(oracle['serve']['y'],service if accepted else initial['serve']),
        'candidate_service_rmse':rmse(oracle['serve']['y'],service),'official_parent_rmse':rmse(oracle['serve']['y'],initial['serve']),
        'candidate_transfer_rmse':rmse(truth(),transfer),'transfer_rows':len(tx),'model_calls':broker.calls,
        'training_executions':len(executor.records),'failed_executions':sum(bool(r['returncode']) for r in executor.records),
        'execution_seconds':sum(r['execution_time'] for r in executor.records),'usage':broker.usage,
        'comparison_scope':'官方源训练，从官方父版开始的一次完整搜索及冻结NASA迁移；探索/确认/服务按设备分离，均为既有开发数据'}
    atomic_json(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--arm',choices=['project','comind'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();budget=ROOT/'results/engineering/contest-continuation/paid-budget-20261115'
    ServiceBudget(budget).initialize(30.,.00114057);os.environ['HARNESS_SERVICE_BUDGET']=str(budget)
    asyncio.run(run(args))
