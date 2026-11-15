"""容器训练/预测入口；支持显式父版和请求文件，实际平台交接另由试运行核实。"""
from argparse import ArgumentParser
import asyncio
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from harness.integrations.service_budget import ServiceBudget
from harness.runtime.storage import atomic_json


def main():
    if '--train' in sys.argv and '--profile' in sys.argv:
        from harness.integrations.platform_train import main as train
        train();return
    p=ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['gas','bearing','engine','all'],default=os.environ.get('HARNESS_TASK','all'))
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('HARNESS_DATA_ROOT','/data')))
    p.add_argument('--output',type=Path,default=Path(os.environ.get('HARNESS_OUTPUT_DIR','/output')))
    p.add_argument('--parent-checkpoint',type=Path)
    p.add_argument('--request',type=Path)
    p.add_argument('--predict',type=Path,help='独立发布目录；标准输入/输出沿官方JSONL预测约定')
    p.add_argument('--stages',type=int,default=10)
    p.add_argument('--calls',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.predict:
        import runpy
        release=args.predict.resolve();os.chdir(release);sys.path.insert(0,str(release))
        sys.argv=['baselines.predict','--checkpoint',str(release/'checkpoint.npz')]
        runpy.run_module('baselines.predict',run_name='__main__');return
    if args.request:
        request=json.loads(args.request.read_bytes())
        for key in ('task','data_root','output','parent_checkpoint','stages','calls','resume'):
            if key in request:setattr(args,key,Path(request[key]) if key in ('data_root','output','parent_checkpoint') and request[key] is not None else request[key])
    if args.task=='all' and args.parent_checkpoint:
        raise ValueError('平台父检查点必须绑定到一项明确任务')
    if not 1 <= args.stages <= 10 or not 2 <= args.calls <= 48:
        raise ValueError('轮数或单轮请求数超出范围')
    args.output.mkdir(parents=True,exist_ok=True)
    budget=Path(os.environ.get('HARNESS_SERVICE_BUDGET',str(args.output/'service-budget')))
    if not budget.exists():
        ServiceBudget(budget).initialize(10.,0.)
    os.environ['HARNESS_SERVICE_BUDGET']=str(budget.resolve())
    from engineering.cli.workflow.native_research import run
    results={}
    for task in ('gas','bearing','engine') if args.task=='all' else (args.task,):
        results[task]=asyncio.run(run(task,args.output/task,driver='service',calls=args.calls,stages=args.stages,
            data_root=args.data_root,parent_checkpoint=args.parent_checkpoint,resume=args.resume))
    atomic_json(args.output/'result.json',{'status':'completed','tasks':results})
    print(json.dumps({'status':'completed','result':str(args.output/'result.json')},ensure_ascii=False))

if __name__=='__main__':main()
