"""在一个受记录的作业内运行两组独立进程；合法输入相同，历史和响应目录分开。"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=ArgumentParser()
    parser.add_argument('--case',required=True,choices=['gas','bearing','engine','BX2','CD1','CE2'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,default=ROOT/'results/experiments/local-defaults/whole-maintenance/frontier-2026-inputs')
    parser.add_argument('--resume-transport',action='store_true',help='只重放已封存的响应和执行，再续跑传输中断点')
    args=parser.parse_args()
    official=args.case in ['gas','bearing','engine']
    folder=args.output.resolve()/args.case if official else args.output.resolve()/'london-2026'/args.case
    folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    children=[]
    for arm in ['comind','project']:
        out=folder/(arm+'-recovered' if args.resume_transport else arm)
        if out.exists():raise ValueError('配对运行不可覆盖已有组：'+str(out))
        if args.resume_transport:
            if not official:raise ValueError('此传输恢复入口仅覆盖官方任务')
            command=[sys.executable,'-B',str(ROOT/'engineering/cli/study/resume_frontier_transport.py'),
                '--previous',str(folder/arm),'--bundle',str(args.bundle.resolve()),'--output',str(out)]
        elif official:
            command=[sys.executable,'-B',str(ROOT/'engineering/cli/workflow/frontier_comparison.py'),
                '--bundle',str(args.bundle.resolve()),'--task',args.case,'--arm',arm,'--calls','48','--output',str(out)]
        else:
            command=[sys.executable,'-B',str(ROOT/'engineering/cli/workflow/mlestar_author.py'),
                '--archive',str(ROOT/'data/public/london-air-2026/stations.zip'),'--start','2026-07-01',
                '--site',args.case,'--library',str(ROOT/'engineering/configs/experiments/london-complete-library.json'),
                '--arm',arm,'--frontier-task','--max-calls','48','--output',str(out)]
        stdout=(folder/(out.name+'-process-stdout.log')).open('w',encoding='utf-8')
        stderr=(folder/(out.name+'-process-stderr.log')).open('w',encoding='utf-8')
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        children.append((arm,out,process,stdout,stderr))
    success=True
    for arm,out,process,stdout,stderr in children:
        code=process.wait();stdout.close();stderr.close();success=success and code==0
        print(json.dumps({'case':args.case,'arm':arm,'exit_code':code}),flush=True)
        result=out/'scores.json'
        if result.exists():print(json.dumps({arm:json.loads(result.read_bytes())},ensure_ascii=False),flush=True)
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
