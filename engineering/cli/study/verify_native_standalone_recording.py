"""新真实运行验证独立CLI记录和同作业补传；不上传旧实验、不重复训练。"""
import asyncio
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow.native_research import run
from engineering.cli.study.audit_native_run import audit
from harness.integrations.native_recording import publish_completed_run
from harness.runtime.storage import atomic_json

DEST=ROOT/'results/experiments/contest-continuation/native-standalone-recording'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class FixedBroker:
    def __init__(self,calls):self.max_calls=calls;self.calls=0
    async def exchange(self,kind,payload):
        self.calls+=1
        if self.calls==1:
            action={'action':'reference','name':'official-one-epoch','hypothesis':'独立记录工程核验：真实父版续训一轮'}
        else:
            action={'action':'finish','execution':1,'candidate':'candidate-001','reason':'工程核验提名已保存候选'}
        return {'text':json.dumps(action,ensure_ascii=False)}


def cli_record(folder, label):
    result=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/workflow/run_workflow.py'),
                           'native-record',str(folder),'--online'],cwd=folder,
                          capture_output=True,text=True,encoding='utf-8',timeout=180)
    (folder/(label+'-stdout.log')).write_text(result.stdout,encoding='utf-8')
    (folder/(label+'-stderr.log')).write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr[-2000:]+result.stdout[-1500:])
    rows=[json.loads(line)['recording'] for line in result.stdout.splitlines() if line.startswith('{"recording":')]
    if len(rows)!=1:raise ValueError('独立记录CLI缺少唯一回执')
    return rows[0]


def verify(task,stages):
    folder=DEST/task
    with (DEST/(task+'-native-stdout.log')).open('w',encoding='utf-8') as stream, redirect_stdout(stream):
        report=asyncio.run(run(task,folder,driver='fixture',calls=2,stages=stages,
                              broker_factory=lambda directory,calls:FixedBroker(calls)))
    audit(folder)
    completion_before=sha256((folder/'completion.json').read_bytes()).hexdigest()
    checkpoint_before=sha256((folder/'release/checkpoint.npz').read_bytes()).hexdigest()
    try:publish_completed_run(folder)
    except PermissionError:pass
    else:raise ValueError('没有online选择仍启动记录')
    interrupted=False
    old_id=None
    if task=='gas':
        def interrupt(index):
            if index==1:raise KeyboardInterrupt('工程注入：第一轮记录已发出后中断')
        try:publish_completed_run(folder,allow_network=True,evidence_hook=interrupt)
        except KeyboardInterrupt:interrupted=True
        else:raise ValueError('未发生预定上传中断')
        first=read(folder/'recording/attempt-001.json')
        if not first.get('partial_cloud_verified') or first.get('partial_evidence_count')!=1:
            raise ValueError('第一轮部分记录未云端核验')
        old_id=read(folder/'recording/session.json')['cloud_run_id']
    result=cli_record(folder,'record')
    if not result['cloud_verified'] or not result['evidence_cloud_verified'] or result['evidence_count']!=stages:
        raise ValueError('独立入口没有完成全部云端正文核验')
    if task=='gas' and (result['cloud_run_id']!=old_id or result['uploaded_now']!=2 or result['reused_cloud_stages']!=1):
        raise ValueError('上传恢复没有只补两轮或改变云端作业')
    before=(folder/'recording/completed.json').read_bytes()
    repeated=cli_record(folder,'record-again')
    if not repeated['reused_completed_recording'] or repeated['uploaded_now']!=0 or before!=(folder/'recording/completed.json').read_bytes():
        raise ValueError('完成记录的重复调用没有复用')
    if completion_before!=sha256((folder/'completion.json').read_bytes()).hexdigest() or checkpoint_before!=sha256((folder/'release/checkpoint.npz').read_bytes()).hexdigest():
        raise ValueError('上传或恢复改变了实际训练产物')
    return {'task':task,'stages':stages,'cloud_run_id':result['cloud_run_id'],'url':result['url'],
            'cloud_verified':True,'evidence_cloud_verified':True,'interruption_recovered':interrupted,
            'uploaded_on_success':result['uploaded_now'],'reused_cloud_stages':result['reused_cloud_stages'],
            'completed_repeat_uploads':0,'model_and_training_receipts_unchanged':True,
            'actual_model_calls':0,'fixture_calls':sum(r['calls'] for r in report['stages'])}


def main():
    DEST.mkdir(parents=True,exist_ok=False)
    paths=['engineering/cli/study/verify_native_standalone_recording.py','engineering/cli/workflow/native_recording.py',
           'engineering/cli/workflow/native_research.py','engineering/cli/workflow/run_workflow.py',
           'engineering/src/harness/integrations/native_recording.py','engineering/src/harness/integrations/research_tracking.py',
           'engineering/configs/experiments/native-standalone-recording.json']
    for name in paths:
        target=DEST/'frozen-code'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    results=[verify(task,3 if task=='gas' else 1) for task in ('gas','bearing','engine')]
    summary={'passed':True,'tasks':results,'new_training_during_recording_or_resume':0,
             'actual_model_calls':0,'independent_of_research_runner':True}
    atomic_json(DEST/'verification.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
