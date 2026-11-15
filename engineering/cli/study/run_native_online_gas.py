"""经真实CLI跑完Luna三轮和独立云端记录；研发包装器只接收汇总。"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.study.audit_native_run import audit
from harness.runtime.storage import atomic_json

DEST=ROOT/'results/experiments/contest-continuation/native-online-gas'


def main():
    DEST.mkdir(parents=True,exist_ok=False)
    run=DEST/'run'
    command=[sys.executable,'-B',str(ROOT/'engineering/cli/workflow/run_workflow.py'),
             'native-research','gas','--driver','queue','--calls','2','--stages','3',
             '--recording','online','--output',str(run)]
    atomic_json(DEST/'command.json',{'command':command,'purpose':'真实Luna与独立在线记录一条命令闭环'})
    with (DEST/'stdout.log').open('w',encoding='utf-8') as stdout,(DEST/'stderr.log').open('w',encoding='utf-8') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=540)
    if result.returncode:
        raise RuntimeError('独立CLI未成功完成，查看原始stderr与运行回执，返回码'+str(result.returncode))
    verified=audit(run)
    report=json.loads((run/'run.json').read_bytes())
    recording=json.loads((run/'recording/completed.json').read_bytes())
    if not recording['cloud_verified'] or recording['evidence_count']!=3:
        raise ValueError('实际独立入口未核验三轮云端正文')
    before=sha256((run/'release/checkpoint.npz').read_bytes()).hexdigest()
    repeat=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/workflow/run_workflow.py'),
                           'native-record',str(run),'--online'],cwd=ROOT,capture_output=True,
                          text=True,encoding='utf-8',timeout=90)
    (DEST/'record-again.log').write_text(repeat.stdout+repeat.stderr,encoding='utf-8')
    if repeat.returncode:raise RuntimeError('独立完成记录未能复用')
    rows=[json.loads(line)['recording'] for line in repeat.stdout.splitlines() if line.startswith('{"recording":')]
    if len(rows)!=1 or not rows[0]['reused_completed_recording'] or rows[0]['uploaded_now']:
        raise ValueError('重复记录发生重新上传')
    if before!=sha256((run/'release/checkpoint.npz').read_bytes()).hexdigest():
        raise ValueError('重复记录改变模型')
    summary={'passed':True,'task':'gas','stages':3,'real_model_calls':verified['actual_model_calls'],
             'accepted_stages':[s['stage'] for s in report['stages'] if s['accepted']],
             'training_workers':verified['launched_training_workers'],'reused_candidates':verified['reused_candidates'],
             'scores':report['scores'],'rows':verified['issued_rows'],'release_bytes':verified['release_bytes'],
             'cloud_url':recording['url'],'cloud_verified':True,'cloud_evidence_stages':3,
             'completed_repeat_uploads':0,'source_program_scope':'真实Luna，正式PyTRiO驱动仍待额度'}
    atomic_json(DEST/'verification.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
