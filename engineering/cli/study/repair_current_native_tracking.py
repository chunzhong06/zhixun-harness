"""只续传当前未完成作业的逐轮记录；不重训、不扫描或补传历史实验。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'engineering/src'))
from harness.integrations.research_tracking import ResearchSwanLab
from run_tracked_experiment import scalars
from harness.runtime.storage import atomic_json


def main():
    ap=ArgumentParser()
    ap.add_argument('--run',required=True)
    args=ap.parse_args()
    state=json.loads((ROOT/'research-state.json').read_bytes())
    if state['runs'][-1]['id']!=args.run or state['active_run'] is not None:
        raise ValueError('只修复当前已结束计算、尚未完成记录的作业')
    receipt=ROOT/'results/experiments/contest-continuation/jobs'/args.run
    record=json.loads((receipt/'run.json').read_bytes())
    if record['status']!='succeeded':
        raise ValueError('计算未成功，不按成功作业续传')
    folder=ROOT/'results/engineering/research-workflow/swanlab'/args.run
    previous=json.loads((folder/'cloud.json').read_bytes())
    if previous.get('cloud_verified'):
        raise ValueError('已经核验的历史作业不重新上传')
    archive=folder/'cloud-before-recording-repair.json'
    if not archive.exists():shutil.copyfile(folder/'cloud.json',archive)
    settings=json.loads((ROOT/'engineering/configs/tracking/swanlab.json').read_bytes())
    import swanlab
    remote=swanlab.Api().run(f"{settings['workspace']}/{settings['project']}/{previous['id']}")
    config={k:v.get('value') for k,v in remote.profile.get('config',{}).items()}
    if config.get('研究运行号')!=args.run:
        raise ValueError('云端目标不属于当前作业')
    config.pop('执行过程证据',None)
    config['当前记录修复']='原profile请求过大；逐轮正文改用Text媒体，同一作业、同一计算证据。'
    original=(folder/'execution-evidence.json').read_bytes()
    expected=json.loads(original)
    tracker=ResearchSwanLab(settings,folder,name=remote.name,config=config,kind='smoke',
              description=remote.description,resume_id=previous['id'])
    tracker.log({'lifecycle/started':1},0)
    step=1
    for line in (receipt/'stdout.log').read_text(encoding='utf-8').splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'native_trace' in value:
            if not tracker.record_evidence(value['native_trace'],step=step):
                continue
        metrics=scalars({k:v for k,v in value.items() if k!='native_trace'}) if isinstance(value,dict) else scalars(value)
        if metrics:tracker.log(metrics,step);step+=1
    if tracker.evidence!=expected:
        raise ValueError('修复改变了原始过程证据')
    tracker.log({'runtime/seconds':record['elapsed_seconds'],'lifecycle/succeeded':1,'recording_repair/new_training_calls':0},step)
    result=tracker.finish(True)
    repair={'run':args.run,'source_stdout_sha256':sha256((receipt/'stdout.log').read_bytes()).hexdigest(),
            'original_evidence_file_sha256':sha256(original).hexdigest(),'new_training_calls':0,
            'same_cloud_run':tracker.id==previous['id'],'evidence_count':len(tracker.evidence),
            'cloud_verified':result['cloud_verified'],'evidence_cloud_verified':result['evidence_cloud_verified'],
            'reason':'原始配置上传超过服务端请求大小，停止无效重试，改用逐轮Text记录'}
    atomic_json(folder/'recording-repair.json',repair)
    print(json.dumps({**repair,'url':result['url'],'verification_error':result['verification_error']},ensure_ascii=False))
    return 0 if result['cloud_verified'] else 3


if __name__=='__main__':
    raise SystemExit(main())
