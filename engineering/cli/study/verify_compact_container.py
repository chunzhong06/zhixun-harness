"""验证精简镜像与原镜像的运行文件一致，并复跑真实续训与三类发布入口。"""
from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = 'zhixun-harness:trial-20261115'
COMPACT = 'zhixun-harness:compact-20261115'


def linux(path):
    return '/mnt/' + str(path)[0].lower() + str(path)[2:].replace('\\', '/')


def docker(args, *, stdin=None, timeout=360):
    result = subprocess.run(['wsl', '-d', 'Ubuntu', '-u', 'root', '--', 'docker', *args],
                            input=stdin, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8', 'replace')[-4000:])
    return result.stdout


def main():
    out = ROOT / 'results/engineering/contest-continuation/compact-verification'
    resume = '--resume' in sys.argv
    out.mkdir(parents=True, exist_ok=resume)
    prune = json.loads(docker(['run', '--rm', '--network', 'none', '--entrypoint', 'cat',
                              COMPACT, '/opt/contest-dependency-prune.json']))
    (out/'dependency-prune.json').write_text(json.dumps(prune, indent=2), encoding='utf8')
    removed = [row['path'] for row in prune['removed']]
    # 对原镜像排除已记录的测试/缓存，再逐个哈希所有保留文件；两边内容须精确一致。
    program = '''from pathlib import Path
from hashlib import sha256
import json
excluded = REMOVED
result={}
for name,location in [('app','/app'),('dependencies','/usr/local/lib/python3.13/site-packages')]:
 root=Path(location);digest=sha256();count=0;size=0
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  relative=p.relative_to(root).as_posix()
  if name=='dependencies' and any(relative==x or relative.startswith(x+'/') for x in excluded):continue
  data=p.read_bytes();digest.update(relative.encode()+b'\\0'+sha256(data).digest())
  count+=1;size+=len(data)
 result[name]={'sha256':digest.hexdigest(),'files':count,'bytes':size}
print(json.dumps(result))
'''.replace('REMOVED', repr(removed))
    identities = {}
    for label, image in [('original', BASE), ('compact', COMPACT)]:
        identities[label] = json.loads(docker(['run', '--rm', '-i', '--network', 'none',
            '--entrypoint', 'python', image, '-B', '-'], stdin=program.encode()))
    assert identities['original'] == identities['compact'], '保留运行文件不一致'
    (out/'file-identity.json').write_text(json.dumps(identities, indent=2), encoding='utf8')

    fixtures = ROOT/'scratch/contest-adaptation-20261115/platform-fixtures'
    target = out/'platform-entry';target.mkdir(exist_ok=resume)
    check = ROOT/'engineering/cli/study/verify_platform_entry_fixture.py'
    if not resume:
        log = docker(['run', '--rm', '--network', 'none', '-v', linux(fixtures)+':/fixtures:ro',
                      '-v', linux(check)+':/check.py:ro', '-v', linux(target)+':/output',
                      '--entrypoint', 'python', COMPACT, '-B', '/check.py'])
        (out/'platform-entry.log').write_bytes(log)
    for task in ('gas', 'bearing', 'engine'):
        before = json.loads((ROOT/f'results/engineering/contest-continuation/platform-entry-fixture/{task}/training.json').read_bytes())
        after = json.loads((target/task/'training.json').read_bytes())
        for key in ('decision', 'validation_before', 'validation_after'):
            assert before[key] == after[key], (task, key)
        assert before['training_audit']['final_tensors'] == after['training_audit']['final_tensors'], task

    task = ROOT/'scratch/contest-adaptation-20261115/container-task'
    common = ['run', '--rm', '--network', 'none', '--cpus', '4', '--memory', '8g',
              '-v', linux(task)+':/input:ro', '-v', linux(ROOT/'data/official')+':/data:ro',
              '-v', linux(out)+':/output', '--entrypoint', 'python', COMPACT]
    parent = task/'initial.npz'
    for round_id, parent_path, program_path, epochs in [
        (1, '/input/initial.npz', '/input/program.py', '60'),
        (2, '/output/round-1/checkpoint.npz', '/input/next.py', '1')]:
        if not resume:
            log = docker(common+['-B', 'engineering/cli/workflow/native_continuation.py',
                '--task-dir', '/input', '--parent', parent_path, '--expected-parent-sha256', sha256(parent.read_bytes()).hexdigest(),
                '--program', program_path, '--output', f'/output/round-{round_id}', '--max-epochs', epochs])
            (out/f'round-{round_id}.log').write_bytes(log)
        parent = out/f'round-{round_id}/checkpoint.npz'
    a = json.loads((out/'round-1/training.json').read_bytes())
    b = json.loads((out/'round-2/training.json').read_bytes())
    assert b['loaded_tensors'] == a['final_tensors']
    comparisons = {}
    for round_id in (1, 2):
        old = ROOT/f'results/engineering/contest-continuation/container-verification/round-{round_id}'
        new = out/f'round-{round_id}'
        before = json.loads((old/'training.json').read_bytes())
        after = json.loads((new/'training.json').read_bytes())
        assert before['final_tensors'] == after['final_tensors'], f'round-{round_id} 张量不一致'
        with np.load(old/'predictions.npz') as original, np.load(new/'predictions.npz') as compact:
            comparisons[str(round_id)] = {key:bool(np.array_equal(original[key], compact[key])) for key in original.files}
            assert all(comparisons[str(round_id)].values()), '预测不一致'

    rows = [json.loads(line) for line in (fixtures/'engine/validation.jsonl').read_bytes().splitlines() if line.strip()]
    requests = ''.join(json.dumps({'task':'regression','inputs':row['inputs']})+'\n' for row in rows).encode()
    release = out/'standalone-model';release.mkdir(exist_ok=resume)
    shutil.copy2(out/'round-2/checkpoint.npz', release/'checkpoint.npz')
    shutil.copytree(target/'engine/baselines', release/'baselines', dirs_exist_ok=resume)
    published = {}
    for label,image in [('original',BASE),('compact',COMPACT)]:
        prediction = docker(['run','--rm','-i','--network','none','-v',
            linux(release)+':/model:ro',image,'--predict','/model'], stdin=requests)
        published[label] = [json.loads(line) for line in prediction.splitlines() if line.strip()]
    assert published['original'] == published['compact'], '独立预测不一致'
    assert len(published['compact']) == 12
    report = {'runtime_files_identical':True, 'three_task_training_identical':True,
              'two_round_parent_inheritance':True, 'two_round_tensors_identical':True,
              'predictions_identical':comparisons, 'standalone_prediction_rows':len(published['compact']),
              'standalone_prediction_identical':True, 'simulated_model_service_for_cli_checks':True,
              'removed_dependency_bytes':prune['removed_bytes']}
    (out/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
