"""打包当前原生候选，并在只含解压文件的目录重建权重；不冒充Docker镜像。"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from harness.integrations.native_research import export_release
from harness.runtime.storage import atomic_json


def checked_run(command, *, cwd, env, data=None):
    result = subprocess.run(command, cwd=cwd, env=env, input=data, capture_output=True,
                            text=True, encoding='utf-8', timeout=180)
    if result.returncode: raise RuntimeError(result.stderr[-4000:] + result.stdout[-1000:])
    return result.stdout


def build(selection, data_root, output):
    output = output.resolve()
    if output.exists(): raise ValueError('交付输出已存在，请保留原版本并使用新目录')
    chosen = json.loads(selection.read_bytes())
    output.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k.upper() in
           {'SYSTEMROOT', 'WINDIR', 'PATH', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP', 'LANG', 'LC_ALL'}}
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with TemporaryDirectory(dir=output, prefix='temporary-build-') as temporary:
        scratch = Path(temporary)
        app = scratch / 'app'
        app.mkdir()
        for directory in ('engineering/src', 'engineering/cli', 'engineering/configs', 'engineering/tests/fixtures'):
            shutil.copytree(ROOT / directory, app / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(ROOT / 'engineering/__init__.py', app / 'engineering/__init__.py')
        models = {}
        for task, specification in chosen['tasks'].items():
            for path, expected in specification['inputs_sha256'].items():
                if sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise ValueError('事前冻结的候选或来源已变化')
            model_dir = app / 'models' / task
            release = export_release(Path(specification['task_dir']), Path(specification['selected']), model_dir)
            initial = Path(specification['initial'])
            with np.load(initial, allow_pickle=False) as arrays:
                meta = json.loads(str(arrays['metadata']))
            task_info = json.loads((Path(specification['task_dir']) / 'task.json').read_bytes())
            with ZipFile(task_info['archive']) as archive:
                rows = [json.loads(row) for row in archive.read('public/validation.jsonl').splitlines()[:12]]
            requests = '\n'.join(json.dumps({'task': meta['task'], 'inputs': row['inputs']}) for row in rows) + '\n'
            examples = app / 'examples'; examples.mkdir(exist_ok=True)
            (examples / (task + '.jsonl')).write_text(requests, encoding='utf-8')
            actual = checked_run([sys.executable, '-B', 'baselines/predict.py', '--checkpoint', 'checkpoint.npz'],
                                 cwd=model_dir, env=env, data=requests)
            atomic_json(examples / (task + '-predictions.json'), [json.loads(line) for line in actual.splitlines()])
            models[task] = {'checkpoint': f'models/{task}/checkpoint.npz',
                'checkpoint_sha256': specification['selected_sha256'],
                'parent_sha256': sha256(initial.read_bytes()).hexdigest(),
                'program': f'models/{task}/training-program.py', 'recipe_name': specification['selected_name'],
                'archive_sha256': task_info['archive_sha256'], 'artifact_bytes': release['artifact_bytes'],
                'example_inputs': f'examples/{task}.jsonl', 'example_predictions': f'examples/{task}-predictions.json',
                'official_validation_score': specification['official_validation_score']}
        # 仅复制相关正式操作说明；本机历史结果和原始训练数据不进入交付源码包。
        (app / 'docs').mkdir()
        guide = (ROOT / 'docs/operations/原生源码续训.md').read_text(encoding='utf-8')
        guide = re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]+\)', r'\1（完整工作区资料）', guide)
        (app / 'docs/原生源码续训.md').write_text(guide, encoding='utf-8')
        (app / 'requirements-native.txt').write_text(f'numpy=={np.__version__}\npsutil==7.2.2\nswanlab==0.10.4\n', encoding='utf-8')
        docker = '''FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements-native.txt ./
RUN python -m pip install --no-cache-dir -r requirements-native.txt
COPY engineering ./engineering
COPY models ./models
COPY examples ./examples
COPY candidate-manifest.json ./
ENTRYPOINT ["python", "-B", "engineering/cli/workflow/run_workflow.py"]
CMD ["native-research", "gas", "--driver", "service", "--stages", "10", "--recording", "online", "--data-root", "/data", "--output", "/output"]
'''
        (app / 'Dockerfile.native').write_text(docker, encoding='utf-8')
        (app / '.dockerignore').write_text('*\n!engineering/\n!engineering/**\n!models/\n!models/**\n!examples/\n!examples/**\n!requirements-native.txt\n!candidate-manifest.json\n', encoding='utf-8')
        readme = '''# 智训引擎原生候选源码与权重

这是可复现源码包，包含三个实际续训候选及独立预测器；不是docker save镜像，也不是平台已通过回执。

## 独立预测
进入models/gas（或bearing、engine），运行：
`python -B -m baselines.predict --checkpoint checkpoint.npz`
标准输入为官方JSONL，示例在examples目录。预测只需Python和NumPy。

## 从官方父版重建
在包根目录准备Python 3.13与requirements-native.txt依赖，将原始官方zip按gas/bearing/engine子目录放入数据目录，然后运行：
`python -B engineering/cli/release/reproduce_native_candidate.py --task gas --data-root <数据目录> --output <新的重建目录>`
其他两项替换任务名。工具核验原包、源程序和父检查点，再真实训练并核对重建张量及独立预测。

## 源码驱动维护
`python -B engineering/cli/workflow/run_workflow.py native-research gas --driver service --calls 4 --stages 10 --recording online --data-root <数据目录> --parent-checkpoint models/gas/checkpoint.npz --output <新的运行目录>`
以用户运行时环境中的PYTRIO_API_KEY访问Qwen/Qwen3.8-27B。密钥不在包内，服务须有可用额度。
外部实际父版通过--parent-checkpoint传入；恢复使用原命令加--resume。完整接口见docs/原生源码续训.md。

## 独立云端记录与补传
`--recording online`在运行完成后直接记录到已配置SwanLab项目，不依赖个人科研运行器。SwanLab凭据由运行时SDK登录或SWANLAB_API_KEY提供，不能写进镜像。
如果训练已经完成而网络上传中断，只运行：
`python -B engineering/cli/workflow/run_workflow.py native-record <已完成运行目录> --online`
记录身份在连接云端前固定；恢复只补传缺失轮次，完成后的重复调用直接复用回执。`--recording local`保留全部本地过程而不上传。
各任务的实际权重、训练程序和选择依据见candidate-manifest.json；模型风险和正式平台验收按相应实测说明。

## 镜像构建准备
`docker build -f Dockerfile.native -t zhixun-native .`
`docker save -o zhixun-native.tar zhixun-native`
镜像须挂载官方数据到/data、输出目录到/output，并按实际任务覆盖命令；平台SDK具体契约还需对接。
本包只完成宿主机隔离目录中的源码与权重重建核验，Docker构建、容器运行和正式PyTRiO调用仍待完成。
'''
        (app / 'README.md').write_text(readme, encoding='utf-8')
        files = {}
        for path in sorted(app.rglob('*')):
            if not path.is_file(): continue
            raw = path.read_bytes()
            if path.suffix in ('.py', '.md', '.json', '.txt') and re.search(rb'(?:trio_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,})', raw):
                raise ValueError('待打包文本含疑似真实凭据，停止发布')
            files[path.relative_to(app).as_posix()] = sha256(raw).hexdigest()
        manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version.split()[0],
                    'numpy': np.__version__, 'selection_sha256': sha256(selection.read_bytes()).hexdigest(),
                    'selection_basis': chosen['selection_basis'],
                    'models': models, 'files': files, 'container_built': False}
        atomic_json(app / 'candidate-manifest.json', manifest)
        zip_path = scratch / 'candidate-source.zip'
        with ZipFile(zip_path, 'w', ZIP_DEFLATED) as archive:
            for path in sorted(app.rglob('*')):
                if path.is_file(): archive.write(path, path.relative_to(app))
        unpacked = scratch / 'unpacked'
        with ZipFile(zip_path) as archive: archive.extractall(unpacked)
        verified = {}
        for task in models:
            text = checked_run([sys.executable, '-I', '-B', 'engineering/cli/release/reproduce_native_candidate.py',
                '--task', task, '--data-root', str(data_root.resolve()), '--output', str(scratch / ('rebuild-' + task))],
                cwd=unpacked, env=env)
            verified[task] = json.loads(text.strip().splitlines()[-1])
        native_help = checked_run([sys.executable, '-B', 'engineering/cli/workflow/run_workflow.py',
                                   'native-research', '--help'], cwd=unpacked, env=env)
        recording_help = checked_run([sys.executable, '-B', 'engineering/cli/workflow/run_workflow.py',
                                      'native-record', '--help'], cwd=unpacked, env=env)
        if '--recording' not in native_help or '--online' not in recording_help:
            raise ValueError('解压包的独立记录入口未接通')
        default_line = next(line for line in (unpacked / 'Dockerfile.native').read_text(encoding='utf-8').splitlines()
                            if line.startswith('CMD '))
        default_command = json.loads(default_line[4:])
        if default_command[default_command.index('--stages') + 1] != '10':
            raise ValueError('完整比赛准备入口未设为十轮')
        final = output / 'zhixun-native-source.zip'
        shutil.copyfile(zip_path, final)
        atomic_json(output / 'candidate-manifest.json', manifest)
        report = {'source_bundle': str(final), 'sha256': sha256(final.read_bytes()).hexdigest(),
                  'bundle_bytes': final.stat().st_size, 'files': len(files) + 1,
                  'all_model_artifacts_bytes': sum(m['artifact_bytes'] for m in models.values()),
                  'container_built': False, 'isolated_unpacked_reproduction': verified,
                  'real_service_called': False, 'secret_scan_passed': True,
                  'unpacked_recording_entrypoints_verified': True,
                  'default_native_stages': 10}
        atomic_json(output / 'verification.json', report)
    return report


def main():
    parser = ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.selection, args.data_root, args.output), ensure_ascii=False))


if __name__ == '__main__': main()
