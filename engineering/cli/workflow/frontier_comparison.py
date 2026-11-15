"""相同任务、供应方和预算下，比较当前完整研究驱动与 CoMind 作者框架。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from harness.integrations.author_mlestar import AuthorBroker, ACTOR
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.author_comind import run_comind
from harness.integrations.free_research_driver import run_free_research, INSTRUCTION
from harness.integrations.frontier_task import description, sources
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json
from engineering.cli.workflow.official_components import accepted


async def run(args):
    source = args.bundle.resolve() / args.task
    out = args.output.resolve()
    if out.exists(): raise ValueError('每次真实比较使用新目录，不覆盖失败')
    out.mkdir(parents=True)
    identity = json.loads((source / 'task-identity.json').read_bytes())
    oracle = json.loads((source / 'oracle.json').read_bytes())
    for name, checksum in identity['task_files'].items():
        if sha256((source / 'task' / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('共同任务文件变化：' + name)
    text = description(args.task, source / 'task')
    (out / 'task_description.txt').write_text(text, encoding='utf-8')
    atomic_json(out / 'input-identity.json', {'task': args.task, 'arm': args.arm,
        'bundle_identity_sha256': sha256((source / 'task-identity.json').read_bytes()).hexdigest(),
        'description_sha256': sha256(text.encode()).hexdigest(), 'max_model_calls': args.calls,
        'common_confirmation_and_delivery': True})
    def measured(prediction):
        value = metric(args.task, oracle['explore_y'], prediction, oracle['explore_groups'])
        return {**value, 'loss': 1 - value['score'] if args.task == 'gas' else value['score']}
    workspace = out / 'workspace'
    workspace.mkdir()
    executor = AuthorExecutor(out / 'executions', workspace,
        {'validation_y': np.asarray(oracle['explore_y']), 'confirmation_y': np.asarray(oracle['confirm_y'])}, scorer=measured)
    broker = AuthorBroker(out / 'queue', max_calls=args.calls, timeout=360)
    try:
        if args.arm == 'comind':
            seed = '\n\n'.join('# ' + key + '\n' + code for key, code in sources(args.task).items())
            selected = await run_comind(text, source / 'task', workspace, executor, broker, seed)
        else:
            instruction = INSTRUCTION.replace('两个探索周的真实RMSE用于比较', '外层真实探索损失用于比较，分类损失是1减Macro-F1，回归损失是RMSE')
            instruction = instruction.replace('最后阶段会按冻结源码重新拟合合法发布权重；确认后的训练内成绩不会重新供你选择。',
                                              '选中模型的实际保存权重用于发布；探索与确认标签不会用于再训练。')
            selected = await run_free_research(text, source / 'task', workspace, executor, broker,
                                              instruction=instruction, reference_sources=sources(args.task))
    finally:
        atomic_json(out / 'driver-complete.json', {'model_calls': broker.calls, 'search_calls': broker.search_calls,
                    'executions': len(executor.records)})
    candidate = selected['candidates'][-1]
    with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
        checks = [metric(args.task, oracle['explore_y'], z['validation'], oracle['explore_groups']),
                  metric(args.task, oracle['confirm_y'], z['confirmation'], oracle['confirm_groups'])]
    parent = [metric(args.task, oracle['explore_y'], pd.read_csv(source / 'task/validation.csv').iloc[:, -1], oracle['explore_groups']),
              metric(args.task, oracle['confirm_y'], pd.read_csv(source / 'task/test.csv').iloc[:, -1], oracle['confirm_groups'])]
    adopted = all(accepted(args.task, a, b) for a, b in zip(parent, checks, strict=True))
    decision = {'selected_execution': selected['sequence'], 'candidate': candidate, 'parent_metrics': parent,
                'candidate_metrics': checks, 'adopted': adopted, 'model_calls': broker.calls,
                'search_calls': broker.search_calls, 'source_sha256': selected['source_sha256']}
    atomic_json(out / 'decision.json', decision)
    # 搜索及确认决定已经封存，之后不再发出 LLM 请求。
    with np.load(source / 'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline, x = z['future'], z['baseline'], z['service_x']
    serving = out / 'serving'
    serving.mkdir()
    weights = Path(candidate['path']).with_suffix('.pkl')
    if sha256(weights.read_bytes()).hexdigest() != candidate['weights_sha256']:
        raise ValueError('发布模型权重变化')
    shutil.copy2(weights, serving / 'model.pkl')
    columns = json.loads((source / 'task/task.json').read_bytes())['feature_columns']
    pd.DataFrame(x[future], columns=columns).to_csv(serving / 'service.csv', index=False)
    script = serving / 'predict.py'
    script.write_text('''from pathlib import Path
import cloudpickle
import pandas as pd
import numpy as np
model=cloudpickle.loads(Path('model.pkl').read_bytes())
data=pd.read_csv('service.csv')
values=[]
for i in range(len(data)):
    value=np.asarray(model.predict(data.iloc[[i]]),dtype=float)
    if value.shape!=(1,) or not np.isfinite(value).all():raise ValueError('逐条预测无效')
    values.append(max(0.,float(value[0])))
np.save('prediction.npy',np.asarray(values))
''', encoding='utf-8')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = subprocess.run([sys.executable, '-B', str(ROOT / 'engineering/cli/study/execute_author_program.py'),
        '--script', str(script)], cwd=serving, env=env, capture_output=True, text=True, encoding='utf-8', timeout=240)
    (serving / 'stderr.log').write_text(process.stderr, encoding='utf-8')
    if process.returncode: raise RuntimeError('完整发布失败：' + process.stderr[-1000:])
    direct = baseline.copy()
    direct[future] = np.load(serving / 'prediction.npy', allow_pickle=False)
    actual = direct if adopted else baseline.copy()
    np.savez_compressed(out / 'issued.npz', ids=np.asarray(oracle['service_ids']), baseline=baseline, direct=direct, actual=actual)
    atomic_json(out / 'predictions-complete.json', {'selection_closed': True, 'rows': len(actual), 'used_new_model': adopted})
    from engineering.cli.workflow import public_maintenance as public
    package = public.read_official_gas(Path(identity['archive'])) if args.task == 'gas' else public.PACKAGES[args.task][2](Path(identity['archive']))
    answers = {r.sample_id: r.label for r in package.validation}
    y = np.asarray([answers[k] for k in oracle['service_ids']])
    scores = {name: metric(args.task, y, values, oracle['service_groups'])
              for name, values in [('baseline', baseline), ('direct', direct), ('actual', actual)]}
    result = {'task': args.task, 'arm': args.arm, 'scope': '官方已见开发任务，同条件完整研究流程及十轮交付',
              'scores': scores, 'model_calls': broker.calls, 'search_calls': broker.search_calls,
              'executions': len(executor.records), 'execution_seconds': sum(r['execution_time'] for r in executor.records),
              'adopted': adopted, 'delivery_rows': len(actual), 'future_rows': int(future.sum()),
              'model_reloaded_for_delivery': True}
    atomic_json(out / 'scores.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--task', choices=['gas', 'bearing', 'engine'], required=True)
    parser.add_argument('--arm', choices=['comind', 'project'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=48)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__': main()
