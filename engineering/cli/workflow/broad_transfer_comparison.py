"""原样调用冻结Harness/CoMind控制器；外层只负责宽域任务、共同确认与实际发布。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.author_mlestar import AuthorBroker
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.author_comind import run_comind
from harness.integrations.free_research_driver import run_free_research, INSTRUCTION
from harness.integrations.broad_transfer_task import verify_core, sources, description, metric, accepted
from harness.runtime.storage import atomic_json


PREDICT = '''from pathlib import Path
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
'''


async def run(args):
    verify_core()
    source = args.bundle.resolve()/args.task; out = args.output.resolve()
    if out.exists():
        raise ValueError('实际比较必须使用新输出目录')
    identity = json.loads((source/'task-identity.json').read_bytes())
    for name, checksum in identity['task_files'].items():
        if sha256((source/'task'/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('任务文件身份变化')
    for name, checksum in identity['private_files'].items():
        if sha256((source/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('封存外层文件身份变化')
    # 新组件实验将共同源码直接封存在任务包；旧任务继续核对原生成源码身份。
    # 代码只从该任务包读取，未来工作区组件更新不会悄悄改变这次比较。
    reference_file = source/'reference-programs.json'
    if 'reference_programs_sha256' in identity:
        if sha256(reference_file.read_bytes()).hexdigest() != identity['reference_programs_sha256']:
            raise ValueError('封存的共同源码文件变化')
        refs = json.loads(reference_file.read_bytes())
        if not isinstance(refs, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in refs.items()):
            raise ValueError('共同源码必须是名字到完整程序的映射')
    else:
        # 旧任务身份已明确登记当时拥有的工具；新增候选不能回写旧比较的能力范围。
        available = sources(args.task)
        refs = {name: available[name] for name in identity['reference_sources']}
    if {k: sha256(v.encode()).hexdigest() for k, v in refs.items()} != identity['reference_sources']:
        raise ValueError('共同参考源码变化')
    out.mkdir(parents=True); text = description(args.task, source/'task', reference_sources=refs)
    (out/'task_description.txt').write_text(text, encoding='utf-8')
    atomic_json(out/'input-identity.json', {'task': args.task, 'arm': args.arm,
                'bundle_identity_sha256': sha256((source/'task-identity.json').read_bytes()).hexdigest(),
                'description_sha256': sha256(text.encode()).hexdigest(), 'max_model_calls': 48})
    oracle = json.loads((source/'oracle.json').read_bytes())
    def measured(prediction):
        return metric(args.task, oracle['explore_y'], prediction, oracle['explore_groups'])
    workspace = out/'workspace'; workspace.mkdir()
    executor = AuthorExecutor(out/'executions', workspace,
        {'validation_y': np.asarray(oracle['explore_y']), 'confirmation_y': np.asarray(oracle['confirm_y'])}, scorer=measured)
    broker = AuthorBroker(out/'queue', max_calls=48, timeout=360)
    selected, failure = None, None
    try:
        if args.arm == 'comind':
            seed = '\n\n'.join('# '+name+'\n'+code for name, code in refs.items())
            selected = await run_comind(text, source/'task', workspace, executor, broker, seed)
        else:
            instruction = INSTRUCTION.replace('两个探索周的真实RMSE用于比较', '外层真实探索损失用于比较，分类损失是1减Macro-F1，回归损失是RMSE')
            instruction = instruction.replace('最后阶段会按冻结源码重新拟合合法发布权重；确认后的训练内成绩不会重新供你选择。',
                                              '选中模型的实际保存权重用于发布；探索与确认标签不会用于再训练。')
            selected = await run_free_research(text, source/'task', workspace, executor, broker,
                                               instruction=instruction, reference_sources=refs)
    except Exception:
        failure = traceback.format_exc(); (out/'driver-failure.log').write_text(failure, encoding='utf-8')
    atomic_json(out/'driver-complete.json', {'model_calls': broker.calls, 'search_calls': broker.search_calls,
                'executions': len(executor.records), 'research_completed': selected is not None})
    parent = [metric(args.task, oracle[name+'_y'], pd.read_csv(source/'task'/file).iloc[:, -1], oracle[name+'_groups'])
              for name, file in [('explore', 'validation.csv'), ('confirm', 'test.csv')]]
    candidate, checks, adopted = None, None, False
    if selected is not None:
        candidate = selected['candidates'][-1]
        with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
            checks = [metric(args.task, oracle[name+'_y'], z[key], oracle[name+'_groups'])
                      for name, key in [('explore', 'validation'), ('confirm', 'confirmation')]]
        adopted = all(accepted(args.task, a, b) for a, b in zip(parent, checks, strict=True))
    decision = {'selected_execution': selected['sequence'] if selected else None, 'candidate': candidate,
                'parent_metrics': parent, 'candidate_metrics': checks, 'adopted': adopted,
                'research_failure': failure is not None, 'model_calls': broker.calls, 'search_calls': broker.search_calls}
    atomic_json(out/'decision.json', decision)
    # 决策封存后只重载实际权重交付；两臂预测全部封存后由配对入口统一开服务答案。
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline, x = z['future'], z['baseline'], z['service_x']
    direct, serving_failure = baseline.copy(), None
    if candidate is not None:
        serving = out/'serving'; serving.mkdir()
        weights = Path(candidate['path']).with_suffix('.pkl')
        if sha256(weights.read_bytes()).hexdigest() != candidate['weights_sha256']:
            raise ValueError('实际候选权重变化')
        shutil.copy2(weights, serving/'model.pkl')
        columns = json.loads((source/'task/task.json').read_bytes())['feature_columns']
        pd.DataFrame(x[future], columns=columns).to_csv(serving/'service.csv', index=False)
        script = serving/'predict.py'; script.write_text(PREDICT, encoding='utf-8')
        env = os.environ.copy(); env.update(PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            process = subprocess.run([sys.executable, '-B', str(ROOT/'engineering/cli/study/execute_author_program.py'), '--script', str(script)],
                cwd=serving, env=env, capture_output=True, text=True, encoding='utf-8', timeout=900)
            (serving/'stderr.log').write_text(process.stderr, encoding='utf-8')
            if process.returncode:
                raise RuntimeError(process.stderr[-2000:])
            values = np.load(serving/'prediction.npy', allow_pickle=False)
            if values.shape != (int(future.sum()),) or not np.isfinite(values).all():
                raise ValueError('实际服务输出不完整')
            if args.task in ['sgp40', 'secom'] and not np.isin(values, [1, 2]).all():
                raise ValueError('实际类别输出非法')
            direct[future] = values
        except Exception:
            serving_failure = traceback.format_exc()
            (serving/'delivery-failure.log').write_text(serving_failure, encoding='utf-8')
            adopted = False
    actual = direct if adopted else baseline.copy()
    if not np.array_equal(actual[~future], baseline[~future]):
        raise ValueError('既有交付被改写')
    np.savez_compressed(out/'issued.npz', ids=np.asarray(oracle['service_ids']), baseline=baseline, direct=direct, actual=actual)
    report = {'task': args.task, 'arm': args.arm, 'selection_closed': True, 'rows': len(actual),
              'adopted': adopted, 'research_completed': selected is not None, 'delivery_failure': serving_failure is not None,
              'model_calls': broker.calls, 'search_calls': broker.search_calls, 'executions': len(executor.records),
              'failed_executions': sum(r['returncode'] != 0 for r in executor.records),
              'execution_seconds': sum(r['execution_time'] for r in executor.records),
              'issued_sha256': sha256((out/'issued.npz').read_bytes()).hexdigest()}
    atomic_json(out/'predictions-complete.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--task', required=True, choices=['sgp40', 'secom', 'wind', 'ncmapss_ds06'])
    parser.add_argument('--arm', required=True, choices=['project', 'comind']); parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
