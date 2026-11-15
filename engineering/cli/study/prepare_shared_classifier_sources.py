"""逐字复制旧合法任务包，只增加双方相同且绑定身份的分类组件源码。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import sources, verify_core
from harness.runtime.storage import atomic_json


def main():
    parser = ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['sgp40', 'secom'], required=True)
    args = parser.parse_args(); verify_core()
    source, dest = args.bundle/args.task, args.output/args.task
    if dest.exists():
        raise ValueError('不覆盖已有任务包')
    identity = json.loads((source/'task-identity.json').read_bytes())
    for key, prefix in [('task_files', source/'task'), ('private_files', source)]:
        for name, checksum in identity[key].items():
            if sha256((prefix/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('旧输入身份改变')
    available = sources(args.task)
    refs = {name: available[name] for name in identity['reference_sources']}
    if {name: sha256(code.encode()).hexdigest() for name, code in refs.items()} != identity['reference_sources']:
        raise ValueError('原共同能力发生变化')
    component = ROOT/'engineering/src/harness/models/soft_state_classifier.py'
    refs['soft-state-classifier'] = available['soft-state-classifier']
    shutil.copytree(source, dest)
    # 既有数据、父版、答案及预处理不变；新身份仅登记共同能力的明确增量。
    for key, prefix in [('task_files', dest/'task'), ('private_files', dest)]:
        for name, checksum in identity[key].items():
            if sha256((prefix/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('复制改变输入内容')
    atomic_json(dest/'reference-programs.json', refs)
    identity.update(reference_programs_sha256=sha256((dest/'reference-programs.json').read_bytes()).hexdigest(),
                    reference_sources={name: sha256(code.encode()).hexdigest() for name, code in refs.items()},
                    predecessor_identity_sha256=sha256((source/'task-identity.json').read_bytes()).hexdigest(),
                    component_source_sha256=sha256(component.read_bytes()).hexdigest())
    atomic_json(dest/'task-identity.json', identity)
    print(json.dumps({'task': args.task, 'data_and_parent_unchanged': True, 'shared_references': list(refs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
