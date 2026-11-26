"""从固定官方 A 开发包准备构建输入，仅提取公开训练部分。"""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from harness.adapters.platforms.baselines.core_model import Model
from harness.adapters.platforms.baselines.data import read_dataset

OFFICIAL = '7a992b92d92c807869c3eccf25b3bf82dcdeba9174e161d6e8b61d82c3ec8851'


def prepare(archive, output):
    with archive.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != OFFICIAL:
            raise ValueError('气体开发包摘要与发行配方不符')
    with ZipFile(archive) as source, TemporaryDirectory() as temp:
        parent = Path(temp)/'parent.npz'
        parent.write_bytes(source.read('checkpoint.npz'))
        meta = Model.load(parent).meta
        issued = Path(temp)/'train.jsonl'
        with source.open('public/train.jsonl') as rows, issued.open('wb') as projected:
            for line in rows:
                row = json.loads(line)
                record = {k:row[k] for k in ('id','inputs','answer')}
                projected.write((json.dumps(record,ensure_ascii=False)+'\n').encode())
        data = read_dataset(issued, meta['task'], meta['classes'], meta['paths'])
    np.savez_compressed(output, x=data['x'], y=data['y'], classes=json.dumps(meta['classes']), paths=json.dumps(meta['paths']))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/gas-fit-source.npz')
    args = parser.parse_args()
    prepare(args.archive, args.output)
