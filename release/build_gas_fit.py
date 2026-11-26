"""构建只读的单线程气体教师状态。输入仅含官方训练特征、源标签和特征顺序。"""
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from harness.learning.inference.quantized_teacher import restore_teacher
from harness.learning.inference.tabpfn_classifier import author_package


def build(source, output):
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with np.load(source, allow_pickle=False) as data, np.load('/app/assets/source-a.npz', allow_pickle=False) as frozen:
        x, y = data['x'], data['y']
        classes, paths = json.loads(str(data['classes'])), json.loads(str(data['paths']))
        spec = json.loads(str(frozen['tabpfn_classifier_state']))
        assert classes == spec['classes'] and x.shape == (len(y), len(paths))
        assert set(y) == set(range(len(classes)))
        info = dict(teacher=spec['asset_sha256'], classes=classes, paths=paths, count=3, seed=42)
        for key, values in [('x', x), ('y', y)]:
            info[key] = [values.dtype.str, list(values.shape), sha256(values.tobytes()).hexdigest()]
        with TemporaryDirectory() as temp:
            weights = Path(temp)/'tabpfn-v3.5-fast-20260909.safetensors'
            restore_teacher(frozen['teacher_asset'], weights)
            predictor = author_package().TabPFNClassifier(model_path=weights, n_estimators=info['count'], device='cpu',
                random_state=info['seed'], fit_mode='fit_with_cache', n_preprocessing_jobs=1, inference_precision=torch.float32)
            predictor.fit(x.astype(np.float32), y)
            archive = output/'raw.tabpfn_fit'
            predictor.save_fit_state(archive)
    with archive.open('rb') as stream:
        info['sha256'] = file_digest(stream, 'sha256').hexdigest()
    (output/'manifest.json').write_text(json.dumps(info, sort_keys=True), encoding='utf8')


if __name__ == '__main__':
    build(Path(sys.argv[1]), Path(sys.argv[2]))
