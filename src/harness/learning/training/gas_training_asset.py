"""仅供训练进程复用确定的教师软目标。预测与公开验证总是计算实际模型输出。"""
from hashlib import sha256
import io
import json
import os
from pathlib import Path

import numpy as np


DIRECTORY = Path('/opt/zhixun/gas-training-v1')


def training_identity(model, raw):
    settings = ['tabpfn-training-128-v1', os.environ.get('ZHIXUN_SDK_PREDICT_THREADS'),
                model.meta['paths'], model.meta['classes'], model.teacher_state, model.state['baseline_context']]
    digest = sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode())
    arrays = [model.auxiliary_arrays[k] for k in
              ('classifier_source_x', 'classifier_source_y', 'baseline_context_x', 'baseline_context_y')]
    for values in [*arrays, raw]:
        matrix = np.ascontiguousarray(values)
        digest.update(matrix.dtype.str.encode())
        digest.update(np.asarray(matrix.shape, dtype='<i8').tobytes())
        digest.update(matrix.tobytes())
    return digest.hexdigest()


def restore_training_targets(model, raw):
    raw = np.asarray(raw, dtype=np.float64)
    manifest = DIRECTORY/'manifest.json'
    if not manifest.is_file() or not model.state.get('baseline_context'):
        return False
    key = training_identity(model, raw)
    try:
        expected = json.loads(manifest.read_bytes()).get(key)
        if not expected:
            return False
        payload = (DIRECTORY/(expected+'.npy')).read_bytes()
        if sha256(payload).hexdigest() != expected:
            return False
        values = np.load(io.BytesIO(payload), allow_pickle=False)
        if (values.dtype not in (np.float32, np.float64) or values.shape != (len(raw), len(model.meta['classes']))
                or not np.isfinite(values).all()):
            return False
    except (OSError, ValueError, EOFError):
        return False
    model._training_teacher.update((row.tobytes(), value.copy()) for row, value in zip(raw, values, strict=True))
    return True
