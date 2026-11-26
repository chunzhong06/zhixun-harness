"""同一冻结教师实例的有界批量预取；不跨模型或跨输入身份复用。"""
from hashlib import sha256
import os
import re
import shutil
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

REGRESSION_QUERY_ROWS = 128


def copy_training_hint(source, destination, *, name=None):
    """接续单份有界训练预计算。命中仍须由本轮的教师、训练特征和内容摘要共同确认。"""
    if not source.is_dir() or source.is_symlink():
        return False
    pattern = r'training-[a-f0-9]{64}\.npz'
    if name is None:
        # 旧状态没有身份记录，只有单份合法文件时才可向后传递。
        files = [p for p in source.glob('training-*.npz') if re.fullmatch(pattern, p.name)]
        if len(files) != 1:
            return False
        selected = files[0]
    elif isinstance(name, str) and re.fullmatch(pattern, name):
        selected = source / name
    else:
        return False
    try:
        if selected.is_symlink() or not selected.is_file() or selected.stat().st_size > 1024 * 1024:
            return False
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected, destination / selected.name)
        return True
    except OSError:
        return False


def frozen_regression_values(folder, namespace, raw, memory, predict, *, training=False):
    return frozen_teacher_values(folder, namespace, raw, memory, predict, training=training)


def frozen_teacher_values(folder, namespace, raw, memory, predict, *, training=False, width=None):
    """复用整个输入批次的冻结教师输出，保留分类概率原有精度及回归文件格式。"""
    def compute(batch):
        prefetch_teacher(memory, batch, predict, width=width, nonnegative=width is None,
                         batch_size=REGRESSION_QUERY_ROWS if width is None else 32)
        return np.asarray([memory[row.tobytes()] for row in batch], dtype=np.float64 if width is None else None)

    values, hit = frozen_batch_values(folder, namespace, raw, compute, training=training, width=width)
    if hit:
        memory.update((row.tobytes(), float(v) if width is None else v.copy()) for row, v in zip(raw, values, strict=True))
    return hit


def batch_cache_key(namespace, raw):
    key = sha256(namespace.encode('utf-8'))
    key.update(np.asarray(raw.shape, dtype='<i8').tobytes())
    key.update(np.ascontiguousarray(raw, dtype='<f8').tobytes())
    return key


def frozen_batch_values(folder, namespace, raw, compute, *, training=False, width=None):
    """按有序输入整批复用；未命中时由调用方决定原有分组，不改变重复行或批边界。"""
    key = batch_cache_key(namespace, raw)
    target = folder / (('training-' if training else 'evaluation-') + key.hexdigest() + '.npz')
    shape = (len(raw),) if width is None else (len(raw), width)
    max_size = 1024 + 8 * len(raw) * (width or 1)
    try:
        # 跨轮可选状态仅接受本实现生成的固定大小、未压缩 NPY 数组。
        from zipfile import ZipFile
        with ZipFile(target) as archive:
            fields = archive.infolist()
            if (len(fields) != 2 or {f.filename for f in fields} != {'values.npy','checksum.npy'}
                    or any(f.compress_type or f.file_size > max_size for f in fields)):
                raise ValueError('训练预计算文件结构不符')
        with np.load(target, allow_pickle=False) as stored:
            values = stored['values']
            valid_dtype = values.dtype == np.float64 or (width is not None and values.dtype == np.float32)
            if (valid_dtype and values.shape == shape
                    and np.isfinite(values).all() and np.all(values >= 0)
                    and str(stored['checksum']) == sha256(key.digest() + values.tobytes()).hexdigest()):
                return values, True
    except (OSError, KeyError, ValueError, EOFError, BadZipFile, ZlibError, RuntimeError):
        pass

    values = compute(raw)
    pending = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=folder, suffix='.tmp', delete=False) as stream:
            pending = stream.name
            np.savez(stream, values=values, checksum=sha256(key.digest() + values.tobytes()).hexdigest())
        os.replace(pending, target)
    except OSError:
        # 临时目录不可写时，沿用本次真实计算结果。
        pass
    finally:
        if pending is not None:
            try:
                os.unlink(pending)
            except OSError:
                pass
    return values, False


def prefetch_teacher(cache, raw, predict, *, width=None, nonnegative=False, batch_size=32):
    """按首次出现顺序计算缺失行；重复、重排和已有缓存不改变输出对应关系。"""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('教师预测输入须为有限二维数组')
    missing = {}
    for row in raw:
        key = row.tobytes()
        if key not in cache:
            missing.setdefault(key, row)
    keys = list(missing)
    for start in range(0, len(keys), batch_size):
        part = keys[start:start+batch_size]
        values = np.asarray(predict(np.asarray([missing[key] for key in part], dtype=np.float32)))
        shape = (len(part),) if width is None else (len(part), width)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('教师批量输出的形状或数值无效')
        if nonnegative:
            values = np.maximum(0., values)
        for key, value in zip(part, values, strict=True):
            cache[key] = value.copy() if width is not None else float(value)


def reuse_frozen_runtime(target, source):
    """保存重载后仅复用逐数组相同教师的内存状态，父网络仍使用重载后的权重。"""
    if type(target) is not type(source) or getattr(source, '_teacher', None) is None:
        return False
    left, right = target.auxiliary_arrays, source.auxiliary_arrays
    if left.keys() != right.keys() or any(not np.array_equal(left[key], right[key]) for key in left):
        return False
    target._teacher = source._teacher
    target._prediction_cache = dict(source._prediction_cache)
    return True
