"""在两轮轴承准备之间传递完整查询批次的冻结教师值。"""
from hashlib import sha256
import os
import shutil

import numpy as np
from .teacher_cache import REGRESSION_QUERY_ROWS as QUERY_ROWS


QUERY_ROWS = 128
FILENAME = "validation-prefix.bin"
MAX_BYTES = 1024 * 1024


def copy_validation_prefix(source, destination):
    """可选状态只包含一个有界文件；缺失或损坏时准备流程照常重算。"""
    path = source / FILENAME
    try:
        if source.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            return False
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination / FILENAME)
        return True
    except OSError:
        return False


def _signature(namespace, raw, keys, values):
    # 同一镜像也可能分配到不同 CPU 或指令后端。发生变化时重新计算。
    import torch
    dispatch = [torch.__version__, torch.backends.cpu.get_cpu_capability()]
    dispatch.extend(os.environ.get(k, "") for k in ("ATEN_CPU_CAPABILITY", "MKL_ENABLE_INSTRUCTIONS",
                                                      "ONEDNN_MAX_CPU_ISA", "MKL_CBWR"))
    digest = sha256("\n".join(["bearing-prefix-v1", namespace, *dispatch]).encode())
    digest.update(np.asarray([raw.shape[1], len(keys), QUERY_ROWS], dtype="<i8").tobytes())
    for key in keys:
        digest.update(key)
    digest.update(values)
    return digest.digest()


def predict_with_prefix(raw, namespace, memory, predict, prior, output):
    """按首次出现顺序对齐教师实际看到的查询块，当前父权重始终重新读出。"""
    if memory:
        # 已有部分行会改变作者模型的查询分组，此时不读取或发布跨轮结果。
        return predict(raw), 0
    keys = list(dict.fromkeys(row.tobytes() for row in raw))
    limit = min(len(keys), (MAX_BYTES - 32) // 8) // QUERY_ROWS * QUERY_ROWS
    keys = keys[:limit]
    restored = 0
    if prior is not None and keys:
        path = prior / FILENAME
        try:
            if not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_BYTES:
                with path.open("rb") as stream:
                    data = stream.read(MAX_BYTES + 1)
                size = len(data) - 32
                if 0 < size <= 8*limit and size % (8*QUERY_ROWS) == 0:
                    block = keys[:size//8]
                    values = memoryview(data)[32:]
                    if _signature(namespace, raw, block, values) == data[:32]:
                        numbers = np.frombuffer(values, dtype="<f8")
                        if np.isfinite(numbers).all() and (numbers >= 0).all():
                            memory.update(zip(block, numbers.tolist(), strict=True))
                            restored = len(block)
        except OSError:
            pass
    predictions = predict(raw)
    if keys:
        values = np.asarray([memory[k] for k in keys], dtype="<f8").tobytes()
        # 本轮独占准备目录。完整校验后才会在后继轮使用中间数值。
        try:
            (output / FILENAME).write_bytes(_signature(namespace, raw, keys, values) + values)
        except OSError:
            pass
    return predictions, restored
