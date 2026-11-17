"""共享的只读运行资源与单臂执行预算；不负责模型选择或研究流程。"""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


class BudgetExhausted(RuntimeError):
    """执行余额已用完，调用者不得再启动子进程或单元格。"""


@dataclass(frozen=True)
class ExecutionLease:
    timeout_seconds: float
    started_at: float

    @property
    def remaining_seconds(self):
        return max(0., self.timeout_seconds - (time.monotonic() - self.started_at))


class ExecutionBudget:
    """同一实例串行计入实际执行墙钟，排队不计时，失败和超时仍计时。

    宿主须把 lease.timeout_seconds 交给子进程超时或内核终止器；本对象只
    管理锁和账本，不会自行终止计算。不得在已持有 execution 时再次进入。
    """
    def __init__(self, total_seconds):
        total_seconds = float(total_seconds)
        if not math.isfinite(total_seconds) or total_seconds < 0:
            raise ValueError('执行总预算必须是有限非负秒数')
        self.total_seconds = total_seconds
        self._used = 0.
        self._started = None
        self._owner = None
        self._stop_reason = None
        self._serial = threading.Lock()
        self._state = threading.Lock()

    @property
    def used_seconds(self):
        with self._state:
            return self._used + (0. if self._started is None else time.monotonic() - self._started)

    @property
    def remaining_seconds(self):
        return max(0., self.total_seconds - self.used_seconds)

    def cancel(self, reason):
        """内核无法确认停止时关闭后续执行，实际耗时仍按原账本报告。"""
        with self._state:
            self._stop_reason = str(reason)

    @contextmanager
    def execution(self, max_seconds=None):
        if max_seconds is not None and (not math.isfinite(float(max_seconds)) or float(max_seconds) <= 0):
            raise ValueError('单次执行上限必须是有限正秒数')
        with self._state:
            if self._owner == threading.get_ident():
                raise RuntimeError('执行预算不能嵌套；直接复用当前执行许可')
        with self._serial:
            with self._state:
                if self._stop_reason is not None:
                    raise BudgetExhausted(self._stop_reason)
                remaining = self.total_seconds - self._used
                if remaining <= 0:
                    raise BudgetExhausted('本臂执行总预算已耗尽')
                self._started = time.monotonic()
                self._owner = threading.get_ident()
                lease = ExecutionLease(min(remaining, float(max_seconds)) if max_seconds is not None else remaining,
                                       self._started)
            try:
                yield lease
            finally:
                with self._state:
                    self._used += time.monotonic() - self._started
                    self._started = self._owner = None


@dataclass(frozen=True)
class RuntimeAssets:
    python_paths: tuple = ()
    read_only_roots: tuple = ()
    read_only_files: tuple = ()
    offline: bool = False

    def allows_read(self, path):
        path = Path(path).resolve()
        return any(path.is_relative_to(root) for root in self.read_only_roots) or any(
            path == file for file, _ in self.read_only_files)


def _absolute(path, *, directory):
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError('运行资源必须使用绝对路径')
    resolved = Path(path).resolve(strict=True)
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError('运行资源类型不符：' + path)
    return resolved


def _verify_files(assets):
    for path, expected in assets.read_only_files:
        digest = sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('只读运行文件 SHA256 不符：' + str(path))


def load_runtime_assets(manifest=None):
    """宿主加载清单并核对精确文件身份；不自动放行文件父目录。"""
    if manifest is None:
        return RuntimeAssets()
    if isinstance(manifest, RuntimeAssets):
        _verify_files(manifest)
        return manifest
    path = _absolute(os.fspath(manifest), directory=False)
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict) or set(record) != {'python_paths', 'read_only_roots', 'read_only_files', 'offline'}:
        raise ValueError('运行资源清单字段不完整或含未知字段')
    if (not isinstance(record['python_paths'], list) or not isinstance(record['read_only_roots'], list)
            or not isinstance(record['read_only_files'], dict) or type(record['offline']) is not bool):
        raise ValueError('运行资源清单类型不符')
    paths = tuple(_absolute(p, directory=True) for p in record['python_paths'])
    roots = tuple(_absolute(p, directory=True) for p in record['read_only_roots'])
    if any(not any(p.is_relative_to(root) for root in roots) for p in paths):
        raise ValueError('Python 依赖路径必须包含在声明的只读根内')
    files = []
    for name, expected in record['read_only_files'].items():
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('运行文件需要小写 SHA256')
        files.append((_absolute(name, directory=False), expected))
    assets = RuntimeAssets(paths, roots, tuple(files), record['offline'])
    _verify_files(assets)
    return assets


def install_runtime_assets(manifest=None):
    """在安装审计钩子前调用；依赖仅加到当前进程，离线模式禁止自动下载。"""
    assets = load_runtime_assets(manifest)
    for path in reversed(assets.python_paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    if assets.offline:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return assets


def guard_asset_mutation(assets, event, args):
    """补充只读资源的元数据/目录保护；普通输入输出仍由现有边界约束。"""
    if event in {'os.remove', 'os.rmdir', 'shutil.rmtree', 'os.rename', 'os.mkdir', 'os.chmod', 'os.utime'}:
        for value in args[:2] if event == 'os.rename' else args[:1]:
            if isinstance(value, (str, bytes, os.PathLike)) and assets.allows_read(os.fsdecode(value)):
                raise PermissionError('声明的运行资源只允许读取')
