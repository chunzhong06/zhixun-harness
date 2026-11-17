"""统一原包和已部署源码的读取，平台无需携带研究数据 ZIP。"""
from contextlib import contextmanager
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import sys

class DirectorySources:
    def __init__(self, directory): self.directory = Path(directory).resolve()
    def read(self, name):
        relative = Path(name).relative_to('baselines')
        if relative.as_posix() == 'model.py': relative = Path('core_model.py')
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory): raise ValueError('源码路径越界')
        return path.read_bytes()

@contextmanager
def source_package(specification):
    if 'baseline_directory' in specification:
        source = DirectorySources(specification['baseline_directory'])
        for name, expected in specification['baseline_sha256'].items():
            if sha256(source.read(name)).hexdigest() != expected:
                raise ValueError('部署训练或预测源码身份变化：' + name)
        yield source
    else:
        path = Path(specification['archive'])
        if sha256(path.read_bytes()).hexdigest() != specification['archive_sha256']:
            raise ValueError('官方原包已改变')
        with ZipFile(path) as package: yield package

@contextmanager
def trainer_for(package, temporary, profile):
    if not isinstance(package, DirectorySources):
        from harness.learning.models.official_rul_cpu import _trainer
        with _trainer(package, Path(temporary), profile) as trainer: yield trainer
        return
    # core_model 只有 NumPy 依赖；独立模块身份避免连续三任务复用旧类。
    name = '_baseline_' + Path(temporary).name.replace('-', '_')
    spec = spec_from_file_location(name, package.directory/'core_model.py')
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield SimpleNamespace(Model=module.Model)
    finally: sys.modules.pop(name, None)
