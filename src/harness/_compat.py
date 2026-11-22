"""为历史模型和脚本解析旧模块名，始终返回实际实现的同一模块对象。"""
from importlib import import_module
from importlib.abc import Loader, PathEntryFinder
from importlib.util import spec_from_loader
from pathlib import Path
import sys

from ._legacy_names import LEGACY_MODULES


class _Aliases(PathEntryFinder, Loader):
    def __init__(self, package):
        self.targets = LEGACY_MODULES[package]

    def find_spec(self, fullname, target=None):
        if fullname.rsplit('.', 1)[-1] in self.targets:
            return spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        sys.modules[name] = import_module(self.targets[name.rsplit('.', 1)[-1]])

    def iter_modules(self, prefix=''):
        for name in sorted(self.targets):
            yield prefix + name, False


def legacy_path(package, paths):
    """只扩展指定包的查找路径，使真实文件和保留的命令行入口优先。"""
    key = str(Path(paths[0]).resolve() / '__legacy_imports__')
    sys.path_importer_cache[key] = _Aliases(package)
    if key not in paths:
        paths.append(key)
