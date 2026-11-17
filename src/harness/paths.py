"""工作区资源定位；容器任务应显式传入资料，包内资源随 wheel 安装。"""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

def workspace_root():
    if os.environ.get('HARNESS_WORKSPACE'):
        return Path(os.environ['HARNESS_WORKSPACE']).resolve()
    for path in PACKAGE_ROOT.parents:
        if (path/'engineering/configs').is_dir():
            return path
    return Path.cwd().resolve()

def package_sources():
    """恢复及缓存绑定当前安装代码，不依赖研究脚本所在路径。"""
    return sorted(PACKAGE_ROOT.rglob('*.py'))
