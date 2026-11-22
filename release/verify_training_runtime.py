"""在最终镜像内核验训练所需模块从交付源码树加载，而非从基镜像的旧安装回退。"""
from pathlib import Path
from hashlib import sha256
import json
import sys
import sysconfig
import tomllib
from importlib.metadata import distributions
from inspect import unwrap


def main():
    app = Path('/app')
    # 与 /app/harness.py describe|train 使用相同的 SDK、模块选择和启动目录。
    sys.path.insert(0, str(app))
    from contest_harness.cli import load_submission
    submission = load_submission()
    from harness import bootstrap, paths
    from harness.adapters.datasets import baseline_sources
    from harness.contest import training as platform_train, gateway as sdk_gateway, submission as sdk_submission
    from harness.adapters.platforms.baselines import core_model, data
    from harness.adapters.providers import service_research_broker

    source = (app/'src/harness').resolve()
    expected = json.loads(Path('/opt/training-sources.json').read_bytes())
    inventory = set(paths.package_sources())
    if unwrap(submission.train) is not sdk_submission.train or paths.PACKAGE_ROOT != source:
        raise RuntimeError('SDK训练入口与镜像源码树不一致')
    for module in (bootstrap, baseline_sources, sdk_submission, sdk_gateway, platform_train,
                   service_research_broker, core_model, data):
        file = Path(module.__file__).resolve()
        if not file.is_relative_to(source) or file not in inventory or file.relative_to(source).as_posix() not in expected:
            raise RuntimeError(f'训练源码清单缺失或引用了基镜像旧副本：{module.__name__} ({file})')
    installed = Path(sysconfig.get_path('purelib'))/'harness'
    # 从安装目录单独读取元数据，避免源码目录的 egg-info 掩盖底包遗留的旧版本。
    versions = [d.version for d in distributions(path=[str(installed.parent)])
                if d.metadata['Name'] == 'zhixun-harness']
    if versions != [tomllib.loads((app/'pyproject.toml').read_text())['project']['version']]:
        raise RuntimeError(f'安装目录的项目版本与发行配方不一致：{versions}')
    for relative, digest in expected.items():
        for root in (source, installed):
            file = root/relative
            if not file.is_file() or sha256(file.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'容器源码与构建输入不一致：{file}')
    if not callable(bootstrap.run_prepared):
        raise RuntimeError('缺少使用平台挂载数据的训练入口')
    print(f'workspace.v2 training imports verified: {len(inventory)} source files')


if __name__ == '__main__':
    main()
