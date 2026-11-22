"""定位同一发行版的项目包，再向官方 SDK 提供原生 train 和 predict。"""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import sysconfig

def first_mismatch(root, expected):
    """训练和预测都可能延迟导入模块，启动时检查整个发行清单。"""
    for name, digest in expected.items():
        path = root/'harness'/name
        try:
            if sha256(path.read_bytes()).hexdigest() == digest:
                continue
        except OSError:
            pass
        return path
    return None


app = Path(__file__).resolve().parent
primary = app/'src'
installed = Path(sysconfig.get_path('purelib'))
expected = json.loads(Path('/opt/training-sources.json').read_bytes())
if not {'__init__.py', 'contest/submission.py', 'contest/training.py'}.issubset(expected):
    raise ImportError('镜像源码清单缺少必需模块')
source = primary
missing = first_mismatch(primary, expected)
if missing is not None:
    failed = first_mismatch(installed, expected)
    if failed is not None:
        raise ImportError(f'主源码校验失败：{missing}；安装副本校验失败：{failed}')
    source = installed
    print(f'主源码校验失败：{missing}，使用已核验的训练源码安装副本：{source}/harness', file=sys.stderr)

author = str(app/'author-runtime')
sys.path[:0] = [author, str(source)]
# 数值训练另起 Python 进程，必须继承已经核验的源码副本和同一作者库。
search = os.environ.get('PYTHONPATH', '').split(os.pathsep)
other = [item for item in search if item and item not in {author, str(primary), str(installed)}]
os.environ['PYTHONPATH'] = os.pathsep.join([author, str(source), *other])
if not os.environ.get('HARNESS_TABULAR_RESOURCES'):
    os.environ['HARNESS_TABULAR_RESOURCES'] = str(app/'assets/tabular-resources.json')
# 发行镜像在构建时备齐清单中的权重。SDK 和按模块名加载的宿主使用同一训练函数。
from harness.contest.submission import TASKS, train, predict
