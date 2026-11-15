"""供受控 Jupyter 内核安装的任务边界；评价答案始终留在内核外。

这是可信模型代码的防误读边界，沿用单程序执行器的范围，不声称是 OS 沙箱。
Jupyter 自身的通信及历史机制在用户单元格外运行；单元格只能读合法输入和环境。
"""
import ast
import os
from pathlib import Path
import sys
import threading


def validate_source(source):
    tree = ast.parse(source)
    forbidden = {'ctypes', 'subprocess', 'socket', 'requests', 'urllib', 'http',
                 'multiprocessing', 'importlib', 'inspect', 'google', 'sys'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names]
            if any(n.split('.')[0] in forbidden for n in names):
                raise ValueError('任务单元格禁止外部能力：' + ','.join(names))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (node.func.id == '__import__' and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'numpy'):
                # 字面量 NumPy 导入与 import numpy 等价，不提供动态模块选择能力。
                continue
            if node.func.id in {'eval', 'exec', 'compile', '__import__', 'get_ipython'}:
                raise ValueError('任务单元格禁止动态执行或修改内核')


def install(work, source_root):
    """在首个模型单元格前调用一次，IPython 事件控制审计作用域。"""
    from IPython import get_ipython
    import numpy as np
    import pandas
    import cloudpickle
    import random
    from harness.integrations import author_task_runtime
    sys.modules['harness_task'] = author_task_runtime
    work = Path(work).resolve()
    readable = [work, work.parent / 'input', Path(source_root).resolve(),
                Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    state = {'active': False, 'thread': threading.get_ident()}
    random.seed(42)
    np.random.seed(42)

    def guard(event, args):
        if not state['active'] or threading.get_ident() != state['thread']:
            return
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode = args[1]
            flags = args[2] if len(args) > 2 else 0
            writing = ((isinstance(mode, str) and any(x in mode for x in 'wax+')) or
                       (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))))
            if writing and not path.is_relative_to(work):
                raise PermissionError('输出必须留在本次编程目录')
            if not writing and not any(path.is_relative_to(p) for p in readable):
                raise PermissionError('禁止读取评价答案或其他运行资料')
        if event.startswith(('socket.', 'subprocess.')) or event in {'os.system', 'os.exec', 'os.posix_spawn', 'os.fork'}:
            raise PermissionError('模型代码不能联网或启动进程')
        if event in {'os.remove', 'os.rmdir', 'shutil.rmtree', 'os.rename'}:
            for value in args[:2] if event == 'os.rename' else args[:1]:
                if isinstance(value, (str, bytes, os.PathLike)) and not Path(os.fsdecode(value)).resolve().is_relative_to(work):
                    raise PermissionError('模型代码不能改动本次目录外文件')

    def before(info):
        validate_source(info.raw_cell)
        state['active'] = True

    def after(result):
        state['active'] = False

    sys.addaudithook(guard)
    shell = get_ipython()
    shell.events.register('pre_run_cell', before)
    shell.events.register('post_run_cell', after)
