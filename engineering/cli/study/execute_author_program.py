"""在限定任务目录执行作者生成程序，阻止读取外层评分文件和启动网络/子进程。"""
from argparse import ArgumentParser
import ast
import os
from pathlib import Path
import random
import runpy
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'engineering/src'))


def main():
    parser=ArgumentParser();parser.add_argument('--script',required=True,type=Path)
    args=parser.parse_args();work=Path.cwd().resolve();script=args.script.resolve()
    if not script.is_relative_to(work):raise ValueError('生成程序不在本次任务目录')
    source=script.read_text(encoding='utf-8');tree=ast.parse(source)
    forbidden={'ctypes','subprocess','socket','requests','urllib','http','multiprocessing','importlib','inspect','google','sys'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
            if any(n.split('.')[0] in forbidden for n in names):raise ValueError('任务程序不需要该外部能力：'+','.join(names))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'eval','exec','compile','__import__'}:
            if (node.func.id=='__import__' and len(node.args)==1 and not node.keywords
                    and isinstance(node.args[0],ast.Constant) and node.args[0].value=='numpy'):
                continue
            raise ValueError('任务程序禁止动态执行')
    import numpy as np
    import pandas
    import cloudpickle
    from harness.integrations import author_task_runtime
    author_task_runtime.bind_training_access(work/'input')
    # 直接提供固定任务接口；模型代码不能通过项目根目录导入其他研究数据。
    sys.modules['harness_task']=author_task_runtime
    random.seed(42);np.random.seed(42)
    readable=[work,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),ROOT/'engineering/src']
    def guard(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(values[0])).resolve();mode=values[1];flags=values[2] if len(values)>2 else 0
            writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            if writing and not path.is_relative_to(work):raise PermissionError('任务输出须留在当前目录')
            if not writing and not any(path.is_relative_to(p) for p in readable):raise PermissionError('任务程序不能读取外层评分或其他任务文件')
        # sklearn/threadpoolctl 正常加载 OpenMP/BLAS 时使用 ctypes；只校验库路径，不能一概拦截。
        if event=='ctypes.dlopen' and values and values[0] is not None:
            library=str(values[0]);path=Path(library)
            system=Path(os.environ.get('SystemRoot','C:/Windows')).resolve()/'System32'
            if path.is_absolute() and not any(path.resolve().is_relative_to(p) for p in readable+[system]):
                raise PermissionError('任务程序不能加载环境外的动态库')
            if not path.is_absolute() and library.lower().removesuffix('.dll') not in {'kernel32','ntdll','msvcrt','psapi','advapi32'}:
                raise PermissionError('未绑定身份的动态库')
        if event.startswith(('socket.','subprocess.')) or event in {'os.system','os.exec','os.posix_spawn','os.fork'}:
            raise PermissionError('任务程序禁止网络和额外进程：'+event)
        if event in {'os.remove','os.rmdir','shutil.rmtree','os.rename'}:
            for value in values[:2] if event=='os.rename' else values[:1]:
                if isinstance(value,(str,bytes,os.PathLike)) and not Path(os.fsdecode(value)).resolve().is_relative_to(work):
                    raise PermissionError('任务程序不能改动当前目录外的文件')
    sys.addaudithook(guard)
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
