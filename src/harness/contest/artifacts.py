"""SDK候选只携带数值模型及必要依据，固定运行库由镜像提供。"""
from hashlib import sha256
import json
from pathlib import Path
import shutil

from harness.adapters.storage.local import atomic_json

MAX_FILES = 64
MAX_BYTES = 512 * 1024**2
PREDICT_ARGV = ['python3', '/app/harness.py', 'predict']


def check_files(directory):
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('SDK产物根目录必须是普通目录')
    root = root.resolve()
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('SDK产物不能包含链接')
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if (len(relative) > 180 or len(Path(relative).parts) > 8
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-' for c in relative)):
                raise ValueError('SDK产物路径不合规：' + relative)
            files.append(path)
        elif not path.is_dir():
            raise ValueError('SDK产物只能包含普通文件和目录')
    total = sum(path.stat().st_size for path in files)
    if len(files) > MAX_FILES or total > MAX_BYTES:
        raise ValueError('SDK产物超过64文件或512MiB')
    return {'file_count': len(files), 'bytes': total}


def pack_candidate(checkpoint, destination, *, parent_sha256, training_report=None):
    """逐字节复制已生成的检查点，不重训、不重编码；发布运行代码来自固定镜像。"""
    checkpoint, destination = Path(checkpoint), Path(destination)
    payload = checkpoint.read_bytes()
    if len(payload) > MAX_BYTES:
        raise ValueError('检查点已超过SDK产物总量')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'checkpoint.npz').write_bytes(payload)
    identity = sha256(payload).hexdigest()
    atomic_json(destination/'provenance.json', {
        'format': 'harness.sdk-artifact.v1', 'checkpoint_sha256': identity,
        'parent_checkpoint_sha256': parent_sha256,
        'runtime': 'fixed-image', 'predict_argv': PREDICT_ARGV,
        'training': training_report or {},
    })
    # 只带实际采用程序和许可；大段运行日志及第三方库不随每轮模型重复提交。
    for name in ('training-program.py', 'TabPFN-model-LICENSE'):
        path = checkpoint.parent/name
        if path.is_file():
            shutil.copyfile(path, destination/name)
    return {**check_files(destination), 'checkpoint_sha256': identity,
            'checkpoint_paths': ['checkpoint.npz'],
            'code_paths': ['training-program.py'] if (destination/'training-program.py').is_file() else []}
