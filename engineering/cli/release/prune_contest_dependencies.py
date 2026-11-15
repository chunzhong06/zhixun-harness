"""仅在临时构建层清理第三方测试与字节码；保留运行源码、库、头文件和许可证。"""
from pathlib import Path
import json
import shutil
import sysconfig


def main():
    root = Path(sysconfig.get_paths()['purelib']).resolve()
    if str(root) != '/usr/local/lib/python3.13/site-packages':
        raise RuntimeError('此脚本只允许在指定的比赛容器构建层运行')
    candidates = set(root.rglob('__pycache__'))
    for package in ('numpy', 'scipy', 'pandas', 'sklearn', 'sympy', 'networkx', 'torch', 'mpmath'):
        candidates.update((root / package).rglob('tests'))
    candidates.add(root / 'torch/test')
    removed = []
    # 父目录优先，避免重复计数；不删除 torch/testing、编译头文件或共享库。
    for path in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
        if not path.exists():
            continue
        path.resolve().relative_to(root)
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f'拒绝非普通目录: {path}')
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        removed.append({'path': str(path.relative_to(root)), 'bytes': size})
        shutil.rmtree(path)
    result = {'removed': removed, 'removed_bytes': sum(r['bytes'] for r in removed),
              'runtime_source_libraries_headers_preserved': True}
    Path('/tmp/contest-dependency-prune.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({'removed_directories': len(removed), 'removed_bytes': result['removed_bytes']}))


if __name__ == '__main__':
    main()
