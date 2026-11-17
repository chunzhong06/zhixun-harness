"""在隔离构建层中裁去桌面绘图及离线模型下载依赖，保留训练、数值库和云端记录 SDK。"""
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path
import json
import shutil

site = Path("/usr/local/lib/python3.13/site-packages")
report = Path("/app/author-runtime/skrub/_reporting/_table_report.py")
assert sha256(report.read_bytes()).hexdigest() == "c0abf4add19b90e47753194f58b7e65a01dfd2182970e8d66853f32ed0626d74"
source = report.read_text()
statement = "from ._summarize import summarize_dataframe"
call = "        return summarize_dataframe("
assert source.count(statement) == source.count(call) == 1
report.write_text(source.replace(statement + "\n", "").replace(call, "        " + statement + "\n" + call))

# 依照 wheel 的 RECORD 删除本发行包独占的文件。保留其他包和各自的许可证。
removed = {}
for name in ("matplotlib", "fonttools", "pillow", "contourpy", "kiwisolver", "hf-xet"):
    wheel = distribution(name)
    paths = [Path(wheel.locate_file(rel)) for rel in wheel.files]
    size = 0
    dirs = set()
    for path in paths:
        path = Path(path.absolute()).resolve()
        if not path.is_relative_to(site.parent.parent.parent) or not path.is_file():
            continue
        if path.is_relative_to(site) or path.parent == Path("/usr/local/bin"):
            size += path.stat().st_size
            path.unlink()
            dirs.update(p for p in path.parents if p.is_relative_to(site) and p != site)
    for directory in sorted(dirs, key=lambda p: -len(p.parts)):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    removed[name] = size
for name in ("protoc", "protoc-3.13.0.0"):
    path = site / "torch/bin" / name
    path.unlink()
print(json.dumps(removed, sort_keys=True))
