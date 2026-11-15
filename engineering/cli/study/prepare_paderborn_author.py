"""从作者固定提交取得实验实际需要的输入；按 Git blob 摘要验真后发布文件与清单。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from pathlib import Path

import requests

from harness.runtime.storage import atomic_json


ROOT = Path(__file__).resolve().parents[3]
REV = "d7167c927aad5b7f197b1d735b21339e3e17844e"
REPO = "alireza-javanmardi/bearing-RUL"
DATA = ROOT / "data/public/paderborn-phme"
SOURCE = ROOT / "references/implementations" / REPO / REV


def destination(path):
    if path.startswith("data/LDM/"):
        return DATA / path.removeprefix("data/LDM/")
    return SOURCE / path


def acquire():
    listing = requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{REV}",
                           params={"recursive": 1}, timeout=30)
    listing.raise_for_status()
    tree = listing.json()
    if tree["truncated"]:
        raise ValueError("作者仓库目录不完整")
    files = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    paths = ["README.md", "main.py", "src/data.py", "transition_time.py", "data/LDM/README.txt",
             "transition_times/bucket_size_20/seed_0/B17.pkl"]
    for i in range(1, 18):
        paths.extend((f"data/LDM/op/B{i:02}.csv", f"data/LDM/fft/B{i:02}.npy"))

    def copy(path):
        end = destination(path)
        if end.is_file():
            body = end.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{REPO}/{REV}/{path}"
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=(20, 90))
                    response.raise_for_status()
                    body = response.content
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
        blob = sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if blob != files[path]["sha"] or len(body) != files[path]["size"]:
            raise ValueError(f"文件与作者版本不符 {path}")
        if not end.exists():
            end.parent.mkdir(parents=True, exist_ok=True)
            temporary = end.with_suffix(end.suffix + ".part")
            temporary.write_bytes(body)
            temporary.replace(end)
        return {"path": path, "git_blob": blob, "sha256": sha256(body).hexdigest(), "bytes": len(body)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(copy, paths))
    atomic_json(DATA / "manifest.json", {"repository": REPO, "revision": REV, "files": rows})
    print(f"作者输入完整，共 {sum(row['bytes'] for row in rows)} 字节。")


if __name__ == "__main__":
    acquire()
