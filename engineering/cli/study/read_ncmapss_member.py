"""从 NASA 的嵌套 ZIP 流式还原第一份 HDF5，校对归档 CRC 后才发布原件。"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
import struct
import zlib

import requests


ROOT = Path(__file__).resolve().parents[3]
URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
ETAG = '"bc0e65c8560b05cec3f88f964c32c403-918"'
OUTER_SIZE = 15760443389
MEMBER = b"data_set/N-CMAPSS_DS01-005.h5"
DEST = ROOT/"data/public/ncmapss/N-CMAPSS_DS01-005.h5"
MEMBER_SHA = "3abdcee2a3c8136e8880e5d43ac8cdf12ab9f7f3ab37145f23b24c11335afbed"
STEP = 8 * 1024 * 1024


def block(start):
    end = start+STEP-1
    for attempt in range(4):
        try:
            res = requests.get(URL, headers={"Range": f"bytes={start}-{end}", "If-Match": ETAG},
                               params={"part": f"nc01-{start}"}, timeout=(12, 55), stream=True)
            with res:
                if (res.status_code != 206 or res.headers.get("Content-Range") != f"bytes {start}-{end}/{OUTER_SIZE}"
                        or res.headers.get("ETag") != ETAG):
                    raise ValueError("NASA 对象或分段版本发生变化")
                data = res.content
                if len(data) != STEP:
                    raise ValueError("NASA 区间未收齐")
                return data
        except requests.RequestException:
            if attempt == 3:
                raise
            sleep(attempt+1)


def local_header(buf, offset):
    header = struct.unpack_from("<4s5H3I2H", buf, offset)
    if header[0] != b"PK\x03\x04":
        raise ValueError("本地 ZIP 头有误")
    size, extra = header[-2:]
    return buf[offset+30:offset+30+size], offset+30+size+extra, header


def fetch():
    if DEST.exists():
        import hashlib
        with DEST.open("rb") as existing:
            if hashlib.file_digest(existing,"sha256").hexdigest() != MEMBER_SHA:
                raise ValueError("缓存的 NASA 原件摘要不符")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"scratch", prefix="ncmapss-ds01-") as temporary, ThreadPoolExecutor(max_workers=5) as pool:
        waiting = deque(pool.submit(block, j*STEP) for j in range(5))
        next_byte = 5*STEP
        outer = zlib.decompressobj(-15)
        hdf = zlib.decompressobj(-15)
        folder = Path(temporary)
        count, consumed, crc = 0, 0, 0
        compressed = expected_crc = raw_size = None
        value = sha256()
        with (folder/"member.h5").open("wb") as out:
            while compressed is None or consumed < compressed:
                data = waiting.popleft().result()
                if compressed is None:
                    _, start, _ = local_header(data,0)
                    name, start, head = local_header(data,start)
                    if not name.endswith(b"/data_set.zip") or head[3] != 8:
                        raise ValueError("NASA 外包布局变化")
                    packed = outer.decompress(data[start:])
                    _, start, _ = local_header(packed,0)
                    name, start, head = local_header(packed,start)
                    if name != MEMBER or head[3] != 8 or head[6:9] != (0xddff9029,1598811454,2873351432):
                        raise ValueError("NASA 首份数据集或 ZIP 元数据变化")
                    expected_crc, compressed, raw_size = head[6:9]
                    packed = packed[start:]
                else:
                    packed = outer.decompress(data)
                packed = packed[:compressed-consumed]
                consumed += len(packed)
                raw = hdf.decompress(packed)
                out.write(raw)
                count += len(raw)
                value.update(raw)
                crc = zlib.crc32(raw,crc)
                if consumed < compressed:
                    waiting.append(pool.submit(block,next_byte))
                    next_byte += STEP
        for pending in waiting:
            pending.cancel()
        if not hdf.eof or count != raw_size or crc != expected_crc or value.hexdigest() != MEMBER_SHA:
            raise ValueError("HDF5 外层校验未通过")
        (folder/"member.h5").replace(DEST)
        print(DEST.name, count, value.hexdigest(), flush=True)


if __name__ == "__main__":
    fetch()
