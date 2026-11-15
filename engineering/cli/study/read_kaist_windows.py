"""从作者固定 ZIP 的每小时文件读取首段双轴波形；保留原始切片摘要与可重取的坐标。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from io import BytesIO, RawIOBase
import json
import struct
from time import sleep
from zipfile import ZipFile, ZIP_DEFLATED
import zlib

import numpy as np
import requests
from unseen_sources import ROOT, channel
from harness.runtime.storage import atomic_json

META = "https://data.mendeley.com/public-api/datasets/5hcdd3tdvb/files?folder_id=root&version=6"
URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/ac741b1b-e2cd-43b4-bbaf-f50d98f6395c"
SIZE = 4325053806
ETAG = '"0357c62716c487aea31f2085d4e11562-207"'
SOURCE_SHA = "7deeafed7334ed18b4da552075ef73391c15bc6b206dcdf6c0c7ec3e4e2657c6"
FILE = ROOT/"data/public/kaist-bearing-v6/hourly-windows.json"
COUNT = 32768


def fetch(start, end):
    for attempt in range(3):
        try:
            r = requests.get(URL,headers={"Range":f"bytes={start}-{end}", "If-Match":ETAG,
                                        "Accept-Encoding":"identity"},timeout=40)
            r.raise_for_status()
            if (r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}"
                    or r.headers.get("ETag")!=ETAG or len(r.content)!=end-start+1):
                raise ValueError("来源范围或对象身份不符")
            return r.content
        except requests.RequestException:
            if attempt==2:
                raise
            sleep(1+attempt)


class ZipTail(RawIOBase):
    """ZIP64 目录内的偏移相对于完整档案。仅缓存目录末尾，禁止误读数据区。"""
    def __init__(self):
        self.start, self.pos = SIZE-131072, 0
        self.data = fetch(self.start,SIZE-1)

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0,self.pos,SIZE)[whence]+offset
        return self.pos

    def read(self, size=-1):
        start = self.pos
        if start<self.start:
            raise ValueError("ZIP 目录超出已核对区间")
        self.pos = SIZE if size<0 else min(SIZE,start+size)
        return self.data[start-self.start:self.pos-self.start]


def window(member):
    stamp = datetime.strptime(member.filename,"LogFile_%Y-%m-%d-%H-%M-%S.csv")
    if member.compress_type!=ZIP_DEFLATED or member.flag_bits & 1:
        raise ValueError("未知的源文件编码")
    for count in (900000,1850000,3500000):
        raw = fetch(member.header_offset,member.header_offset+count-1)
        if raw[:4]!=b"PK\x03\x04":
            raise ValueError("源文件头不符")
        name_size, extra = struct.unpack_from("<HH",raw,26)
        if raw[30:30+name_size].decode("ascii")!=member.filename:
            raise ValueError("源文件名不符")
        prefix = zlib.decompressobj(-15).decompress(raw[30+name_size+extra:])
        lines = prefix.split(b"\n",COUNT)
        if len(lines)==COUNT+1:
            break
    else:
        raise ValueError("首段不足 1.28 秒")
    source = b"\n".join(lines[:COUNT])+b"\n"
    xy = np.loadtxt(BytesIO(source),delimiter=",",usecols=(0,1))
    info = {"file":member.filename,"time":stamp.isoformat(),"header_offset":member.header_offset,
            "source_crc32":member.CRC,"window_sha256":sha256(source).hexdigest()}
    if xy.shape!=(COUNT,2):
        raise ValueError(f"{member.filename} 双通道尺寸 {xy.shape}")
    if not np.isfinite(xy).all():
        return {**info,"invalid_nonfinite_values":int(np.sum(~np.isfinite(xy)))}
    values = channel(xy[:,0])+channel(xy[:,1])
    if not np.isfinite(values).all():
        raise ValueError("特征数值无效")
    return {**info,"features":values}


def prepare():
    if FILE.exists():
        return json.loads(FILE.read_bytes())
    r = requests.get(META,timeout=25)
    r.raise_for_status()
    m, = r.json()
    if (m["content_details"]["sha256_hash"]!=SOURCE_SHA or m["size"]!=SIZE
            or m["filename"]!="Vibration_Bearing_RuntoFailure.zip"):
        raise ValueError("数据源与固定版本不符")
    with ZipFile(ZipTail()) as archive:
        members = sorted(archive.infolist(),key=lambda m:m.filename)
    if len(members)!=129 or len({m.filename for m in members})!=129:
        raise ValueError("每小时原始文件不齐全")
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(window,members))
    result = {"doi":"10.17632/5hcdd3tdvb.6","archive_reported_sha256":SOURCE_SHA,"etag":ETAG,
              "sampling_hz":25600,"samples_per_window":COUNT,"unit":"g","rows":rows}
    atomic_json(FILE,result)
    return result


if __name__=="__main__":
    print("KAIST windows",len(prepare()["rows"]))
