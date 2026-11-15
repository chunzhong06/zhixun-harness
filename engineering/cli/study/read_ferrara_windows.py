"""从 Ferrara 作者固实 7z 中只读取目录及首批完整波形，核对档内 CRC 后提取径向特征。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import lzma
import re
import struct
import zlib

import numpy as np
import requests
from scipy.io import loadmat

from compare_group_objectives import ROOT, atomic_json, read
from unseen_sources import channel


FOLDER = ROOT/"data/public/ferrara-rtf"
ARCHIVES = ("htk59pp5wx", "zz8hpyx939")


def uint7(data, pos):
    """7z 变长无符号数，处理超过四字节的包偏移。"""
    first = data[pos]
    size = 0
    while size < 8 and first & (0x80 >> size):
        size += 1
    end = pos+1+size
    if end > len(data):
        raise ValueError("归档目录整数不完整")
    return int.from_bytes(data[pos+1:end],"little") + ((first & (127 >> size)) << (8*size)), end


class RemoteArchive:
    def __init__(self, client, file):
        self.client = client
        self.url = file["content_details"]["download_url"]
        self.size = file["size"]
        self.bytes_read = 0
        self.etag = None

    def section(self, start, size):
        end = start + size - 1
        for attempt in range(3):
            try:
                r = self.client.get(self.url, params={"section":f"{start}-{end}"}, headers={
                    "Range":f"bytes={start}-{end}", "Accept-Encoding":"identity"}, timeout=(15,90))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{self.size}" or len(r.content)!=size:
                    raise ValueError("作者归档返回的字节范围不符")
                if self.etag is not None and r.headers.get("ETag")!=self.etag:
                    raise ValueError("一次读取中源对象发生变化")
                self.etag = r.headers.get("ETag")
                self.bytes_read += size
                return r.content
            except requests.RequestException:
                if attempt == 2:
                    raise

    def directory(self):
        head = self.section(0,32)
        if head[:6]!=bytes.fromhex("377abcaf271c") or zlib.crc32(head[12:])!=int.from_bytes(head[8:12],"little"):
            raise ValueError("7z 固定头校验失败")
        offset, size, crc = struct.unpack_from("<QQI",head,12)
        encoded = self.section(32+offset,size)
        if zlib.crc32(encoded)!=crc or encoded[:2]!=b"\x17\x06":
            raise ValueError("压缩目录与固定头不符")
        location,pos = uint7(encoded,2)
        if encoded[pos:pos+2]!=b"\x01\x09":
            raise ValueError("预期单一压缩目录")
        count,_ = uint7(encoded,pos+2)
        key = encoded.index(bytes.fromhex("23030101055d"))+6
        size_dictionary = struct.unpack_from("<I",encoded,key)[0]
        outsize, pos = uint7(encoded,encoded.index(b"\x0c",key+4)+1)
        if encoded[pos:pos+2] != b"\x0a\x01":
            raise ValueError("目录缺少解压 CRC")
        directory_crc = struct.unpack_from("<I",encoded,pos+2)[0]
        dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
            {"id":lzma.FILTER_LZMA1,"dict_size":size_dictionary,"lc":3,"lp":0,"pb":2}])
        raw = dec.decompress(self.section(32+location,count),max_length=outsize)
        if len(raw)!=outsize or zlib.crc32(raw)!=directory_crc:
            raise ValueError("解压后目录未通过作者档内校验")
        return raw


def members(data, device):
    # 作者的 E1 归档分为两个固实块，各块的末文件尺寸由该块总长减去前面文件的长度得到。
    matches = []
    for start in (0,1):
        matches.append(re.findall(device+r"/"+device+r"_(\d+)\.mat\x00",data[start:].decode("utf-16-le",errors="ignore")))
    names = [int(i) for i in max(matches,key=len)]
    if sorted(names)!=list(range(1,len(names)+1)) or len(names)<16:
        raise ValueError("原始测量序号缺失或重复")
    pos=data.index(bytes.fromhex("00070b"))+3
    blocks,pos=uint7(data,pos)
    if not 1<=blocks<=2 or data[pos]!=0:
        raise ValueError("未知的固实块数或外部编码器")
    pos+=1
    properties=[]
    for _ in range(blocks):
        if data[pos:pos+4]!=bytes.fromhex("01212101") or data[pos+4]>=40:
            raise ValueError("不支持该固实波形编码")
        properties.append(data[pos+4])
        pos+=5
    prop=properties[0]
    dictionary = (2 | (prop & 1)) << (prop//2+11)
    if data[pos]!=12:
        raise ValueError("固实块缺少解压长度")
    pos+=1
    totals=[]
    for _ in range(blocks):
        size,pos=uint7(data,pos)
        totals.append(size)
    if data[pos:pos+3]!=b"\x00\x08\x0d":
        raise ValueError("不支持该子文件目录")
    pos+=3
    counts=[]
    for _ in range(blocks):
        size,pos=uint7(data,pos)
        counts.append(size)
    n=sum(counts)
    if n != len(names) or data[pos]!=9:
        raise ValueError("波形文件数与归档流数量不符")
    lengths = []
    pos+=1
    for total,count in zip(totals,counts,strict=True):
        block=[]
        for _ in range(count-1):
            size,pos = uint7(data,pos)
            block.append(size)
        lengths.extend([*block,total-sum(block)])
    if data[pos:pos+2]!=b"\x0a\x01":
        raise ValueError("子文件独立校验未齐备")
    crcs = struct.unpack_from("<"+"I"*n,data,pos+2)
    return names,lengths,crcs,dictionary


def metadata(project,version,device,client):
    res = client.get(f"https://data.mendeley.com/public-api/datasets/{project}/files",
                     params={"version":version,"folder_id":"root"},timeout=30)
    res.raise_for_status()
    file, = [f for f in res.json() if f["filename"]==device+".7z" and f["status"]=="COMPLETED"]
    return file


def collect(device):
    dest=FOLDER/(device+".json")
    if dest.exists():
        record=read(dest)
        if record["device"]!=device or len(record["rows"])!=16:
            raise ValueError("已保存的测量记录不完整")
        return record
    project=ARCHIVES[0 if device in ("E1","E2","E3") else 1]
    with requests.Session() as client:
        info=metadata(project,1,device,client)
        remote=RemoteArchive(client,info)
        directory=remote.directory()
        names,sizes,crcs,dictionary=members(directory,device)
        second=RemoteArchive(client,metadata(project,2,device,client))
        names_v2,_,_,_=members(second.directory(),device)
        if sorted(names)!=sorted(names_v2):
            raise ValueError("两版档案的测量身份不一致")
        decoder=lzma.LZMADecompressor(lzma.FORMAT_RAW,filters=[{"id":lzma.FILTER_LZMA2,"dict_size":dictionary}])
        needed=sum(sizes[:16])
        # 该源是单一固实块，只读预定物理前缀；每个完整 MAT 均需独立通过 CRC。
        position=32
        buffer=bytearray()
        while len(buffer)<needed:
            request=min(8*1024**2,max(1024**2,needed-len(buffer)))
            if position>32+2*needed+1024**2:
                raise ValueError("压缩前缀超出固定的小样本预算")
            buffer.extend(decoder.decompress(remote.section(position,request)))
            position+=request
        rows=[]
        start=0
        for index, size, crc in zip(names[:16],sizes[:16],crcs[:16],strict=True):
            body=buffer[start:start+size]
            start+=size
            if zlib.crc32(body)!=crc:
                raise ValueError("抽取的完整 MATLAB 测量 CRC 错误")
            arrays=loadmat(BytesIO(body))
            items=[np.asarray(v).ravel() for k,v in arrays.items() if not k.startswith("__")]
            signal,=[a for a in items if a.size==5*25600]
            rate,=[a.item() for a in items if a.size==1]
            if rate!=25600 or not np.isfinite(signal).all():
                raise ValueError("波形尺寸、采样频率或取值不符")
            rows.append({"index":index,"features":list(channel(signal[:32768],sample_rate=rate)),
                         "sha256_mat":sha256(body).hexdigest(),"crc32_mat":crc})
    report={"project":project,"device":device,"version":1,"source":remote.url,"size":remote.size,
            "publisher_archive_sha256":info["content_details"]["sha256_hash"],"directory_sha256":sha256(directory).hexdigest(),
            "terminal_index":max(names),"bytes_received":remote.bytes_read+second.bytes_read,"rows":rows}
    atomic_json(dest,report)
    print(device,len(names),[r["index"] for r in rows],flush=True)
    return report


if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(collect,[f"E{i}" for i in range(1,7)]))
