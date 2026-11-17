"""将已核NF4/混合/SINQ存储恢复为作者float32检查点；不执行低比特矩阵计算。

NF4码本与研究编码器一致，来源为bitsandbytes get_4bit_type('nf4')。
SINQ的1D位布局与运算顺序沿作者huawei-csl/SINQ@42ca83e（Apache-2.0）。
编码器留在研究目录；此独立读取器仅依赖NumPy和随发布提供的safetensors。
"""
import io
import json
import zipfile
from hashlib import sha256
import numpy as np

CODEBOOK=np.asarray([-1.,-.6961928009986877,-.5250730514526367,-.39491748809814453,
    -.28444138169288635,-.18477343022823334,-.09105003625154495,0.,.07958029955625534,
    .16093020141124725,.24611230194568634,.33791524171829224,.44070982933044434,
    .5626170039176941,.7229568362236023,1.],dtype=np.float32)


def teacher_content_sha256(path):
    """核对张量及元数据内容；容器头中映射的序列化顺序不改变模型身份。"""
    from safetensors import safe_open
    digest = sha256()
    with safe_open(path, framework='np') as tensors:
        digest.update(json.dumps(tensors.metadata() or {}, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
        for name in sorted(tensors.keys()):
            value = tensors.get_tensor(name)
            digest.update(json.dumps([name, value.dtype.str, list(value.shape)],
                separators=(',', ':')).encode('utf-8'))
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def restore_teacher(payload,destination):
    """发布中存储压缩字节；只在任务临时目录生成作者加载所需文件。"""
    from safetensors.numpy import save_file
    raw=np.asarray(payload)
    if raw.dtype!=np.uint8 or raw.ndim!=1 or len(raw)>54*1024**2:
        raise ValueError('教师存储字节无效或超出资产预算')
    packed=raw.tobytes()
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        names=[entry.filename for entry in archive.infolist()]
        if len(names)!=len(set(names)) or sum(entry.file_size for entry in archive.infolist())>128*1024**2:
            raise ValueError('教师压缩目录无效或展开存储过大')
    weights={}
    with np.load(io.BytesIO(packed),allow_pickle=False) as archive:
        keys=[key for key in ('sinq_state','mixed_state','nf4_state') if key in archive.files]
        if len(keys)!=1:raise ValueError('教师数值存储描述必须唯一')
        key=keys[0]
        state=json.loads(str(archive[key]))
        sinq=state.get('format')=='harness.sinq-storage.v1'
        if not sinq and (state.get('format') not in ('harness.nf4-storage.v1','harness.mixed-nf4-storage.v1') or state.get('block_size')!=64):
            raise ValueError('未知教师数值存储格式')
        for name,item in state['tensors'].items():
            shape=tuple(item['shape']);mode=item['mode']
            if mode=='sinq' and sinq:
                codes=archive[item['codes']]
                if codes.dtype!=np.uint8 or codes.ndim!=2 or len(shape)!=2:
                    raise ValueError('SINQ代码或张量维度不符')
                if item['nbits']==4:
                    unpacked=np.concatenate((codes>>4,codes&15),axis=0).astype(np.float32)
                elif item['nbits']==8:
                    unpacked=codes.astype(np.float32)
                else:raise ValueError('未支持的SINQ位宽')
                scale,zero,scale2=(archive[item[k]] for k in ('scale','zero','scale2'))
                if (scale.shape!=(len(unpacked),1) or zero.shape!=scale.shape
                        or scale2.shape!=(1,shape[1]) or unpacked.size!=int(np.prod(shape))):
                    raise ValueError('SINQ分组尺度或元素数量不符')
                value=((unpacked-zero)*scale).reshape(shape)*scale2
            elif mode=='nf4':
                codes=archive[item['codes']];scales=archive[item['scales']]
                indices=np.empty(codes.size*2,dtype=np.uint8)
                indices[0::2]=codes>>4;indices[1::2]=codes&15
                value=(CODEBOOK[indices].reshape(-1,64)*scales.astype(np.float32)[:,None]).ravel()
                value=value[:int(np.prod(shape))].reshape(shape)
            elif mode=='int8_channel':
                value=archive[item['codes']].astype(np.float32)*archive[item['scales']][:,None]
            elif mode=='float16':
                value=archive[item['field']].astype(np.float32)
            elif mode=='raw':
                value=archive[item['field']].copy()
            else:
                raise ValueError('未知教师张量编码')
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('教师张量恢复形状或数值不符')
            weights[name]=value
    save_file(weights,destination,metadata=state['safetensors_metadata'])
