"""从固定 SGP40 原包读取三个独立物理器件；输入和训练目标只按作者的混合物分组。"""
from hashlib import md5, sha256
from pathlib import Path

import numpy as np

from gas_complementary_feedback import ROOT, digest
from score_chongqing_feedback import fixed


SOURCE = ROOT / "data/public/sgp40-transfer/fullData.mat"
MD5 = "6ce3fefe1852219e1a66b3fcc24db045"


def grouped(file, sensor, split):
    targets = file[f"targets_{split}"]
    numbers = np.asarray(targets["range"]).ravel()
    if not np.array_equal(numbers, numbers.astype(int)):
        raise ValueError("混合物原始身份不是整数")
    ids, indices = np.unique(numbers.astype(int), return_inverse=True)
    blocks = []
    for ref in file[f"sensor{sensor}_{split}"][:].ravel():
        signal = np.asarray(file[ref], dtype=float)
        if signal.shape != (1440, len(indices)) or not np.isfinite(signal).all():
            raise ValueError("子传感器未提供完整温度周期")
        blocks.append(signal.T.reshape(len(indices), 12, 120).mean(axis=2))
    x = np.concatenate(blocks, axis=1)
    values = np.asarray([x[indices == j].mean(0) for j in range(len(ids))])
    return ids, values, indices


def target(file, split, indices):
    # 发布说明的总 VOC 估计在原始 MATLAB 结构中登记为 VOCsum。
    voc = np.asarray(file[f"targets_{split}/VOCsum"]).ravel()
    groups = [np.unique(voc[indices == j]) for j in range(int(indices.max()) + 1)]
    if any(len(v) != 1 or not np.isfinite(v[0]) for v in groups):
        raise ValueError("同一种混合物的重复读数未对齐同一浓度")
    return np.array([v[0] for v in groups])


def prepare():
    """一次性从已核对的 HDF5 原包准备重复可用的均值特征；正常复跑只需 numpy。"""
    import h5py
    with SOURCE.open("rb") as fp:
        if md5(fp.read()).hexdigest() != MD5:
            raise ValueError("原包 MD5 与发布者记录不符")
    data = {"md5": np.array(MD5)}
    with h5py.File(SOURCE, "r") as file:
        ids, x, source_groups = grouped(file, "A", "train")
        data.update(A_ids=ids, A_x=x, A_y=target(file, "train", source_groups))
        for sensor in ("B", "C"):
            test_ids, rows, groups = grouped(file, sensor, "test")
            if set(test_ids) & set(ids) or len(ids) != 700 or len(test_ids) != 200:
                raise ValueError("作者训练与测试混合物重叠或分组身份异常")
            data.update({sensor+"_ids": test_ids, sensor+"_x": rows, sensor+"_y": target(file, "test", groups)})
    path = ROOT/"data/prepared/gas/sgp40-mixtures.npz"
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=path.parent) as work:
        stage = Path(work)/path.name
        np.savez_compressed(stage, **data)
        stage.replace(path)


def prepared():
    path = ROOT/"data/prepared/gas/sgp40-mixtures.npz"
    if not path.is_file():
        prepare()
    with np.load(path, allow_pickle=False) as file:
        if file["md5"].item() != MD5:
            raise ValueError("均值特征与登记的原包不同")
        data = {k:file[k] for k in file.files if k != "md5"}
    if len(set(data["A_ids"]) & set(data["B_ids"])) or not np.array_equal(data["B_ids"], data["C_ids"]):
        raise ValueError("源、目标混合物划分或两台目标芯片的配对身份不同")
    return data


def streams():
    data = prepared()
    ids, x = data["A_ids"], data["A_x"]
    threshold = float(np.median(data["A_y"]))
    train_classes = 1 + (data["A_y"] > threshold).astype(int)
    # B 和 C 共用一套在 A 上拟合的模型。源端标准化及训练不使用目标特征。
    outputs = fixed(x, train_classes, np.concatenate([data[k+"_x"] for k in ("B","C")]))
    count = 0
    for sensor in ("B", "C"):
        keys = list(map(str,data[sensor+"_ids"]))
        values = {name: np.asarray(v[count:count+len(keys)]) for name,v in outputs.items()}
        count += len(keys)
        labels = 1 + (data[sensor+"_y"] > threshold).astype(int)
        stream = []
        for ix in np.array_split(np.arange(len(keys)), 8):
            batch_ids = [keys[j] for j in ix]
            p = {k: v[ix] for k,v in values.items()}
            stream.append((batch_ids, p, batch_ids, p, labels[ix]))
        source = {"doi": "10.5281/zenodo.6821340", "file_md5": MD5, "device": sensor,
                  "source_device": "A", "threshold_source_vocsum": threshold,
                  "training_ids": ids.tolist(), "features": x.shape[1], "test_features": digest(data[sensor+"_x"].tolist()),
                  "train_features_sha256": sha256(x.astype("<f8").tobytes()).hexdigest()}
        yield f"sgp40-{sensor}", stream, dict(zip(keys, labels, strict=True)), source
