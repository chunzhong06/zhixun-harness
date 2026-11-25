"""轴承已拟合教师的发行契约。镜像提供只读状态，现场发放决定它是否适用。"""
from hashlib import file_digest, sha256
import json
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile

DIRECTORY = Path("/opt/zhixun/bearing-fit-v1")
ARCHIVE = "source.tabpfn_fit"
WEIGHTS = "tabpfn-v3.5-fast-20260909.safetensors"


def identity(teacher):
    # 对身份有影响的是实际数值、特征顺序及拟合参数。JSONL 的空白和附加字段不参与预测。
    fields = ("asset_sha256","source_x_sha256","source_y_sha256","feature_paths","fit_mode",
              "inference_precision","random_state","n_estimators")
    data = {k:teacher[k] for k in fields}
    return sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def manifest():
    path = DIRECTORY/"manifest.json"
    return json.loads(path.read_bytes()) if path.is_file() else None


def issued_indices(train, spec):
    """只有同一批已获发的源特征与标签能授权使用拟合状态；可接受重排和追加数据。"""
    lookup = {key:i for i,key in enumerate(train["ids"])}
    try:
        indices = [lookup[key] for key in spec["source_ids"]]
    except KeyError:
        return None
    for field in ("x","y"):
        if sha256(train[field][indices].tobytes()).hexdigest() != spec["source_"+field+"_sha256"]:
            return None
    return indices


def load(teacher, package, weights):
    """将作者拟合文件绑定到本进程已核验的权重，由作者接口恢复模型。"""
    record = manifest()
    if record is None or record["teacher"] != identity(teacher):
        return None
    # 作者格式把权重路径记在 init_params.json，只重绑该路径；执行状态逐字节复制。
    # 两个临时文件均由调用方的私有目录管理，不修改当前目录、共享缓存或作者库。
    bound = weights.with_suffix(".tabpfn_fit")
    with (DIRECTORY/ARCHIVE).open("rb") as stream:
        if file_digest(stream,"sha256").hexdigest() != record["files"][ARCHIVE]:
            raise ValueError("只读轴承拟合产物的摘要不符")
        stream.seek(0)
        with ZipFile(stream) as source, ZipFile(bound,"x") as output:
            params = json.loads(source.read("init_params.json"))
            params["model_path"] = str(weights.resolve())
            output.writestr("init_params.json",json.dumps(params))
            for name in ("fitted_attrs.joblib","executor_state.joblib"):
                with source.open(name) as reader, output.open(name,"w") as writer:
                    copyfileobj(reader,writer,1024*1024)
    return package.TabPFNRegressor.load_from_fit_state(bound,device="cpu")
