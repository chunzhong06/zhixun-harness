"""通过标准输入逐瓶索取测量并发布模型；预测进程只读取服务混合物的周期波形。"""
from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.models.calibration_query import RULES as SOURCE_RULES, CalibrationQuery
from harness.models.paired_design import RULES as PAIRED_RULES, PairedDesign, QueryDeployment
from harness.models.year_gas_calibration import GasBase, feature_cycles
from harness.models.year_install_readout import InstallReadout
from harness.models import bottle_alignment, bottle_challenge, bottle_contrast

CHEM = ("acetone","carbonMonoxide","ethanol","ethylAcetate","formaldehyde","hydrogen","toluene","water")
LEGACY = {"按气体与预算固定","按气体与预算论文","单用论文续训","单用成分响应","全曲线回源","目标单训"}
ALIGN_RULES = (bottle_alignment.ACTIVE,bottle_alignment.FROZEN)
BALANCED_RULES = (bottle_challenge.ACTIVE,bottle_challenge.FROZEN)
CONTRAST_RULES = (bottle_contrast.ACTIVE,bottle_contrast.FROZEN)
ALIGN_READOUTS = {bottle_alignment.PRIMARY,bottle_alignment.PAPER,"固定等权波形配准与原装机各半",
                  *bottle_alignment.TOOLS, *(f"原装机与{n}占{w:g}" for n in bottle_alignment.TOOLS for w in (.25,.5,.75))}
BALANCED_READOUTS = {bottle_challenge.PRIMARY,bottle_challenge.PAPER,"对已配准读数只做分界校准",
                     *(f"原装机与分界校准读数占{w:g}" for w in (.25,.5,.75))}
CONTRAST_READOUTS = {bottle_contrast.PRIMARY,bottle_contrast.PAPER,"只用成对标气学习浓度",
                     *(f"原装机与标气对照占{w:g}" for w in (.25,.5,.75))}


def mean_cycles(groups,cycles,ids):
    if (groups.ndim!=1 or groups.dtype.kind not in "iu" or len(groups)!=len(cycles) or len(set(ids))!=len(ids)
            or set(ids)!=set(groups.tolist())):
        raise ValueError("配方分组与周期身份不一致")
    features = feature_cycles(cycles)
    return np.stack([features[groups==uid].mean(0) for uid in ids])


def source_data(path):
    with np.load(path,allow_pickle=False) as source:
        ids = source["ids"]
        if ids.ndim!=1 or ids.dtype.kind not in "iu":
            raise ValueError("标气编号需要一维整数")
        composition = source["composition"]
        if composition.shape!=(len(ids),len(CHEM)) or not np.isfinite(composition).all():
            raise ValueError("标气配方列不完整")
        return ids,mean_cycles(source["group"],source["cycles"],ids),composition


def install(args):
    ids,source,chem = source_data(args.source)
    index = CHEM.index(args.gas)
    base = GasBase().fit(source,chem[:,index])
    if args.rule in CONTRAST_RULES:
        sampler = bottle_contrast.ContrastQuery
    elif args.rule in BALANCED_RULES:
        sampler = bottle_challenge.BalancedQuery
    elif args.rule in ALIGN_RULES:
        sampler = bottle_alignment.AlignmentQuery
    else:
        sampler = PairedDesign if args.rule in PAIRED_RULES else CalibrationQuery
    policy = sampler(source,chem,index,base)
    used, observed = [], []
    for _ in range(args.budget):
        i = policy.next(args.rule,used,observed)
        if i in used or not 0<=i<len(ids):
            raise ValueError("无效或重复的标气请求")
        print(json.dumps({"request":int(ids[i])}),flush=True)
        reply = json.loads(input())
        if set(reply)!={"id","file"} or reply["id"]!=ids[i]:
            raise ValueError("现场返回了未请求的标气")
        with np.load(reply["file"],allow_pickle=False) as record:
            if set(record.files)!={"cycles","id"} or record["id"].item()!=ids[i]:
                raise ValueError("测量文件与标气请求不同")
            observed.append(feature_cycles(record["cycles"]).mean(0))
        used.append(i)
    ix = np.asarray(used)
    tx = np.stack(observed)
    if args.readout in LEGACY or args.readout == "全部周期均值/论文与校正各半":
        model = InstallReadout().fit(base,source,chem,ix,tx,index,args.gas)
    elif args.readout in CONTRAST_READOUTS:
        model = bottle_contrast.ContrastDeployment().fit(source,chem,ix,tx,index)
    elif args.readout in BALANCED_READOUTS:
        model = bottle_challenge.ChallengeDeployment().fit(source,chem,ix,tx,index)
    elif args.readout in ALIGN_READOUTS:
        model = bottle_alignment.AlignmentDeployment().fit(source,chem,ix,tx,index)
    else:
        model = QueryDeployment().fit(source,chem,ix,tx,index)
    model.predict(tx[:1])[args.readout]
    # 原子替换确保训练中断后旧发布版本仍可用。
    args.artifact.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=args.artifact.parent,delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        joblib.dump({"model":model,"method":args.readout,"threshold":float(policy.cut)},temp,compress=3)
        with temp.open("rb") as stream:
            identity = file_digest(stream,"sha256").hexdigest()
        temp.replace(args.artifact)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"sha256":identity,"obtained":ids[ix].tolist(),"threshold":float(policy.cut)}),flush=True)


def predict(args):
    with args.artifact.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=args.sha256:
            raise ValueError("发布身份校验失败")
        f.seek(0)
        release = joblib.load(f)
    with np.load(args.input,allow_pickle=False) as data:
        if set(data.files)!={"cycles","group"}:
            raise ValueError("服务输入只接受读数和分组")
        group = data["group"]
        ids = list(dict.fromkeys(group.tolist()))
        x = mean_cycles(group,data["cycles"],ids)
    values = release["model"].predict(x)[release["method"]]
    classes = 1+(values>release["threshold"])
    print(json.dumps({"ids":ids,"concentration":values.tolist(),"class":classes.tolist()},allow_nan=False))


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="action",required=True)
    setup = cmd.add_parser("install")
    setup.add_argument("--source",type=Path,required=True)
    setup.add_argument("--gas",choices=("acetone","toluene"),required=True)
    setup.add_argument("--rule",choices=(*SOURCE_RULES,*PAIRED_RULES,*ALIGN_RULES,*BALANCED_RULES,*CONTRAST_RULES),required=True)
    setup.add_argument("--budget",type=int,choices=(20,80),required=True)
    setup.add_argument("--readout",required=True)
    setup.add_argument("--artifact",type=Path,required=True)
    serve = cmd.add_parser("predict")
    serve.add_argument("--artifact",type=Path,required=True)
    serve.add_argument("--sha256",required=True)
    serve.add_argument("--input",type=Path,required=True)
    args = cli.parse_args()
    if args.action=="install":
        install(args)
    else:
        predict(args)
