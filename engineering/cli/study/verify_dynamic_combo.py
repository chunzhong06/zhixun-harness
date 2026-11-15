"""把新赢得开发比较的差分和储备池概率直接组合接回原始服务入口。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import file_digest
from zipfile import ZipFile
import json
from verify_dynamic_gas_service import RAW,EXPECTED,LocalStore,atomic_json,issue
from score_gas_transitions import DEST as SCORED

DEST=SCORED.parent/"dynamic-combo-entry"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("该组合已部署核对")
        with RAW.open("rb") as f:
            if file_digest(f,"sha256").hexdigest()!=EXPECTED:raise ValueError("原始连续数据变化")
        meta=json.loads((SCORED/"scores.json").read_bytes())
        with ZipFile(RAW) as z:
            results={s:issue(s,t,z,Path(temp),meta[s],SCORED,DEST,True) for s,t in (("co","methane"),("methane","co"))}
        atomic_json(DEST/"scores.json",results)
        print(json.dumps({s+"部署":{m+"F1":v["F1"] for m,v in r["scores"].items()} for s,r in results.items()},ensure_ascii=False))
