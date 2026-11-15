"""验证真实发放端的反馈和较晚到达的测量只能用于下一批。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import file_digest
from zipfile import ZipFile
import json
from verify_dynamic_gas_service import RAW,EXPECTED,LocalStore,atomic_json,issue
from score_delayed_teacher import DEST as SCORED

DEST=SCORED.parent/"delayed-entry"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("延迟训练已完成部署复核")
        with RAW.open("rb") as f:
            if file_digest(f,"sha256").hexdigest()!=EXPECTED:raise ValueError("原始测量变化")
        meta=json.loads((SCORED/"scores.json").read_bytes())
        with ZipFile(RAW) as z:
            result={s:issue(s,t,z,Path(temp),meta[s],SCORED,DEST,True,True) for s,t in (("co","methane"),("methane","co"))}
        atomic_json(DEST/"scores.json",result)
        print(json.dumps({k+"实时部署":{role+"F1":v["F1"] for role,v in r["scores"].items()} for k,r in result.items()},ensure_ascii=False))
