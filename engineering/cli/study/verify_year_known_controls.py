"""在目标气体与装机预算已知时，重发两种最强固定参照。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from verify_mixture_response import DEST as PRIOR, LocalStore, run

DEST = PRIOR.parent / "year-known-controls"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("同协议已经交付")
        run(Path(temp),DEST,metadata_policy=True)
