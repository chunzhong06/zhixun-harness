"""下载并核对Linux/amd64容器依赖，避免构建进程再次下载。"""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
if __name__=='__main__':
    subprocess.run([sys.executable,'-m','pip','download','--dest',str(ROOT/'scratch/contest-adaptation-20261115/wheels'),
        '--only-binary=:all:','--require-hashes','--python-version','313','--implementation','cp','--abi','cp313',
        '--platform','manylinux_2_28_x86_64','--platform','manylinux_2_27_x86_64','--platform','manylinux2014_x86_64','--platform','linux_x86_64',
        '--extra-index-url','https://download.pytorch.org/whl/cpu','-r',str(ROOT/'engineering/configs/environment/requirements-contest-wheels.txt')],check=True)
