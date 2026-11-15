"""仅上传或恢复一个已完成原生运行的记录，不运行训练或模型推理。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'engineering/src'))
from harness.integrations.native_recording import DEFAULT_SETTINGS,publish_completed_run


def main(argv=None):
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--online',action='store_true',required=True,help='使用运行时SDK凭据连接已配置SwanLab项目')
    ap.add_argument('--settings',type=Path,default=DEFAULT_SETTINGS)
    args=ap.parse_args(argv)
    result=publish_completed_run(args.run,settings_path=args.settings,allow_network=args.online)
    print(json.dumps({'recording':result},ensure_ascii=False),flush=True)
    return 0 if result['cloud_verified'] else 3


if __name__=='__main__':raise SystemExit(main())
