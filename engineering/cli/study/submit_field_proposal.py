"""从真实请求派生响应身份与路径，避免现场代理手写长摘要造成误投。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.runtime.storage import atomic_json


def main():
    ap=ArgumentParser(description=__doc__);ap.add_argument("--request",type=Path,required=True)
    ap.add_argument("--plan-file",type=Path,required=True)
    ap.add_argument("--check-only",action="store_true",help="只核对路由和已有响应，不写文件")
    args=ap.parse_args();request_path=args.request.resolve();plan_path=args.plan_file.resolve()
    if not request_path.is_relative_to(ROOT) or not plan_path.is_relative_to(ROOT):raise ValueError("请求和方案必须位于当前项目")
    if request_path.parent.parent.name!="requests" or request_path.suffix!=".json":raise ValueError("需使用queue/requests/<任务>内的真实JSON请求")
    request=json.loads(request_path.read_bytes());key=request["request_id"]
    if not re.fullmatch(r"[0-9a-f]{64}",key) or not request_path.stem.endswith('-'+key):
        raise ValueError("请求文件名与原摘要不一致")
    raw=json.loads(plan_path.read_bytes())
    # 可用原误投回复修复传输；只提取已有plan，身份始终从真实请求取得。
    plan=raw["plan"] if isinstance(raw,dict) and "request_id" in raw and "plan" in raw else raw
    if not isinstance(plan,dict):raise ValueError("方案必须是JSON对象")
    destination=request_path.parent.parent.parent/"responses"/request_path.parent.name/request_path.name
    payload={"request_id":key,"plan":plan};exists=destination.is_file()
    if exists and json.loads(destination.read_bytes())!=payload:raise ValueError("已有不同回复，不可覆盖现场决定")
    if not args.check_only and not exists:atomic_json(destination,payload)
    print(json.dumps({"request":str(request_path),"response":str(destination),"already_identical":exists,
                      "written":not args.check_only and not exists,"plan_changed":False},ensure_ascii=False))


if __name__=="__main__":main()
