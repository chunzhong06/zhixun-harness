"""研究命令参数解析；执行和装配由公共内核拥有。"""
from argparse import ArgumentParser
from pathlib import Path
import asyncio,json
from harness.bootstrap import run,current_parent,ROOT

def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['gas', 'bearing', 'engine'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--driver', choices=['queue', 'service'], default='queue')
    parser.add_argument('--calls', type=int, default=4)
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/official')
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument('--parent-checkpoint', type=Path, help='本轮实际父检查点，平台指定父版优先使用此项')
    parent_group.add_argument('--current', action='store_true', help='接续本项目当前已核验的完整模型')
    parser.add_argument('--no-candidate-cache', action='store_true', help='每次参考动作均启动训练，不复用相同确定性候选')
    parser.add_argument('--recording', choices=['local','online'], default='local', help='online在运行结束后直接上传并核验逐轮过程')
    parser.add_argument('--recording-settings',type=Path,default=ROOT/'engineering/configs/tracking/swanlab.json')
    args = parser.parse_args(argv)
    if not 1 <= args.calls <= 16 or not 1 <= args.stages <= 10:
        raise ValueError('每阶段调用为1至16次，阶段数为1至10')
    result = asyncio.run(run(args.task, args.output, driver=args.driver, calls=args.calls,
                             stages=args.stages, model=args.model, resume=args.resume,
                             data_root=args.data_root, parent_checkpoint=current_parent(args.task) if args.current else args.parent_checkpoint,
                             candidate_cache=not args.no_candidate_cache))
    if args.recording=='online':
        from harness.adapters.tracking.native_recording import publish_completed_run
        recording=publish_completed_run(args.output,settings_path=args.recording_settings,allow_network=True)
        print(json.dumps({'recording':recording},ensure_ascii=False),flush=True)
        if not recording['cloud_verified']:return 3
    return 0 if result['driver_completed_stages'] == args.stages else 2


if __name__ == '__main__':
    raise SystemExit(main())
