"""保留已有首案，让现场模型据与总分一致的分组误差重做第二次提案。"""
from pathlib import Path
from argparse import ArgumentParser
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from harness.contracts import digest
from harness.integrations.proposal_inbox import ProposalInbox
from harness.runtime.storage import LocalStore,atomic_json


class FirstProposalReplay(ProposalInbox):
    def __init__(self,folder,original,replay_arms=(),refresh=()):
        super().__init__(folder)
        self.original=original
        self.replay_arms=set(replay_arms)
        self.refresh=set(refresh)

    def send(self,request):
        context=json.loads(request.body["messages"][-1]["content"])
        pair=context["task_id"]+":"+context["site"]
        if (context["batch_id"].endswith("-trial-1") or context["task_id"] in self.replay_arms
                or (self.refresh and pair not in self.refresh)):
            arm,site=context["task_id"],context["site"]
            record=self.original/site/(arm+"-decision.json")
            turn=int(context["batch_id"].rsplit("-",1)[1])
            first=json.loads(record.read_bytes())["trials"][turn-1]["proposal"]
            key=digest(request.body)
            target=self.folder/"responses"/arm/(context["batch_id"]+"-"+key+".json")
            # 指定提案逐字重放原实际Luna输出；未指定组的第二案由现场驱动返回。
            atomic_json(target,{"request_id":key,"plan":first,"replayed_from":str(record.relative_to(ROOT)),"replayed_trial":turn})
        return super().send(request)


def main():
    base=ROOT/"results/experiments/local-defaults/whole-maintenance"
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--original",type=Path,default=base/"london-adaptive-recovery")
    cli.add_argument("--output",type=Path,default=base/"london-consistent-observations")
    cli.add_argument("--replay-arm",action="append",choices=("project","paper"),default=[])
    cli.add_argument("--expand-models",action="store_true")
    cli.add_argument("--refresh",action="append",default=[],help="仅指定组:站点的第二案重新提案；其余原样重放")
    cli.add_argument("--start",default="2024-07-01")
    cli.add_argument("--archive",type=Path,default=ROOT/"data/public/london-air-2024/stations.zip")
    cli.add_argument("--initial-spec",type=Path)
    args=cli.parse_args()
    out=args.output.resolve()
    original=args.original.resolve()
    if any(pair not in {a+":"+s for a in ("project","paper") for s in scene.TARGETS} for pair in args.refresh):
        raise ValueError("重新提案对象需为既有组名和预列站点")
    scene.configure(args.start)
    inbox=FirstProposalReplay(out/"queue",original,args.replay_arm,args.refresh)
    with LocalStore(out).exclusive():
        if any((out/site).exists() for site in scene.TARGETS):
            raise ValueError("不可覆盖已重放现场")
        initial=json.loads(args.initial_spec.read_bytes()) if args.initial_spec else None
        scene.run(args.archive,out,adaptive=True,inbox=inbox,flexible=args.expand_models,initial_spec=initial)


if __name__=="__main__":
    main()
