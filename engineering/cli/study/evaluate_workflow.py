"""只读已有运行快照与独立答案，补算终局/分批评价；不创建或调用任何模型。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.data.package_input import read_package
from harness.evaluation.final_answers import prepare_answers, score_final
from harness.evaluation.longitudinal import evaluate_history
from harness.runtime.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state", "manifest", "answers", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="事前阶段与恢复定义")
    args = parser.parse_args()
    protected={p.resolve() for p in (args.state,args.manifest,args.answers,args.evaluation) if p is not None}
    if any((args.output/name).resolve() in protected for name in ("final-evaluation.json","final-longitudinal.json")):
        raise ValueError("评价输出不能覆盖运行快照、输入或答案原件")
    package = read_package(args.manifest)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    prepared = prepare_answers(package.task, package.batches, json.loads(args.answers.read_text(encoding="utf-8")))
    final = score_final(package.task, state, prepared)
    definition = json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else {}
    atomic_json(args.output/"final-evaluation.json", final)
    atomic_json(args.output/"final-longitudinal.json", evaluate_history(state, **definition, final_report=final))
    print("已对应原服务预测与终局答案；覆盖状态="+final["status"]+"，未调用模型。")


if __name__ == "__main__":
    main()
