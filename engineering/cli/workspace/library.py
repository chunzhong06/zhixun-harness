"""按题名、来源、方法或 N 编号查询当前资料和团队研究。"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def topics():
    """从实际文稿发现专题，重号或文件名与正文不符时报告具体文件。"""
    entries = {}
    for path in sorted((ROOT / "analysis/topics").glob("*/N[0-9][0-9][0-9]-*.md")):
        body = path.read_text(encoding="utf-8")
        heading = re.search(r"(?m)^# (N\d{3}) (.+)$", body)
        if not heading or not path.name.startswith(heading[1] + "-") or heading[1] in entries:
            raise ValueError(f"专题编号与标题不符或重复：{path}")
        entries[heading[1]] = (path.relative_to(ROOT), heading[2], body)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("find", "note"))
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=20, help="每类结果的显示上限")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 须为正整数")
    notes = topics()
    if args.command == "note":
        entry = notes.get(args.query.upper())
        if entry is None:
            parser.error("没有该专题编号")
        print(entry[2].strip())
        return
    key = args.query.casefold()
    sources = json.loads((ROOT / "references/index/catalog.json").read_text(encoding="utf-8"))
    for category in ("papers", "implementations", "manuals", "official", "retrieval"):
        matches = [row["path"] for row in sources if row["category"] == category
                   and key in json.dumps(row, ensure_ascii=False).casefold()]
        for path in matches[:args.limit]:
            print(f"{category}\t{path}")
    hits = [(number, entry) for number, entry in sorted(notes.items())
            if key in entry[2].casefold() or key == number.casefold()]
    for number, (path, title, _) in hits[:args.limit]:
        print(f"{number}\t{title}\t{path.as_posix()}")
    # 编号专题保留初始调查；后续实测和研究判断在主题主说明中继续维护。
    # 查当前知识时一并返回这些位置，兼容锚点目录不重复制造命中。
    for label,folder in (("研究判断","analysis/synthesis"),("实测结果","results/experiments/local-defaults"),
                         ("现有设计","docs/design"),("运行与核验","docs/operations"),("推进记录","docs/history")):
        found=[]
        for path in sorted((ROOT/folder).glob('*.md')):
            body=path.read_text(encoding='utf-8-sig').split('## 旧引用入口',1)[0]
            if key in body.casefold():found.append(path)
        for path in found[:args.limit]:
            print(f"{label}\t{path.stem}\t{path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
