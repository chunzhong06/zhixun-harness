"""按文件或章节读取事实并返回身份；超出预算明确拒绝截断。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def sections(text):
    """识别本项目Markdown标题及显式锚点，忽略代码块内标题。"""
    found, counts, fence = [], Counter(), None
    for index, line in enumerate(text.splitlines(keepends=True)):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1][0]
            fence = None if fence == token else token if fence is None else fence
            continue
        if fence:
            continue
        title = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        anchor = re.search(r'<a\s+id="([^"]+)"', line)
        if title:
            slug = re.sub(r"[^\w\-\s\u4e00-\u9fff]", "", title[2].lower()).replace(" ", "-")
            key = slug + (f"-{counts[slug]}" if counts[slug] else "")
            counts[slug] += 1
            found.append((index, len(title[1]), key))
        elif anchor:
            found.append((index, None, anchor[1]))
    return found


def read_ref(ref, *, root=ROOT):
    requested_ref = ref
    routes_path = root / "docs/index/section-routes.json"
    routes = json.loads(routes_path.read_text(encoding="utf-8")) if routes_path.exists() else {}
    seen = set()
    # 冻结记录可能引用拆分前的章节；只跟随本地登记，不从旧页截取相邻内容。
    while ref in routes:
        if ref in seen:
            raise ValueError("章节转向存在循环：" + requested_ref)
        seen.add(ref)
        ref = routes[ref]
    name, _, anchor = ref.partition("#")
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix not in {".md", ".txt"}:
        raise ValueError("入口须为工作区内的Markdown或文本文件")
    raw = path.read_bytes()
    body = raw.decode("utf-8-sig")
    lines, entries = body.splitlines(keepends=True), sections(body)
    if anchor:
        matches = [x for x in entries if x[2] == anchor]
        if len(matches) != 1:
            raise ValueError("锚点缺失或重复："+ref)
        start, level, _ = matches[0]
        body_start = start
        if level is None:
            following = next((x for x in entries if x[0] > start and x[1] is not None), None)
            level = following[1] if following else 2
            body_start = following[0] if following else start
        end = next((i for i, depth, _ in entries if i > body_start and depth is not None and depth <= level), len(lines))
        first_section = next((i for i, depth, _ in entries if depth is not None and depth >= 2), len(lines))
        ranges = [(0, first_section)] if start >= first_section and first_section else []
        ranges.append((start, end))
    else:
        ranges = [(0, len(lines))]
    content = "".join("".join(lines[a:b]) for a,b in ranges)
    return {"ref": requested_ref, "resolved_ref": ref, "file_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "line_ranges": [[a+1,b] for a,b in ranges], "complete_selection": True, "content": content}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", action="append", default=[])
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        values = [read_ref(ref) for ref in args.ref or ["docs/llms.txt"]]
        size = sum(len(v["content"]) for v in values)
        if size > args.max_chars:
            raise ValueError(f"完整内容为{size}字符，超过预算{args.max_chars}；按章节分次读取或显式扩大预算")
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+"\n")
    if args.json:
        print(json.dumps(values, ensure_ascii=False, indent=2))
    else:
        for value in values:
            location=value['ref']
            if value['resolved_ref']!=location:
                location+=' → '+value['resolved_ref']
            print(f"【{location}；文件SHA256 {value['file_sha256']}】\n{value['content']}")


if __name__ == "__main__":
    main()
