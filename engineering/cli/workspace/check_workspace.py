"""核查现行文档、原始来源和源码入口；不运行模型或研究代码。"""
from pathlib import Path
from urllib.parse import unquote
from collections import Counter
import ast
from functools import lru_cache
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
LINK = re.compile(r'\[[^\]\n]+\]\(([^)\n]+)\)')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def anchors(path):
    """支持本项目标题及显式HTML锚点，重复标题按GitHub惯例编号。"""
    text = path.read_text(encoding='utf-8-sig')
    found = set(re.findall(r'<a\s+id="([^"]+)"', text))
    counts = Counter()
    for heading in re.findall(r'^#{1,6}\s+(.+)$', text, re.M):
        slug = re.sub(r'[^\w\-\s\u4e00-\u9fff]', '', heading.strip().lower()).replace(' ', '-')
        suffix = '' if counts[slug] == 0 else '-' + str(counts[slug])
        found.add(slug + suffix)
        counts[slug] += 1
    return found


def main():
    errors = []
    counts = Counter()
    docs = [ROOT / name for name in ['README.md', 'todo.md', 'docs/llms.txt',
            'references/README.md', 'data/README.md', 'results/README.md',
            'analysis/README.md', 'results/experiments/README.md']]
    # 这里只检查人工维护的结果主题和派生导航，不遍历原始运行日志。
    docs.extend(sorted((ROOT/'results/experiments/local-defaults').glob('*.md')))
    for folder in ('docs', 'analysis'):
        docs.extend(sorted((ROOT / folder).rglob('*.md')))
    for doc in docs:
        if not doc.exists():
            errors.append('文档缺失：' + str(doc.relative_to(ROOT)))
            continue
        # 命令片段中的括号不作为导航；外部原件不受本地编辑规则改写。
        body = re.sub(r'```[\s\S]*?```', '', doc.read_text(encoding='utf-8-sig'))
        for target in LINK.findall(body):
            if target.startswith(('https:', 'http:', 'mailto:', 'chatgpt-', 'plugin:', 'codex:')):
                continue
            decoded = unquote(target)
            file, sep, fragment = decoded.partition('#')
            path = (doc.parent / file).resolve() if file else doc
            counts['local_links'] += 1
            if not path.exists():
                errors.append(f'{doc.relative_to(ROOT)} → {target}')
            elif fragment and path.suffix == '.md' and fragment not in anchors(path):
                errors.append(f'锚点：{doc.relative_to(ROOT)} → {target}')

    question = ROOT / 'analysis/questions/问题清单.md'
    if sha(question) != '6c2e6e526bdfe9f933ba8bef94906f86d9ca8bc1b5a80ea698472465d99cb4a1':
        errors.append('原问题清单内容变化')
    catalogs = {
        'references/index/catalog.json': ('references/papers', 'references/implementations',
            'references/manuals', 'references/official', 'references/retrieval'),
        'data/index/inputs.json': ('data/official', 'data/public'),
    }
    for catalog, folders in catalogs.items():
        rows = json.loads((ROOT / catalog).read_text(encoding='utf-8'))
        expected = {p.relative_to(ROOT).as_posix() for folder in folders
                    for p in (ROOT / folder).rglob('*') if p.is_file()}
        if 'data/public' in folders:
            # 下载器在每个数据源根目录保留进程锁；它在释放后仍存在，不是输入原件。
            public_inputs = ROOT / 'data/public'
            expected.difference_update(p.relative_to(ROOT).as_posix()
                                       for p in public_inputs.glob('*/lock/.lock'))
        registered = {row['path'] for row in rows}
        errors.extend('原件未登记：' + name for name in sorted(expected - registered))
        if len(registered) != len(rows):
            errors.append('重复的文件登记：' + catalog)
        for row in rows:
            path = ROOT / row['path']
            if not path.is_file() or sha(path) != row['sha256']:
                errors.append('原件缺失或内容变化：' + row['path'])
            counts['source_catalog_rows'] += 1
    topic_root = ROOT / 'analysis/topics'
    topics = {p.relative_to(topic_root).as_posix() for p in topic_root.glob('*/N[0-9][0-9][0-9]-*.md')}
    navigation = set(LINK.findall((topic_root / 'README.md').read_text(encoding='utf-8')))
    errors.extend('正文未进入专题导航：' + name for name in sorted(topics - navigation))
    for folder in ['engineering/src', 'engineering/tests', 'engineering/cli']:
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            counts['parsed_python_files'] += 1
    result = {'ok': not errors, 'counts': dict(counts), 'errors': errors, 'scope': 'workspace_integrity', 'model_training': False}
    target = ROOT / 'results/engineering/workspace/checks/integrity.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'ok': result['ok'], 'counts': result['counts'], 'errors': errors[:25], 'error_count': len(errors)}, ensure_ascii=False))
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
