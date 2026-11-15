"""打包已冻结设计与少量原始证据供组内审阅；不训练、不联网、不替换模型。"""
from datetime import datetime, timezone
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/reviews/2026-11-15-design-review'
BUNDLE = ROOT / 'results/engineering/contest-native-delivery/2026-11-15-transfer'
EVIDENCE = {
    'current-bundle-verification.json': 'results/engineering/contest-native-delivery/2026-11-15-transfer/verification.json',
    'current-selection.json': 'results/experiments/contest-continuation/native-engine-long-balanced/delivery-selection.json',
    'luna-native-gas.json': 'results/experiments/contest-continuation/native-online-gas/verification.json',
    'ten-stage-reuse.json': 'results/experiments/contest-continuation/native-ten-stages/verification.json',
    'native-recovery.json': 'results/experiments/contest-continuation/final-verification/verification.json',
    'standalone-recording.json': 'results/experiments/contest-continuation/native-standalone-recording/verification.json',
    'bearing-adoption.json': 'results/experiments/contest-continuation/bearing-adoption-audit/summary.json',
    'engine-ablation.json': 'results/experiments/contest-continuation/native-engine-balanced-ablation/verification.json',
    'engine-final.json': 'results/experiments/contest-continuation/native-engine-long-balanced/verification.json',
    'cross-task.json': 'results/experiments/contest-continuation/native-balanced-cross-task/summary.json',
    'feedback-secom.json': 'results/experiments/contest-continuation/feedback-research-secom/summary.json',
    'comind-engine.json': 'results/experiments/local-defaults/whole-maintenance/robust-shared-full-pair/engine/comparison.json',
    'comind-transfer.json': 'results/experiments/local-defaults/whole-maintenance/robust-shared-transfer/scores.json',
    'comind-london.json': 'results/experiments/local-defaults/whole-maintenance/london-frozen-tail/scores.json',
    'latest-cloud-receipt.json': 'results/engineering/research-workflow/swanlab/run_0430/cloud.json',
}


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)


def render(text):
    """只处理本审阅稿用到的静态Markdown结构，输出无需网络依赖的页面。"""
    lines = text.splitlines(); parts = []; toc = []; number = 0; i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith('```'):
            i += 1; code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            parts.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>'); i += 1; continue
        if line.startswith('#'):
            match = re.match(r'^(#{1,3}) (.+)$', line)
            if not match:
                raise ValueError('不支持的标题格式')
            level, title = len(match[1]), match[2]
            attr = ''
            if level == 2:
                number += 1; attr = f' id="section-{number}"'; toc.append((number, title))
            parts.append(f'<h{level}{attr}>' + inline(title) + f'</h{level}>'); i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                if not re.fullmatch(r'[| :\-]+', lines[i]):
                    rows.append([x.strip() for x in lines[i].strip('|').split('|')])
                i += 1
            body = ['<div class="table-wrap"><table><thead><tr>']
            body.extend('<th scope="col">' + inline(x) + '</th>' for x in rows[0])
            body.append('</tr></thead><tbody>')
            for row in rows[1:]:
                if len(row) != len(rows[0]):
                    raise ValueError('表格列数不一致')
                body.append('<tr>' + ''.join('<td>' + inline(x) + '</td>' for x in row) + '</tr>')
            body.append('</tbody></table></div>'); parts.append(''.join(body)); continue
        if line.startswith('- ') or re.match(r'^\d+\. ', line):
            ordered = not line.startswith('- '); tag = 'ol' if ordered else 'ul'; items = []
            expression = r'^\d+\. ' if ordered else r'^- '
            while i < len(lines) and re.match(expression, lines[i]):
                items.append('<li>' + inline(re.sub(expression, '', lines[i], count=1)) + '</li>'); i += 1
            parts.append(f'<{tag}>' + ''.join(items) + f'</{tag}>'); continue
        paragraph = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|- |\d+\. )', lines[i]):
            paragraph.append(lines[i]); i += 1
        parts.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    navigation = '<nav aria-label="章节">' + ' '.join(f'<a href="#section-{n}">{html.escape(t)}</a>' for n, t in toc) + '</nav>'
    css = '''body{margin:0;background:#f3f5f7;color:#202b37;font:16px/1.8 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}main{max-width:1000px;margin:32px auto;padding:40px 56px;background:white;box-sizing:border-box}h1{font-size:30px;line-height:1.4;color:#132b43;margin:0 0 12px}h2{font-size:23px;line-height:1.5;margin:38px 0 14px;scroll-margin-top:20px}h3{font-size:18px;margin:24px 0 8px}p{margin:12px 0}a{color:#175b8a;text-underline-offset:3px}nav{font-size:14px;padding:14px 0 4px;display:flex;flex-wrap:wrap;gap:8px 22px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:14px 0}th{background:#eaf0f5;text-align:left}th,td{border-bottom:1px solid #d7dfe6;padding:10px 12px;vertical-align:top}thead{border-top:2px solid #607b93}.table-wrap{overflow:auto}code{font:13px/1.65 Consolas,monospace;overflow-wrap:anywhere;background:#f1f3f6;padding:2px 4px}pre{padding:16px;background:#f1f3f6;overflow:auto}pre code{padding:0}li{margin:8px 0}footer{margin-top:32px;color:#546473;font-size:13px}@media(max-width:700px){main{margin:0;padding:24px 20px}h1{font-size:26px}body{font-size:15px}}@media print{body{background:white;font-size:10pt}main{margin:0;padding:0;max-width:none}nav{display:none}h2,h3{break-after:avoid}tr{break-inside:avoid}a{color:inherit}table{font-size:9pt}pre{white-space:pre-wrap}h2{margin-top:22px}@page{size:A4;margin:18mm}}'''
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智训引擎设计审阅</title><style>' + css + '</style></head><body><main>' + navigation + '\n'.join(parts) + '<footer>本地审阅快照 · 代码、模型与证据由清单绑定 · 未执行外部发送或比赛提交</footer></main></body></html>'


VERIFIER = '''"""只读核验本审阅包，不联网或训练。修改审阅意见后，摘要不匹配会如实报告。"""
from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
failures = []
for name, item in manifest['files'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(name + ': missing or invalid path')
    elif path.stat().st_size != item['bytes'] or sha256(path.read_bytes()).hexdigest() != item['sha256']:
        failures.append(name + ': changed')
print(json.dumps({'passed': not failures, 'checked_files': len(manifest['files']), 'failures': failures}, ensure_ascii=False))
raise SystemExit(bool(failures))
'''


def build():
    package = OUT / 'package'
    package.mkdir(exist_ok=False)
    (package / 'runtime').mkdir(); (package / 'evidence').mkdir()
    source = BUNDLE / 'zhixun-native-source.zip'
    verification = json.loads((BUNDLE / 'verification.json').read_bytes())
    if checksum(source) != verification['sha256']:
        raise ValueError('当前运行包与已核验版本不一致')
    shutil.copyfile(source, package / 'runtime' / source.name)
    for name in ('设计审阅.md', '运行说明.md', '审阅反馈.md'):
        shutil.copyfile(OUT / name, package / name)
    text = (OUT / '设计审阅.md').read_text(encoding='utf-8')
    (package / 'index.html').write_text(render(text), encoding='utf-8')
    (package / '验证审阅包.py').write_text(VERIFIER, encoding='utf-8')
    source_map = {}
    for name, relative in EVIDENCE.items():
        original = ROOT / relative
        shutil.copyfile(original, package / 'evidence' / name)
        source_map[name] = {'workspace_source': relative, 'sha256': checksum(original)}
    state = json.loads((ROOT / 'research-state.json').read_bytes())
    runs = []
    for r in state['runs']:
        if 399 <= int(r['id'].split('_')[-1]) <= 430:
            protocol = json.loads((ROOT / r['protocol']).read_bytes())
            runs.append({k: r[k] for k in ('id', 'status', 'started', 'finished', 'elapsed_seconds')} | {'name': protocol['中文名称']})
    (package / 'evidence/current-run-index.json').write_text(json.dumps({'scope': 'run399—430，包括失败；墙钟不等同CPU或Token成本',
        'runs': runs, 'elapsed_seconds_sum': sum(r['elapsed_seconds'] for r in runs)}, ensure_ascii=False, indent=2), encoding='utf-8')
    (package / 'evidence/source-map.json').write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding='utf-8')
    (package / '先读我.txt').write_text('请先用浏览器打开 index.html。\n代码与当前权重在 runtime/zhixun-native-source.zip；关键原始证据在 evidence/。\n本包用于组内设计审阅，未正式提交；自主研究保持暂停。\n可运行 python -B 验证审阅包.py 离线核验文件。\n', encoding='utf-8')
    for name in ('设计审阅.md', '运行说明.md', '审阅反馈.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (package / name).read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#') and not (package / target).exists():
                raise ValueError('审阅文件有断链：' + target)
    files = {}
    for path in sorted(package.rglob('*')):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if path.suffix in ('.md', '.html', '.txt', '.json', '.py') and re.search(rb'(?:trio_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,})', raw):
            raise ValueError('审阅包疑似含凭据')
        files[path.relative_to(package).as_posix()] = {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'purpose': '组内设计审阅，非比赛提交',
        'research_status': state['status'], 'runtime_bundle_sha256': verification['sha256'], 'files': files}
    (package / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    destination = OUT / '智训引擎设计审阅包.zip'
    with ZipFile(destination, 'w', ZIP_DEFLATED) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                z.write(path, '智训引擎设计审阅/' + path.relative_to(package).as_posix())
    with ZipFile(destination) as z, TemporaryDirectory(dir=OUT, prefix='temporary-review-check-') as temporary:
        if z.testzip() is not None:
            raise ValueError('压缩包CRC错误')
        z.extractall(temporary)
        result = subprocess.run([sys.executable, '-B', '验证审阅包.py'], cwd=Path(temporary) / '智训引擎设计审阅',
            capture_output=True, text=True, encoding='utf-8', timeout=30)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        checked = json.loads(result.stdout)
    report = {'zip': str(destination), 'bytes': destination.stat().st_size, 'sha256': checksum(destination),
        'checked_files': checked['checked_files'], 'all_local_links_valid': True, 'secrets_scan_passed': True,
        'unpacked_hash_check_passed': True, 'runtime_bundle_unchanged': True,
        'new_training': 0, 'model_service_calls': 0, 'external_submission': False}
    (OUT / 'package-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    build()
