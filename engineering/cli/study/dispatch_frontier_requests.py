"""用短别名传递原请求，原文就绪后代为发布；只修传输，不增加研究信息。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engineering.cli.study.publish_luna_response import publish


def pending(case):
    if not case.is_relative_to(ROOT/'results/experiments'):
        raise ValueError('传输别名只位于本项目实验目录')
    items, published = [], 0
    for arm in ['project', 'comind']:
        folder = case/arm; queue = folder/'queue'
        for request in sorted((queue/'requests/mle-model').glob('*.json')):
            response = queue/'responses/mle-model'/request.name
            if response.exists():
                continue
            payload = json.loads(request.read_bytes()); number = f"{payload['context']['sequence']:04d}"
            raw = folder/'response-text'/(number+'.txt')
            alias = folder/'dispatch'/(number+'.json')
            binding = alias.with_suffix('.binding.json')
            if binding.exists():
                saved = json.loads(binding.read_bytes())
                if saved['request_id'] != payload['request_id'] or alias.read_bytes() != request.read_bytes():
                    raise ValueError('请求别名身份或内容变化')
            else:
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.write_bytes(request.read_bytes())
                binding.write_text(json.dumps({'request': str(request.resolve()), 'request_id': payload['request_id'],
                    'sha256': sha256(request.read_bytes()).hexdigest(), 'mechanical_alias_only': True},
                    ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            # 只接受角色完成的封存文本；原子就绪标记避免读到正在写的半个响应。
            if raw.with_suffix('.ready').exists():
                if not raw.is_file() or not raw.read_text(encoding='utf-8').strip():
                    # 保留原请求与故障，交由同角色补写；一臂的空文件不能阻塞另一臂。
                    items.append({'arm': arm, 'sequence': number,
                        'request_alias': alias.relative_to(ROOT).as_posix(),
                        'response_text': raw.relative_to(ROOT).as_posix(),
                        'transport_error': '就绪原文缺失或为空', 'requires_existing_actor_recovery': True})
                    continue
                publish(request, raw)
                saved = json.loads(binding.read_bytes())
                saved.update(published=True, temporary_alias_removed=True,
                             raw_response_sha256=sha256(raw.read_bytes()).hexdigest(),
                             published_response_sha256=sha256(response.read_bytes()).hexdigest())
                binding.write_text(json.dumps(saved, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                raw.with_suffix('.ready').unlink()
                alias.unlink()
                published += 1
            else:
                items.append({'arm': arm, 'sequence': number, 'request_alias': alias.relative_to(ROOT).as_posix(),
                    'response_text': raw.relative_to(ROOT).as_posix(), 'ready_marker': raw.with_suffix('.ready').relative_to(ROOT).as_posix()})
    return items, published


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('--case', type=Path, required=True)
    case = parser.parse_args().case.resolve()
    items, published = pending(case)
    if published:
        # 给短程序和作者控制器交接留出时间，再列出新请求，减少每格额外轮询等待。
        time.sleep(3)
        items, _ = pending(case)
    for item in items:
        print(json.dumps(item, ensure_ascii=False))
