"""列出待处理请求，并将误投到 requests/responses 的原响应按身份原样路由。

仅修正目录位置，不改写模型文本。原件及每次路由的 SHA256 均保留。
"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path


def poll(root):
    for request in sorted(root.glob('**/queue/requests/*/*.json')):
        queue=request.parents[2];arm=request.parents[3]
        if (arm/'driver-complete.json').exists():continue
        if request.name.startswith('0032-comind_coder_Thread-16_') and 'gas/comind/' in request.as_posix():
            continue  # 原32请求已超时，实际流程已用33完成报告；保留误投原件。
        response=queue/'responses'/request.parent.name/request.name
        if not response.exists():
            candidates=[queue/'requests/responses'/request.parent.name/request.name,response.with_suffix('')]
            misplaced=next((p for p in candidates if p.is_file()),None)
            if misplaced:
                raw=misplaced.read_bytes();value=json.loads(raw)
                if value.get('request_id')!=json.loads(request.read_bytes())['request_id'] or not isinstance(value.get('plan',{}).get('text'),str):
                    raise ValueError('误投响应身份或结构不符，不能自动路由')
                response.parent.mkdir(parents=True,exist_ok=True)
                temporary=response.with_suffix('.routing.tmp');temporary.write_bytes(raw);temporary.replace(response)
                audit=arm/'transport-repairs.json'
                events=json.loads(audit.read_bytes()) if audit.exists() else []
                events.append({'request':request.name,'original':str(misplaced),'delivered':str(response),
                               'sha256':sha256(raw).hexdigest(),'model_text_changed':False,'extra_model_calls':0})
                audit.write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
                print('已原样路由：'+str(request.relative_to(root)))
        if not response.exists():print(str(request.relative_to(root)))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    poll(parser.parse_args().root.resolve())
