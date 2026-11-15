"""把一次模型原文封装为队列响应，编号和目标路径均从原请求取得。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'engineering/src'))
from harness.integrations.response_format import normalize_action_json


def publish(request, text_file):
    request=request.resolve();text_file=text_file.resolve()
    if request.parent.name!='mle-model' or request.parents[1].name!='requests':
        raise ValueError('需要模型队列中的原始请求')
    payload=json.loads(request.read_bytes());body=text_file.read_text(encoding='utf-8')
    if not body.strip():raise ValueError('模型原文为空')
    response=request.parents[2]/'responses'/request.parent.name/request.name
    if response.exists():raise ValueError('已有正式响应，不能覆盖')
    original=body;repair=None
    if payload.get('context',{}).get('actor')=='harness_research_controller':
        body,repair=normalize_action_json(body)
    response.parent.mkdir(parents=True,exist_ok=True)
    temporary=response.with_suffix('.tmp')
    temporary.write_text(json.dumps({'request_id':payload['request_id'],'plan':{'text':body}},ensure_ascii=False),encoding='utf-8')
    assert json.loads(temporary.read_bytes())['plan']['text']==body
    if repair:
        response.with_suffix('.format-repair.json').write_text(json.dumps({
            'request_id':payload['request_id'],'original_text':str(text_file),
            'original_text_sha256':sha256(original.encode()).hexdigest(),
            'delivered_text_sha256':sha256(body.encode()).hexdigest(),
            'repair':repair,'response_text_completed':True,
            'research_fields_modified':False,'new_model_calls':0},ensure_ascii=False),encoding='utf-8')
    temporary.replace(response)
    print('已原子发布：'+str(response))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--text',type=Path,required=True)
    args=parser.parse_args();publish(args.request,args.text)
