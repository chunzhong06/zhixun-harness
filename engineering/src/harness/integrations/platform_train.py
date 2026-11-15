"""沿官方train命令接收平台实际数据与父版；一次调用对应平台一轮。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from harness.models.complete_checkpoint import load_checkpoint
from harness.models.native_continuation import ContinuationSession, run_program
from harness.integrations.pytrio_planner import PyTrioConfig, ChatRequest
from harness.integrations.service_transport import HttpChatTransport
from harness.integrations.service_budget import ServiceBudget
from harness.runtime.storage import atomic_json


def main():
    from baselines.core_model import Model
    from baselines.data import read_dataset
    parser=ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--train',type=Path,required=True);parser.add_argument('--validation',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--resume',type=Path);group.add_argument('--from-baseline',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage-id',default='platform-round')
    parser.add_argument('--epochs',type=int,default=60);parser.add_argument('--label-provenance')
    parser.add_argument('--learning-rate',type=float);parser.add_argument('--batch-size',type=int);parser.add_argument('--seed',type=int)
    args=parser.parse_args();parent=args.resume or args.from_baseline
    model=load_checkpoint(parent,Model)
    if model.meta['profile']!=args.profile:raise ValueError('任务与实际父版不匹配')
    train=read_dataset(args.train,model.meta['task'],model.meta['classes'],model.meta['paths'])
    valid=read_dataset(args.validation,model.meta['task'],model.meta['classes'],model.meta['paths'])
    if set(train['ids'])&set(valid['ids']):raise ValueError('训练和验证身份交叉')
    args.output.mkdir(parents=True,exist_ok=False)
    initial_steps,initial_epochs=model.meta['steps'],model.meta['epochs']
    def loss(m):
        p=m.predict(valid['x']);y=valid['y']
        if m.meta['task']=='regression':return float(np.sqrt(np.mean((p-y)**2)))
        labels=p.argmax(1);scores=[]
        for k in range(len(m.meta['classes'])):
            denom=np.sum(labels==k)+np.sum(y==k);scores.append(2*np.sum((labels==k)&(y==k))/denom if denom else 0.)
        return 1-float(np.mean(scores))
    before=loss(model)
    # 与平台三任务各10轮相对应，每轮至多一请求、预留最多0.25元；一次试运行上限7.5元。
    budget=ServiceBudget(args.output/'call-budget');budget.initialize(.25)
    os.environ['HARNESS_SERVICE_BUDGET']=str((args.output/'call-budget').resolve())
    instruction='你在真实续训官方工业模型。只返回JSON {"python_source":"def train(session): ...","reason":"简述依据"}。程序只能import numpy或math。session.metadata/features/targets/groups提供合法训练副本；session.train_epoch(learning_rate=可选,batch_size=可选,trainable=可选参数名)更新父权重Adam；只训练输出层参数名w2,b2。最多60遍。普通回归父版可build_regression_components(condition_indices,decimals,epochs=60,head_epochs=12)建立源工况与残差，需真实有序设备数据。只使用已提供的字段语义，无把握用少量低学习率续训。'
    facts={'profile':args.profile,'task':model.meta['task'],'paths':model.meta['paths'],'train_rows':len(train['y']),
           'classes':model.meta['classes'],'parent_epochs':initial_epochs,'validation_loss':before,
           'complete_parent':hasattr(model,'auxiliary_arrays')}
    body={'model':'Qwen/Qwen3.8-27B','messages':[{'role':'system','content':instruction},{'role':'user','content':json.dumps(facts,ensure_ascii=False)}],'max_tokens':8192,'temperature':0}
    action=None;error=None;audit=None;accepted=False
    try:
        if len(json.dumps(body,ensure_ascii=False).encode())>10240:raise ValueError('本轮请求超出预留费用的输入上限')
        from harness.integrations.qwen_text_template import render
        body['prompt']=render(body.pop('messages'));body['stop']=['<|im_end|>','<|endoftext|>']
        config=PyTrioConfig(model=body['model'],max_tokens=8192,timeout_seconds=600,raw_completion=True)
        transport=HttpChatTransport(config,os.environ.get('PYTRIO_API_KEY'),allow_network=True)
        response=transport.send(ChatRequest(transport.endpoint,body,600))
        atomic_json(args.output/'model-response.json',{'status':response.status,'body':response.body})
        if response.status!=200:raise RuntimeError('模型HTTP状态'+str(response.status))
        action=json.loads(response.body['choices'][0]['text'])
        candidate=load_checkpoint(parent,Model)
        raw=[json.loads(line) for line in args.train.read_text(encoding='utf8').splitlines() if line.strip()]
        groups=np.asarray([str((row.get('meta',{}).get('source_subset','source'),row.get('meta',{}).get('unit',row.get('meta',{}).get('bearing','source')))) for row in raw])
        session=ContinuationSession(candidate,train['x'],train['y'],groups=groups,max_epochs=min(args.epochs,60))
        audit=run_program(action['python_source'],session);candidate=session._model
        after=loss(candidate);accepted=after<before
        if accepted:model=candidate
    except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
    model.save(args.output/'checkpoint.npz')
    report={'format':'harness.platform-round.v1','stage_id':args.stage_id,'profile':args.profile,
        'parent_checkpoint_sha256':sha256(parent.read_bytes()).hexdigest(),'checkpoint_sha256':sha256((args.output/'checkpoint.npz').read_bytes()).hexdigest(),
        'initial_steps':initial_steps,'final_steps':model.meta['steps'],'initial_epochs':initial_epochs,'final_epochs':model.meta['epochs'],
        'decision':'adopt' if accepted else 'retain','validation_before':before,'validation_after':loss(model),
        'train_sha256':train['sha256'],'validation_sha256':valid['sha256'],'action':action,'training_audit':audit,'error':error}
    atomic_json(args.output/'training.json',report)
    (args.output/'history.jsonl').write_text(json.dumps(report,ensure_ascii=False)+'\n',encoding='utf8')
    import shutil
    shutil.copytree(Path('/app/baselines'),args.output/'baselines')
    if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>64*1024**2:raise ValueError('交付超过64MiB')
    print(json.dumps(report,ensure_ascii=False),flush=True)
