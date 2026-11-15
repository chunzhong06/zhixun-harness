"""三类官方CLI的容器集成检查；使用明确标注的模型回复替身，不作性能结论。"""
from pathlib import Path
import json
import sys
from harness.integrations import platform_train
from harness.integrations.pytrio_planner import ChatResponse

class Fixture:
    endpoint='https://pytrio.com/api/openai/v1/completions'
    def __init__(self,*args,**kwargs):pass
    def send(self,request):
        return ChatResponse(200,{'choices':[{'text':json.dumps({'reason':'工程替身，只验证入口','python_source':'def train(session):\n    session.train_epoch()\n'})}], 'fixture':True})

if __name__=='__main__':
    platform_train.HttpChatTransport=Fixture
    results={}
    for task,profile in [('gas','A'),('bearing','B'),('engine','C')]:
        root=Path('/fixtures')/task;out=Path('/output')/task
        sys.argv=['train','--profile',profile,'--train',str(root/'train.jsonl'),'--validation',str(root/'validation.jsonl'),
                  '--from-baseline',str(root/'checkpoint.npz'),'--output',str(out),'--epochs','1']
        platform_train.main()
        r=json.loads((out/'training.json').read_bytes())
        assert r['error'] is None and r['training_audit']['final_steps']>r['training_audit']['initial_steps']
        assert (out/'checkpoint.npz').is_file() and (out/'baselines/predict.py').is_file()
        results[task]={'actual_gradient_update':True,'decision':r['decision'],'error':r['error']}
    Path('/output/verification.json').write_text(json.dumps({'simulated_model_service':True,'real_container_cli':True,'tasks':results},ensure_ascii=False,indent=2),encoding='utf8')
