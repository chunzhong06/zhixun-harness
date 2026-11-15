"""同任务接口与预算下，自主选择检索、完整源码试验和结束的项目驱动。"""
import json
from hashlib import sha256
from pathlib import Path

from harness.integrations.research_broker import ACTOR
from harness.integrations.research_ledger import history_entries
from harness.integrations.candidate_selection import candidate_id, choose_candidate
from harness.runtime.storage import atomic_json


INSTRUCTION='''你是Harness的现场研究驱动。目标是改进此任务的完整后续交付；只按本请求的合法资料和真实返回结果判断。
可自行选择检查什么机制、采用何种模型或组合、下一次试验和何时结束。不要为了用完额度继续调用。
你的输入、Python执行接口、搜索能力、数据时间边界和资源上限与完整作者对照相同。已有较强源码在task_description中；available_references列出的源码可以原样执行，先取得真实参照再决定怎样改动。也可提出新的完整实现。
返回纯JSON，选择以下一种动作：
1. {"action":"experiment","hypothesis":"中文可检验判断","python_source":"完整单文件Python程序"}。程序遵守任务说明，通过evaluate_model或submit_model保存实际模型；一次程序可以批量评估多个有明确标签的候选，并复用同一次合法训练得到的部件。外层会返回每个候选的ID和真实探索分数。可打印训练资料检查和模型诊断，但不能自行读取外层答案或伪造评分。
2. {"action":"search","queries":["公开查询"]}，一次至多两项，搜索结果将原样回传。
3. {"action":"finish","execution":已成功执行的编号,"candidate":"该执行中的候选ID，例如candidate-001","reason":"结束并提名该实际模型的理由"}。请显式给出candidate；省略时仅按历史接口选择该程序最后保存的模型，不会自动选择分数最低者。标签和候选生成顺序应在同一冻结程序的发布补训中保持。
4. {"action":"reference","name":"available_references中的名字","hypothesis":"运行该既有方案需要核查什么"}。外层会原样执行已提供源码并返回真实结果；不用重新抄写实现，也不能在此动作中另附python_source。该动作同样消耗一次模型调用及一次程序执行，不能把参考源码当作已测成绩。
每个新实验都会真实执行，失败、超时和退步也消耗预算。两个探索周的真实RMSE用于比较；最后确认周不返回，提名完成后由共同发布程序决定是否采用。新有效方案可以成为下一步改动的起点，保留未改动的有效部分。最后阶段会按冻结源码重新拟合合法发布权重；确认后的训练内成绩不会重新供你选择。
'''


async def run_free_research(task_description,task,workspace,executor,broker,*,instruction=None,reference_sources=None,resume=False,copy_inputs=True):
    """不共享对手生成的方案或已开分答案；只接收自身合法执行记录。"""
    work=Path(workspace)/'project';(work/'input').mkdir(parents=True,exist_ok=resume)
    import shutil
    # 原生续训执行器直接绑定宿主task_dir，不读取此副本；十轮运行不必重复复制训练数组。
    for p in Path(task).iterdir() if copy_inputs else ():
        target=work/'input'/p.name
        if target.exists():
            if not resume or p.read_bytes()!=target.read_bytes():raise ValueError('恢复工作区的任务输入变化')
        else:shutil.copy2(p,target)
    observations=[];events=[]
    references=dict(reference_sources or {})
    # 仅宿主显式准备的任务包开放此能力；旧任务和官方任务缺省不扩充权限。
    access_file=Path(task)/'feedback-training.json'
    if access_file.exists() and json.loads(access_file.read_bytes()).get('enabled') is True:
        from harness.integrations.feedback_task import describe_training_access,reference_sources as feedback_sources
        feedback_references=feedback_sources()
        if any(name in references and references[name]!=code for name,code in feedback_references.items()):
            raise ValueError('反馈训练参考源码与调用方已有同名程序冲突')
        references.update(feedback_references)
        task_description += '\n'+describe_training_access(task)+'\n反馈训练参考源码：\n'+ '\n'.join(
            '### '+name+'\n```python\n'+code+'\n```' for name,code in feedback_references.items())
    if any(not isinstance(k,str) or not isinstance(v,str) for k,v in references.items()):
        raise ValueError('参考入口必须绑定调用方明确提供的名字和完整源码')
    while True:
        ACTOR.set('harness_research_controller')
        successful=[r for r in executor.records if r['returncode']==0 and r['candidates']]
        best={r['sequence']:min(r['candidates'],key=lambda c:c['score']) for r in successful}
        ranked=sorted(successful,key=lambda r:best[r['sequence']]['score'])
        context={'task_description':task_description,'remaining_model_invocations':broker.max_calls-broker.prior_model_invocations-broker.calls+broker.reused_model_calls,
            'available_references':[{'name':name,'source_sha256':sha256(code.encode()).hexdigest()} for name,code in references.items()],
            'history':history_entries(executor.records,events),
            'best_working_programs':[{'execution':r['sequence'],'source':Path(r['source']).read_text(encoding='utf-8'),
                'output':r['stdout'],'best_candidate_id':candidate_id(best[r['sequence']]),
                'best_candidate_label':best[r['sequence']]['label'],
                'scores':best[r['sequence']]['exploration_rmse']} for r in ranked[:3]],
            'recent_failures':[{'execution':r['sequence'],'source':Path(r['source']).read_text(encoding='utf-8'),'stderr':r['stderr']} for r in executor.records[-3:] if r['returncode']],
            'search_observations':observations}
        payload={'request':{'model':'gpt-6-luna','config':{'system_instruction':instruction or INSTRUCTION},
            'contents':[{'role':'user','parts':[{'text':json.dumps(context,ensure_ascii=False)}]}]},'search_allowed':False}
        answer=await broker.exchange('model',payload)
        action=json.loads(answer['text'])
        if action.get('action')=='search':
            queries=action.get('queries',[])
            if not isinstance(queries,list) or not 1<=len(queries)<=2:raise ValueError('搜索动作需要一至两条查询')
            for query in queries:observations.append({'query':query,'result':await broker.exchange('search',{'query':query,'purpose':'自主项目研究的公开查询'})})
        elif action.get('action') in ('experiment','reference'):
            if action['action']=='reference':
                name=action.get('name')
                if not isinstance(name,str) or name not in references or 'python_source' in action:
                    raise ValueError('参考动作只能执行登记源码，不能指定路径或替换其内容')
                # 原始响应仍由broker保存；台账另存实际解析出的代码，保证假设和执行身份一致。
                action={**action,'python_source':references[name]}
            if not action.get('hypothesis') or not isinstance(action.get('python_source'),str):raise ValueError('源码试验缺少假设或程序')
            executor(action['python_source'],work,'candidate.py',90)
        elif action.get('action')=='finish':
            selected=next((r for r in successful if r['sequence']==action.get('execution')),None)
            if selected is None or not action.get('reason'):raise ValueError('只能提名实际成功的已执行程序，并说明结束理由')
            candidate=choose_candidate(selected,action.get('candidate'))
            events.append(action);atomic_json(Path(workspace)/'project-decisions.json',events)
            # 返回选中的真实权重，不改写完整执行记录中的其他候选。
            return {**selected,'candidates':[candidate],
                    'selection_mode':'explicit_candidate' if action.get('candidate') is not None else 'legacy_last_candidate'}
        else:raise ValueError('未知自主研究动作')
        events.append(action);atomic_json(Path(workspace)/'project-decisions.json',events)
