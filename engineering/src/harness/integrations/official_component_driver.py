"""同一执行接口上的论文部件对照：真实诊断、可追溯知识修订与定向源码试改。"""
import json
from pathlib import Path

from harness.integrations.author_mlestar import ACTOR
from harness.runtime.storage import atomic_json

POLICIES={
    'mle_star':'采用MLE-STAR的消融定位、定向改进、保留较好解部件。先结合同条件组件成绩确定应替换的一项；实施后用真实结果决定继续该项或保持。可以实际执行消融程序。',
    'iris':'采用Iris的信息获取与知识修订部件。先识别会改变下一步选择的未知问题；证据不足时执行诊断，充分时直接试改。每次用真实观察支持、缩小或否定已有判断，避免只有方案名和分数。knowledge_update必须引用已发生事件，保留适用范围和反证。'
}

COMMON='''你是隔离的官方工业任务现场研究驱动，只使用此请求中当时可见的信息。
外层真实运行程序并计分；搜索与确认答案不提供给程序。每次可使用load_data返回的源训练标签，无标签探索和确认输入；input/task.json包含合法训练组身份。不要读取任务目录之外的信息。
返回纯JSON选择一个动作：
{"action":"experiment","hypothesis":"可检验判断","component":"改动部件","python_source":"完整单文件程序"}
{"action":"diagnose","question":"会改变什么选择","python_source":"只打印诊断结果的完整程序"}
{"action":"finish","execution":已成功的执行编号,"reason":"依据"}
实验程序from harness_task import load_data,evaluate_model，X,y,V,T=load_data()；构建任意sklearn兼容对象，fit只用X,y，最后evaluate_model(model,label='名称')。X、V、T为pandas DataFrame；每次predict仅收到一行，返回长度1数组。气体返回1至6类别；寿命返回非负剩余分钟/循环。可以自建特征、重新选择学习器或训练子集，不必只修父版残差。执行90秒、单线程，numpy/scipy/pandas/sklearn/torch CPU可用，不能联网或启动子进程。
诊断程序也可load_data，执行数据/特征/训练可靠性检查，但不调用evaluate_model，不提交候选。已有候选的真实错误分层来自外层观察。诊断与实验共享调用额度，失败也计入。
每个动作可附knowledge_update:[{"claim":"判断","scope":"适用条件","status":"active|qualified|invalidated","evidence":[已完成的执行编号]}]。原主张被反例缩小时请明确更新。只引用当前观察实际出现的编号。
最终只提名实际成功候选。外层在结束后用独立确认决定采用，再真实交付余下预测；原来已经发出的预测保持。次数用完仍未结束时，程序提名探索损失最低的实际候选。不要为了耗尽预算而继续。'''


def revise(ledger,updates,records):
    """主张只连到已完成原始执行；反证不会被后续文字摘要覆盖掉。"""
    known={r['sequence'] for r in records}
    for update in updates:
        if not isinstance(update,dict) or not all(update.get(k) for k in ('claim','scope','status','evidence')):raise ValueError('知识更新缺少主张、范围、状态或证据')
        if update['status'] not in ('active','qualified','invalidated') or not set(update['evidence'])<=known:raise ValueError('知识更新引用未知事件')
        ledger.append({**update,'revision':len(ledger)+1})


def decode_action(answer):
    """仅去除传输封装，不改动模型实际动作与源码；未知形式留作一次失败。"""
    value=answer
    for _ in range(4):
        if isinstance(value,str):value=json.loads(value)
        if isinstance(value,dict) and value.get('action') in ('experiment','diagnose','finish'):return value
        if isinstance(value,dict) and len(value)==1 and next(iter(value)) in ('text','response'):
            value=next(iter(value.values()));continue
        raise ValueError('回复缺少可识别的实际动作')
    raise ValueError('回复封装过深')


async def search(arm,description,workspace,executor,broker,limit=4,recorded_calls=0):
    work=Path(workspace);ledger=[];events=[]
    for step in range(limit):
        used=broker.prior_model_invocations+broker.calls-broker.reused_model_calls
        if step>=recorded_calls and used>=broker.max_calls:break
        successful=[r for r in executor.records if not r['returncode'] and r['candidates']]
        best=sorted(successful,key=lambda r:r['candidates'][-1]['score'])[:3]
        context={'task_description':description,'remaining_calls':limit-step if step<recorded_calls else broker.max_calls-used,
            'history':[{'execution':r['sequence'],'diagnostic':r['diagnostic'],'returncode':r['returncode'],
                'seconds':r['execution_time'],'output':r['stdout'][-12000:],'stderr':r['stderr'][-2500:],
                'metrics':[c.get('task_metrics',{}) for c in r['candidates']]} for r in executor.records],
            'best_working_programs':[{'execution':r['sequence'],'source':Path(r['source']).read_text(encoding='utf-8'),
                'loss':r['candidates'][-1]['score']} for r in best],
            'knowledge':ledger,'past_actions':events}
        ACTOR.set(arm+'_official_controller')
        try:
            answer=await broker.exchange('model',{'request':{'model':'gpt-6-luna','config':{'system_instruction':COMMON+'\n'+POLICIES[arm]},
                'contents':[{'role':'user','parts':[{'text':json.dumps(context,ensure_ascii=False)}]}]},'search_allowed':False})
            action=decode_action(answer)
        except (ValueError,KeyError,TimeoutError) as exc:
            events.append({'action':'failed_request','error':str(exc),'consumes_budget':True})
            atomic_json(work.parent/'decisions.json',{'events':events,'knowledge':ledger});continue
        # 修订只使用动作执行前已见的证据，避免把自己的计划登记成已发生实验。
        revise(ledger,action.get('knowledge_update',[]),executor.records)
        if action.get('action') in ('experiment','diagnose'):
            if not isinstance(action.get('python_source'),str):raise ValueError('缺少实际可执行程序')
            if not action.get('hypothesis' if action['action']=='experiment' else 'question'):raise ValueError('缺少实验假设或诊断问题')
            executor.diagnostic=action['action']=='diagnose'
            executor(action['python_source'],work,'action.py',90)
            executor.diagnostic=False
        elif action.get('action')=='finish':
            selected=next((r for r in successful if r['sequence']==action.get('execution')),None)
            if selected is None:raise ValueError('只能提名已成功执行的候选')
            events.append(action);atomic_json(work.parent/'decisions.json',{'events':events,'knowledge':ledger})
            return selected
        else:raise ValueError('未知研究动作')
        events.append(action);atomic_json(work.parent/'decisions.json',{'events':events,'knowledge':ledger})
    successful=[r for r in executor.records if not r['returncode'] and r['candidates']]
    chosen=min(successful,key=lambda r:r['candidates'][-1]['score'])
    atomic_json(work.parent/'budget-selection.json',{'reason':'达到共同调用上限，按真实探索损失提名','execution':chosen['sequence']})
    return chosen
