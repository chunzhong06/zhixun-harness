"""同任务接口与预算下，自主选择检索、完整源码试验和结束的项目驱动。"""
import json
from hashlib import sha256
from pathlib import Path

from harness.adapters.providers.research_broker import ACTOR
from harness.research.research_ledger import history_entries
from harness.research.candidate_selection import candidate_id, choose_candidate
from harness.adapters.storage.local import atomic_json


INSTRUCTION='''你是Harness的现场研究驱动。目标是改进此任务的完整后续交付；只按本请求的合法资料和真实返回结果判断。
可自行选择检查什么机制、采用何种模型或组合、下一次试验和何时结束。不要为了用完额度继续调用。
你的输入、Python执行接口、搜索能力、数据时间边界和资源上限与完整作者对照相同。已有较强源码在task_description中；available_references列出的源码可以原样执行，先取得真实参照再决定怎样改动。也可提出新的完整实现。
返回纯JSON，选择以下一种动作：
1. {"action":"experiment","hypothesis":"中文可检验判断","python_source":"完整单文件Python程序"}。程序遵守任务说明，通过evaluate_model或submit_model保存实际模型；一次程序可以批量评估多个有明确标签的候选，并复用同一次合法训练得到的部件。外层会返回每个候选的ID和真实探索分数。可打印训练资料检查和模型诊断，但不能自行读取外层答案或伪造评分。
2. {"action":"search","queries":["公开查询"]}，一次至多两项，搜索结果将原样回传。
3. {"action":"finish","execution":已成功执行的编号,"candidate":"该执行中的候选ID，例如candidate-001","reason":"结束并提名该实际模型的理由"}。execution和candidate必须来自本轮available_candidates。require_explicit_candidate默认为true，此时必须显式给出candidate；调用方显式设为false的历史接口允许省略，沿用该程序最后保存的候选并记录legacy_last_candidate。任务说明中的参考成绩不是本轮已执行的候选。标签和候选生成顺序应在同一冻结程序的发布补训中保持。
4. {"action":"reference","name":"available_references中的名字","hypothesis":"运行该既有方案需要核查什么"}。外层会原样执行已提供源码并返回真实结果；不用重新抄写实现，也不能在此动作中另附python_source。该动作同样消耗一次模型调用及一次程序执行，不能把参考源码当作已测成绩。
每个新实验都会真实执行，失败、超时和退步也消耗预算。两个探索周的真实RMSE用于比较；最后确认周不返回，提名完成后由共同发布程序决定是否采用。新有效方案可以成为下一步改动的起点，保留未改动的有效部分。最后阶段会按冻结源码重新拟合合法发布权重；确认后的训练内成绩不会重新供你选择。
无效动作会作为action_feedback回传，原回复仍计入实际调用。只能使用剩余预算纠正；到达连续无效动作上限或预算耗尽时停止，不会自动训练或替你选择模型。
'''


class InvalidAction(ValueError):
    """仅表示已收到的模型回复无效；服务或执行器异常不在此处重试。"""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ResearchControlError(RuntimeError):
    def __init__(self, code, message, details):
        super().__init__(message)
        self.code, self.details = code, details


def _remaining_calls(broker):
    return broker.max_calls-broker.prior_model_invocations-broker.calls+broker.reused_model_calls


def _available_candidates(successful):
    """仅列出能被执行号与候选ID唯一定位的真实产物，不加入参考文字中的结果。"""
    available = []
    for record in successful:
        if sum(r['sequence'] == record['sequence'] for r in successful) != 1:
            continue
        ids = [candidate_id(candidate) for candidate in record['candidates']]
        available.extend({'execution': record['sequence'], 'candidate': key, 'label': candidate['label']}
                         for candidate, key in zip(record['candidates'], ids) if ids.count(key) == 1)
    return available


def _validate_action(answer, successful, references, *, require_explicit_candidate=True):
    if not isinstance(answer, dict) or not isinstance(answer.get('text'), str):
        raise InvalidAction('response_text_missing', '回复必须包含字符串text中的纯JSON动作')
    def reject_constant(value):
        raise ValueError('JSON不允许非有限常量：' + value)
    try:
        action = json.loads(answer['text'], parse_constant=reject_constant)
    except ValueError as exc:
        if answer.get('finish_reason') == 'length':
            raise InvalidAction('truncated_response',
                '模型输出达到长度上限，JSON未完整返回。请在剩余预算内返回更短动作，'
                '可直接引用available_references；不补写或执行被截断程序。') from exc
        raise InvalidAction('invalid_json', '回复不是有效纯JSON：' + str(exc)) from exc
    if not isinstance(action, dict):
        raise InvalidAction('action_not_object', '动作须为JSON对象，不能返回列表、字符串或null')
    kind = action.get('action')
    if kind == 'search':
        queries = action.get('queries')
        if (not isinstance(queries, list) or not 1 <= len(queries) <= 2
                or any(not isinstance(query, str) or not query.strip() for query in queries)):
            raise InvalidAction('invalid_search_queries', '搜索动作需要一至两条非空字符串查询')
    elif kind in ('experiment', 'reference'):
        if kind == 'reference':
            name = action.get('name')
            if not isinstance(name, str) or name not in references or 'python_source' in action:
                raise InvalidAction('invalid_reference', '参考动作只能指定available_references中的名称，不能替换源码')
            action = {**action, 'python_source': references[name]}
        if (not isinstance(action.get('hypothesis'), str) or not action['hypothesis'].strip()
                or not isinstance(action.get('python_source'), str) or not action['python_source'].strip()):
            raise InvalidAction('invalid_experiment', '源码试验须有非空字符串假设和完整程序')
    elif kind == 'finish':
        selected = [r for r in successful if type(action.get('execution')) is int
                    and r['sequence'] == action['execution']]
        if len(selected) != 1:
            raise InvalidAction('finish_unknown_execution',
                'finish.execution未唯一对应本轮成功执行；参考成绩不能代替实际执行，请核对available_candidates')
        if not isinstance(action.get('reason'), str) or not action['reason'].strip():
            raise InvalidAction('finish_reason_missing', 'finish必须说明提名当前实际候选的理由')
        if require_explicit_candidate and (not isinstance(action.get('candidate'), str) or not action['candidate']):
            raise InvalidAction('finish_candidate_missing', 'finish须显式给出available_candidates中的候选ID，不自动补选')
        try:
            candidate = choose_candidate(selected[0], action.get('candidate'))
        except ValueError as exc:
            raise InvalidAction('finish_invalid_candidate', str(exc) + '；请核对available_candidates') from exc
        return action, (selected[0], candidate)
    else:
        raise InvalidAction('unknown_action', 'action须为experiment、reference、search或finish')
    return action, None


async def run_free_research(task_description,task,workspace,executor,broker,*,instruction=None,reference_sources=None,resume=False,copy_inputs=True,max_consecutive_invalid_actions=3,require_explicit_candidate=True,program_seconds=90):
    """无效回复只在原预算内回传纠错；历史缺省候选语义须由调用方显式开启。"""
    if type(max_consecutive_invalid_actions) is not int or max_consecutive_invalid_actions < 1:
        raise ValueError('连续无效动作上限必须为正整数')
    if type(require_explicit_candidate) is not bool:
        raise ValueError('require_explicit_candidate必须为布尔值')
    work=Path(workspace)/'project';(work/'input').mkdir(parents=True,exist_ok=resume)
    import shutil
    # 原生续训执行器直接绑定宿主task_dir，不读取此副本；十轮运行不必重复复制训练数组。
    for p in Path(task).iterdir() if copy_inputs else ():
        target=work/'input'/p.name
        if target.exists():
            if not resume or p.read_bytes()!=target.read_bytes():raise ValueError('恢复工作区的任务输入变化')
        else:shutil.copy2(p,target)
    observations=[];events=[];invalid_feedback=[];consecutive_invalid=0
    def stop(code, message, successful):
        details = {'action': 'controller_stopped', 'error_code': code, 'error': message,
            'model_calls': broker.calls, 'remaining_model_invocations': _remaining_calls(broker),
            'consecutive_invalid_actions': consecutive_invalid,
            'available_candidates': _available_candidates(successful),
            'last_invalid_action': invalid_feedback[-1] if invalid_feedback else None}
        events.append(details)
        atomic_json(Path(workspace)/'project-decisions.json', events)
        raise ResearchControlError(code, message, details)
    references=dict(reference_sources or {})
    # 仅宿主显式准备的任务包开放此能力；旧任务和官方任务缺省不扩充权限。
    access_file=Path(task)/'feedback-training.json'
    if access_file.exists() and json.loads(access_file.read_bytes()).get('enabled') is True:
        from harness.adapters.research.feedback_task import describe_training_access,reference_sources as feedback_sources
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
        if _remaining_calls(broker) <= 0:
            stop('model_budget_exhausted', '模型调用预算已耗尽，尚未收到合法提名；已有候选保留，本驱动不代选', successful)
        best={r['sequence']:min(r['candidates'],key=lambda c:c['score']) for r in successful}
        ranked=sorted(successful,key=lambda r:best[r['sequence']]['score'])
        context={'task_description':task_description,'remaining_model_invocations':_remaining_calls(broker),
            'available_candidates':_available_candidates(successful),
            'require_explicit_candidate':require_explicit_candidate,
            'action_feedback':invalid_feedback[-max_consecutive_invalid_actions:],
            'consecutive_invalid_actions':consecutive_invalid,
            'max_consecutive_invalid_actions':max_consecutive_invalid_actions,
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
        try:
            action,selection=_validate_action(answer,successful,references,
                                               require_explicit_candidate=require_explicit_candidate)
        except InvalidAction as exc:
            consecutive_invalid+=1
            # 原回复与费用归broker记录；此处保留无效文本，反馈不创造执行号或候选。
            failure={'action':'invalid_response','error_code':exc.code,'error':str(exc),
                'response_text':answer.get('text') if isinstance(answer,dict) and isinstance(answer.get('text'),str) else None,
                'finish_reason':answer.get('finish_reason') if isinstance(answer,dict) else None,
                'model_calls':broker.calls,'remaining_model_invocations':_remaining_calls(broker),
                'consecutive_invalid_actions':consecutive_invalid}
            invalid_feedback.append(failure);events.append(failure)
            atomic_json(Path(workspace)/'project-decisions.json',events)
            if _remaining_calls(broker)<=0:
                stop('model_budget_exhausted','模型调用预算已耗尽，最后回复无效：'+str(exc)+'；不自动补选候选',successful)
            if consecutive_invalid>=max_consecutive_invalid_actions:
                stop('invalid_action_limit','连续无效动作达到上限，最后错误：'+str(exc),successful)
            continue
        consecutive_invalid=0
        # 在产生执行/检索副作用前保存已验证的原动作，基础设施异常保持显式失败。
        events.append(action);atomic_json(Path(workspace)/'project-decisions.json',events)
        if action.get('action')=='search':
            for query in action['queries']:observations.append({'query':query,'result':await broker.exchange('search',{'query':query,'purpose':'自主项目研究的公开查询'})})
        elif action.get('action') in ('experiment','reference'):
            executor(action['python_source'],work,'candidate.py',program_seconds)
        elif action.get('action')=='finish':
            selected,candidate=selection
            # 返回选中的真实权重，不改写完整执行记录中的其他候选。
            return {**selected,'candidates':[candidate],
                    'selection_mode':'explicit_candidate' if action.get('candidate') is not None else 'legacy_last_candidate'}
