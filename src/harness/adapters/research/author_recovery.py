"""从作者真实执行原件恢复定向改进后的状态，不用终局答案重新选案。"""
import ast
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from harness.adapters.research.author_compatibility import complete_refinement_selection


def restore_refined_state(origin, workspace, task_name, description, configuration, original_callback):
    if configuration.outer_loop_round!=1:
        raise ValueError('本恢复器只适用于已执行的一轮外层改进，不能跳过后续真实轮次')
    origin=Path(origin);workspace=Path(workspace)
    records=json.loads((origin/'executions/index.json').read_bytes())
    state=asdict(configuration)
    state.update(workspace_dir=str(workspace),task_name=task_name,task_description=description)
    audit=[]

    def verified(path):
        source=path.read_text(encoding='utf-8');tree=ast.dump(ast.parse(source),include_attributes=False)
        matches=[]
        for record in records:
            if record['returncode'] or not record['candidates']:continue
            executed=Path(record['source']).read_text(encoding='utf-8')
            if sha256(executed.encode()).hexdigest()!=record['source_sha256']:
                raise ValueError('原作者执行源码身份改变')
            if ast.dump(ast.parse(executed),include_attributes=False)==tree:matches.append(record)
        if not matches:raise ValueError('待恢复源码缺少真实成功执行：'+str(path))
        record=matches[-1]
        result={key:record[key] for key in ('returncode','stdout','stderr','execution_time')}
        result['score']=record['candidates'][-1]['score']
        return source,result,record['sequence']

    for task_id in range(1,configuration.num_solutions+1):
        old=origin/'author-workspace'/task_name/str(task_id)
        dest=workspace/task_name/str(task_id);dest.mkdir(parents=True)
        source,result,parent=verified(old/'train0.py')
        state[f'train_code_0_{task_id}']=source
        state[f'train_code_exec_result_0_{task_id}']=result
        state[f'refine_step_{task_id}']=0
        state[f'prev_ablations_{task_id}']=[];state[f'prev_code_blocks_{task_id}']=[]
        improved=[]
        for index in range(configuration.inner_loop_round+1):
            source,result,execution=verified(old/f'train0_improve{index}.py')
            state[f'train_code_improve_{index}_0_{task_id}']=source
            state[f'train_code_improve_exec_result_{index}_0_{task_id}']=result
            improved.append({'index':index,'execution':execution,'score':result['score']})
        state[f'inner_iter_{task_id}']=len(improved)
        complete_refinement_selection(original_callback,SimpleNamespace(agent_name=f'ablation_and_refine_agent_{task_id}',state=state))
        audit.append({'branch':task_id,'parent_execution':parent,'actually_executed_improvements':improved,
                      'selected_score':state[f'train_code_exec_result_1_{task_id}']['score']})
    return state,audit
