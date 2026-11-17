"""固定作者实现的两处可复现实错修正；不修改保存的作者原件。"""
import ast


def submission_condition(original, agent_name, raw_code):
    """任务接口已负责生成 CSV；真实产物仍由执行器在运行后核验。"""
    if original(agent_name, raw_code):
        return True
    if not agent_name.startswith('submission') or 'exit()' in raw_code:
        return False
    try:
        tree = ast.parse(raw_code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'submit_model' for node in ast.walk(tree))


def complete_refinement_selection(original, callback_context):
    """原图先执行初案，再执行 N 次改进；选择必须包括实际完成的第 N 项。"""
    state = callback_context.state
    task_id = callback_context.agent_name.split('_')[-1]
    step = state.get(f'refine_step_{task_id}', 0)
    configured = state.get('inner_loop_round', 2)
    completed = state.get(f'inner_iter_{task_id}', 0)
    count = max(configured, completed)
    for index in range(count):
        result = state.get(f'train_code_improve_exec_result_{index}_{step}_{task_id}', {})
        if 'score' not in result:
            raise ValueError('不能把没有实际执行成绩的改进加入作者选择')
    # 原回调同步执行，临时改变遍历上界后立即恢复；其比较与状态更新原样保留。
    state['inner_loop_round'] = count
    try:
        return original(callback_context)
    finally:
        state['inner_loop_round'] = configured
