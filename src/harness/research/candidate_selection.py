"""用执行号和候选ID绑定模型；发布补训保持同一程序中的候选身份。"""
from pathlib import Path


def candidate_id(candidate):
    return Path(candidate['path']).stem


def choose_candidate(record, reference=None):
    """显式ID只能指向本次成功执行保存的模型；省略时兼容历史最后候选语义。"""
    if record['returncode'] != 0 or not record['candidates']:
        raise ValueError('只能选择实际成功执行的已保存候选')
    if reference is None:
        return record['candidates'][-1]
    if not isinstance(reference, str):
        raise ValueError('候选ID必须使用请求中列出的字符串')
    matches = [candidate for candidate in record['candidates'] if candidate_id(candidate) == reference]
    if len(matches) != 1:
        raise ValueError('候选ID不属于该执行或不能唯一确定模型')
    return matches[0]


def match_publication(selected, record):
    """同一冻结程序补训后按ID、标签和反馈设置核对，禁止自动换为最后一个模型。"""
    published = choose_candidate(record, candidate_id(selected))
    if published['label'] != selected['label'] or published['feedback'] != selected['feedback']:
        raise ValueError('发布补训改变了选中候选的标签或反馈设置')
    return published
