"""动作 JSON 的最小格式修复；不补字段、不改源码、不推断研究选择。"""
import json


def normalize_action_json(text):
    """仅在文件末尾少一个对象结束括号且补齐后合法时修复。

    其他语法错误交还调用者处理。返回原始或修复后的文本及修复原因，
    方便传输层同时保留原文和实际投递内容。
    """
    reason=None
    try:
        value=json.loads(text)
    except json.JSONDecodeError as error:
        if error.pos!=len(text):raise
        candidate=text.rstrip()+'}'
        value=json.loads(candidate)
        text=candidate
        reason='仅补动作JSON末尾的一个对象结束括号'
    if not isinstance(value,dict):raise ValueError('动作响应必须是JSON对象')
    return text,reason
