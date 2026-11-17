"""Qwen官方模板的纯文本、无工具分支；以官方非思考前缀调用PyTRiO续写API。

依据Qwen/Qwen3.8-27B的chat_template.jinja，避免兼容聊天接口不支持模板参数。
源模板保存在references/retrieval/platform-contract-20261115/qwen-chat-template.jinja。
"""
def render(messages):
    result=[]
    for index,message in enumerate(messages):
        role=message['role'];content=message['content'].strip()
        if role not in ('system','user','assistant') or (role=='system' and index!=0):
            raise ValueError('仅支持纯文本标准角色')
        if role=='assistant':content='<think>\n\n</think>\n\n'+content
        if content:result.append('<|im_start|>'+role+'\n'+content+'<|im_end|>\n')
    return ''.join(result)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
