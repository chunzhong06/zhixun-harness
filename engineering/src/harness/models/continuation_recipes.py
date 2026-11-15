"""可编辑的通用续训程序；按任务类型和训练组工作，不嵌入数据集名称或验证分数。"""


REFERENCE_PROGRAMS = {
    'official-one-epoch': '''def train(session):
    session.train_epoch()
''',
    'stable-fine-tuning': '''def train(session):
    for epoch in range(12):
        session.train_epoch(learning_rate=0.001)
''',
    'balanced-fine-tuning': '''import numpy as np
def train(session):
    identity = session.targets if session.metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(4):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    'expanded-balanced-fine-tuning': '''import numpy as np
def train(session):
    metadata = session.metadata
    outputs = len(metadata['classes']) if metadata['task'] == 'classification' else 1
    capacity = min(128, (99999 - outputs) // (len(metadata['paths']) + outputs + 1))
    if metadata['hidden'] < capacity:
        session.expand_hidden(capacity)
    identity = session.targets if session.metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(4):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    'extended-balanced-fine-tuning': '''import numpy as np
def train(session):
    metadata = session.metadata
    outputs = len(metadata['classes']) if metadata['task'] == 'classification' else 1
    capacity = min(64, (99999 - outputs) // (len(metadata['paths']) + outputs + 1))
    if metadata['hidden'] < capacity:
        session.expand_hidden(capacity)
    identity = session.targets if metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(12):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    'long-balanced-fine-tuning': '''import numpy as np
def train(session):
    identity = session.targets if session.metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(12):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
}


def reference_programs():
    """调用方取得副本并冻结源码身份；现场仍须实际执行、比较和决定采用。"""
    return dict(REFERENCE_PROGRAMS)
