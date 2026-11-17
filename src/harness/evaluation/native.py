"""原生任务评价和合法探索诊断。"""
import numpy as np
def measure(task, labels, prediction):
    labels = np.asarray(labels)
    if task == 'gas':
        probability = np.asarray(prediction)
        if probability.shape != (len(labels), 6) or not np.isfinite(probability).all():
            raise ValueError('气体概率输出无效')
        predicted = probability.argmax(axis=1) + 1
        if not np.isin(labels, np.arange(1, 7)).all():
            raise ValueError('气体标签须为官方六类编号')
        # 分轮样本中可能缺少稀有类；各轮仍按官方六类等权，空分母计0。
        values = []
        for c in range(1, 7):
            denominator = np.sum(predicted == c) + np.sum(labels == c)
            values.append(2 * np.sum((predicted == c) & (labels == c)) / denominator if denominator else 0.)
        score = float(np.mean(values))
        return {'score': score, 'loss': 1 - score, 'metric': 'Macro-F1'}
    prediction = np.asarray(prediction)
    if prediction.shape != labels.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('寿命输出无效')
    score = float(np.sqrt(np.mean((prediction - labels) ** 2)))
    return {'score': score, 'loss': score, 'metric': 'RMSE'}

def exploration_diagnostics(task, labels, prediction, groups):
    """只汇总允许观察的探索标签，给驱动定位类别或设备误差的依据。"""
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    if task == 'gas':
        predicted = prediction.argmax(axis=1) + 1
        confusion = [[int(np.sum((labels == actual) & (predicted == guessed)))
                      for guessed in range(1, 7)] for actual in range(1, 7)]
        return {'labels': list(range(1, 7)), 'confusion_true_rows_prediction_columns': confusion}
    group_names = np.asarray([str(group) for group in groups])
    result = []
    for name in sorted(set(group_names)):
        selected = group_names == name
        error = prediction[selected] - labels[selected]
        result.append({'group': name, 'rows': int(selected.sum()),
                       'rmse': float(np.sqrt(np.mean(error ** 2))),
                       'prediction_minus_target_mean': float(np.mean(error)),
                       'target_mean': float(np.mean(labels[selected]))})
    return {'by_group': result}
