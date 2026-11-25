"""按任务公开声明的评价协议发布；比赛公开选案与研究独立确认分开。"""
import math
import numpy as np


def balanced_brier(labels, probabilities):
    """各个当轮可见类别等权，衡量预测给真实标签及其他标签分配的概率。"""
    y = np.asarray(labels, dtype=int) - 1
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 6) or not len(y) or np.any((y < 0) | (y >= 6)):
        raise ValueError("气体概率与当轮标签不匹配")
    square = np.sum((p - np.eye(6)[y])**2, axis=1)
    return float(np.mean([square[y == c].mean() for c in np.unique(y)]))


def select_public_validation(parent, candidates, *, validation_loss, probability_loss=None):
    """对真实成功候选重算同一完整公开验证损失，平分保留先前版本。

    candidates为按执行顺序的(role, candidate)，不以模型最后提名缩小范围。
    调用者只传已通过权重身份和执行核验的候选；此处记录选择全过程，
    不把公开选案评价写作独立确认或隐藏测试。
    """
    incumbent = parent
    best = float(validation_loss(parent))
    if not math.isfinite(best):
        raise ValueError('父版公开验证损失必须有限')
    seen = {parent['weights_sha256']}
    reviews = []
    confidence = probability_loss(parent) if probability_loss else None
    if probability_loss:
        candidates = sorted(candidates, key=lambda row: row[0] != 'reference')
    for role, candidate in candidates:
        key = candidate['weights_sha256']
        if key in seen:
            continue
        seen.add(key)
        loss = float(validation_loss(candidate))
        accepted = math.isfinite(loss) and loss < best
        reason = 'public_validation_improved' if accepted else 'no_public_validation_gain' if math.isfinite(loss) else 'nonfinite_loss'
        extra = {}
        if probability_loss:
            score = probability_loss(candidate)
            extra = {'balanced_brier_before':confidence, 'balanced_brier_after':score}
            if accepted and not score <= confidence:
                accepted, reason = False, 'balanced_brier_increased'
            if accepted:
                confidence = score
        reviews.append({'role': role, 'candidate_sha256': key,
                        'incumbent_sha256': incumbent['weights_sha256'], 'accepted': accepted,
                        'validation_before': best, 'validation_after': loss if math.isfinite(loss) else None,
                        'reason': reason, **extra})
        if accepted:
            incumbent, best = candidate, loss
    return incumbent, reviews


def protect_incumbent(parent, reference, nomination, *, confirmation_loss):
    """三者都必须是实际保存候选，score是共同探索损失，越低越好。

    固定参考在研究开始前选定并执行。全部研究结束后才按固定顺序确认，
    最多检查参考和提名两项；较差提名、研究失败或确认退步不撤销参考收益。
    这是一项显式发布策略，不能把采用参考的收益归因于研究驱动。
    """
    incumbent = parent
    reviews = []
    confirmed = {}

    def loss(candidate):
        key = candidate['weights_sha256']
        if key not in confirmed:
            value = float(confirmation_loss(candidate))
            if not math.isfinite(value):
                raise ValueError('确认损失必须有限')
            confirmed[key] = value
        return confirmed[key]

    for role, candidate in (('reference', reference), ('research', nomination)):
        if candidate is None:
            continue
        if not math.isfinite(candidate['score']) or not math.isfinite(incumbent['score']):
            raise ValueError('探索损失必须有限')
        review = {'role': role, 'candidate_sha256': candidate['weights_sha256'],
                  'incumbent_sha256': incumbent['weights_sha256'], 'accepted': False,
                  'exploration_before': incumbent['score'], 'exploration_after': candidate['score']}
        if candidate['weights_sha256'] == incumbent['weights_sha256']:
            review['reason'] = 'already_incumbent'
        elif candidate['score'] >= incumbent['score']:
            review['reason'] = 'no_exploration_gain'
        else:
            before, after = loss(incumbent), loss(candidate)
            review.update(confirmation_before=before, confirmation_after=after)
            review['accepted'] = after < before
            review['reason'] = 'both_losses_improved' if review['accepted'] else 'no_confirmation_gain'
            if review['accepted']:
                incumbent = candidate
        reviews.append(review)
    return incumbent, reviews
