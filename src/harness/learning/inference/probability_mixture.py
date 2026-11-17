"""分类概率算术组合及对父网logits的导数，不读取标签或改变教师。"""
import numpy as np


def arithmetic_output(parent_logits,teacher_probabilities,parent_weight):
    """返回log(组合概率)及[n,输出类别,父类别]的Jacobian。

    算术概率组合的梯度依赖两路概率，不能沿用几何组合的常数系数。
    训练端应将损失对输出的梯度右乘本Jacobian，仍只更新原父网络。
    """
    logits=np.asarray(parent_logits,dtype=np.float64)
    teacher=np.asarray(teacher_probabilities,dtype=np.float64)
    alpha=float(parent_weight)
    if (logits.ndim!=2 or logits.shape!=teacher.shape or logits.shape[1]<2
            or not np.isfinite(logits).all() or not np.isfinite(teacher).all()
            or np.any(teacher<0) or not np.isfinite(alpha) or not 0<alpha<=1):
        raise ValueError('算术分类组合需要同形状有限输入及(0,1]内父权重')
    totals=teacher.sum(axis=1,keepdims=True)
    if np.any(totals<=0) or not np.allclose(totals,1.,rtol=1e-5,atol=1e-7):
        raise ValueError('教师概率各行须归一化')
    teacher=teacher/totals
    parent=np.exp(logits-logits.max(axis=1,keepdims=True))
    parent/=parent.sum(axis=1,keepdims=True)
    mixture=alpha*parent+(1-alpha)*teacher
    safe=np.maximum(mixture,1e-30)
    factor=alpha*parent/safe
    # 数值下限分支为常数；避免极端下溢时返回不属于实际输出的导数。
    factor=np.where(mixture>1e-30,factor,0.)
    jacobian=factor[:,:,None]*(np.eye(logits.shape[1])[None,:,:]-parent[:,None,:])
    return np.log(safe),jacobian
