"""从已到期参考学习当前可见读数下的候选误差，先发放再取得该时刻答案。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def select(predictions, context, reference, clock, service_start, initial, rule, memory_days=28):
    predictions=np.asarray(predictions,dtype=float)
    reference=np.asarray(reference,dtype=float)
    clock=np.asarray(clock)
    features=np.column_stack((context,predictions,predictions-predictions.mean(axis=1,keepdims=True)))
    if (features.ndim!=2 or len(features)!=len(reference) or reference.shape!=clock.shape
            or not np.isfinite(features).all() or np.any(np.diff(clock)<=0)):
        raise ValueError("候选上下文或参考时钟不一致")
    if rule not in ("global-hard","global-soft","tree-hard","tree-soft","tree-half-global"):
        raise ValueError("未知的候选误差选择方法")
    if predictions.shape[1]==1:
        return predictions[:,0].copy(),np.zeros(len(clock),dtype=int),[]
    output=np.empty(len(clock))
    identities=np.full(len(clock),initial,dtype=int)
    arrived=[]
    model=None
    mean_loss=None
    cached_risk=None
    cached_start=0
    fits=[]
    for i,now in enumerate(clock):
        # 到期参考和对应的候选影子预测构成训练资料，当前答案在输出之后加入。
        arrived=[j for j in arrived if clock[j]>now-memory_days*24]
        if now>=service_start and (now-service_start)%24==0 and arrived:
            indices=np.asarray(arrived)
            loss=(predictions[indices]-reference[indices,None])**2
            mean_loss=loss.mean(axis=0)
            model=None
            if rule.startswith("tree") and len(indices)>=24:
                model=ExtraTreesRegressor(n_estimators=48,max_depth=4,min_samples_leaf=12,
                                          random_state=17,n_jobs=1)
                model.fit(features[indices],loss)
                fits.append({"publish_hour":int(now),"last_reference":int(clock[indices[-1]]),"rows":len(indices)})
        if now>=service_start and (now-service_start)%24==0 and model is not None:
            # 一天内权重固定，按行独立预测可以合批计算，各行仍只依赖自己的可见输入。
            cached_risk=model.predict(features[i:i+24])
            cached_start=i
        if now<service_start or mean_loss is None:
            output[i]=predictions[i,initial]
        else:
            risk=cached_risk[i-cached_start] if model is not None else mean_loss
            if rule.endswith("hard"):
                selected=int(np.argmin(risk))
                output[i]=predictions[i,selected]
                identities[i]=selected
            else:
                weights=1/np.maximum(risk,1e-6)
                weights/=weights.sum()
                if rule=="tree-half-global":
                    shared=1/np.maximum(mean_loss,1e-6)
                    weights=(weights+shared/shared.sum())/2
                output[i]=float(predictions[i]@weights)
                identities[i]=-1
        if np.isfinite(reference[i]):
            arrived.append(i)
    return output,identities,fits
