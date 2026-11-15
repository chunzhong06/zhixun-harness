"""以唯一可训练的源批拟合论文基础分类器，按源批交叉预测检查特定分歧方向。"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from harness.models.gas_views import represent


KINETIC_VIEWS=("ratio","mispaired","transient","steady","combined")


def kinetic_view(fit,query,view):
    fit = np.asarray(fit).reshape(-1,16,8)
    floor = np.maximum(1e-12,.01*np.median(np.abs(fit[:,:,0]),axis=0))
    def features(rows):
        channels = np.asarray(rows).reshape(-1,16,8)
        amplitude = np.maximum(np.abs(channels[:,:,0]),floor)
        if view=="mispaired":
            amplitude = np.roll(amplitude,1,axis=1)
        slope = channels[:,:,2:]
        rates = (slope/amplitude[:,:,None]).reshape(len(channels),96)
        steady = channels[:,:,1]
        if view=="combined":
            return np.column_stack((rates,steady))
        if view=="steady":
            return steady
        return slope.reshape(len(channels),96) if view=="transient" else rates
    bank = features(fit)
    center = np.median(bank,axis=0)
    lower,upper = np.quantile(bank,[.25,.75],axis=0)
    scale = np.maximum(1e-12,upper-lower)
    return np.clip((bank-center)/scale,-10,10),np.clip((features(query)-center)/scale,-10,10)


def classify(model, x_fit, y, query, parent_probs, recipe):
    view, mode = recipe.split("-")
    estimator = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
    if view in KINETIC_VIEWS:
        train,x = kinetic_view(x_fit,query,view)
    elif view=="rank":
        from scipy.stats import rankdata
        def ranks(values):
            # 同一测量内部排序；同读数共享平均秩，独立于其它到达测量。
            channels = np.asarray(values).reshape(-1,16,8)
            return ((rankdata(channels,method="average",axis=1)-8.5)/7.5).reshape(-1,128)
        train,x = ranks(x_fit),ranks(query)
    else:
        train = represent(x_fit,view,model.mean,model.scale)
        x = represent(query,view,model.mean,model.scale)
    learned = estimator.fit(train,y)
    proposed = learned.predict(x)
    evidence = {"train_rows":len(y), "support_vectors":len(learned.support_), "gamma":float(1/(train.shape[1]*train.var()))}
    if mode=="full":
        return proposed+1, evidence
    if mode!="gated":
        raise ValueError("未知的支持向量机使用方式")
    folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=17)
    if view in KINETIC_VIEWS:
        held = np.empty_like(y)
        for indices,validation in folds.split(x_fit,y):
            a,b = kinetic_view(x_fit[indices],x_fit[validation],view)
            held[validation] = estimator.fit(a,y[indices]).predict(b)
    else:
        held = cross_val_predict(estimator,train,y,cv=folds,n_jobs=1)
    parent_fit = model.predict(x_fit).argmax(axis=1)
    parent = parent_probs.argmax(axis=1)
    output = parent.copy()
    allowed = []
    for src in range(6):
        for dst in range(6):
            if src==dst:
                continue
            mask = (parent_fit==src)&(held==dst)
            n = int(mask.sum())
            correct = int(np.sum(y[mask]==dst))
            broken = int(np.sum(y[mask]==src))
            if n>=5 and correct>broken:
                allowed.append({"from":src+1,"to":dst+1,"support":n,"correct":correct,"broken":broken})
                output[(parent==src)&(proposed==dst)] = dst
    evidence.update(allowed=allowed,folds=5)
    return output+1,evidence
