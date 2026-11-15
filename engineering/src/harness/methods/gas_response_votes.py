"""给逐批交付与研究回放共用的源/现场核分类工具。"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from harness.models.field_classification import weights


def kernel_models(source, sy, current, past_x, past_y, pool, source_svm):
    site = StandardScaler().fit(pool)
    new = site.transform(current)
    shown = site.transform(past_x) if len(past_x) else past_x
    mixed = SVC(C=4, gamma="scale", random_state=17)
    sx = StandardScaler().fit_transform(source)
    sw = weights(sy, len(sy))
    if len(past_y):
        sw = np.r_[sw, weights(past_y, len(sy))]
    mixed.fit(np.vstack((sx, shown)), np.r_[sy,past_y], sample_weight=sw)
    if len(np.unique(past_y)) >= 2:
        target = SVC(C=4, gamma="scale", class_weight="balanced", random_state=17).fit(shown,past_y)
    else:
        target = mixed
    return {"固定源核":source_svm.predict(new), "固定合训核":mixed.predict(new), "固定现场核":target.predict(new)}
