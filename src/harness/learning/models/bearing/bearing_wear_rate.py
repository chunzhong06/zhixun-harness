"""从最近的实测变化速度外推到过去的失效振幅，用已到达的无标签波形比较外推速度。"""
import numpy as np

WINDOWS = (5,20,80)
LIMITS = (1.,2.,4.)
PRIMARY = "由本台后续已到达波形验证速度再预测寿命"


def amplitude(spectrum):
    return np.log1p(np.sqrt((np.asarray(spectrum,dtype=float)**2).mean(axis=1)))


class WearRate:
    def fit(self, past):
        self.terminal = float(np.median([np.median(amplitude(u.spectral)[-3:]) for u in past]))
        self.scale = float(np.quantile([u.durations["capped"] for u in past],.85))
        self.ratio = np.sum([u.durations["calendar"] for u in past])/np.sum([u.durations["capped"] for u in past])
        return self

    def predict(self, spectra, active, calendar):
        """每到一个新点，只给以前已发出的振幅预测评分。标签只来自已经结束的装机设备。"""
        y = amplitude(spectra)
        limit = self.scale*np.array(LIMITS)
        smooth = np.array([np.median(y[max(0,i-2):i+1]) for i in range(len(y))])
        loss = np.zeros(len(WINDOWS))
        slopes = np.zeros(len(WINDOWS))
        banks = np.zeros((len(y),len(WINDOWS),len(LIMITS)))
        chosen = []
        for i,t in enumerate(active):
            if i:
                # 预测的是真实到达的波形，而非随未来寿命标签变化的代理答案。
                expected = smooth[i-1]+slopes*(t-active[i-1])
                loss = .9*loss + .1*np.minimum(np.abs(y[i]-expected),2.)
            for k,w in enumerate(WINDOWS):
                x = active[max(0,i-w+1):i+1]
                v = smooth[max(0,i-w+1):i+1]
                center=x-x.mean()
                slopes[k]=np.dot(center,v-v.mean())/max(np.dot(center,center),1e-8)
                if smooth[i] >= self.terminal:
                    banks[i,k] = 0.
                elif slopes[k] > 0:
                    banks[i,k] = np.minimum(limit,(self.terminal-smooth[i])/slopes[k])
                else:
                    banks[i,k] = limit
            # 第一个读数和几种斜率完全相同，用中等窗作为起点。
            picked=1 if np.ptp(loss)<1e-12 else int(loss.argmin())
            chosen.append(picked)
        choices=np.asarray(chosen)
        multiplier=self.ratio if calendar else 1.
        banks *= multiplier
        cols={f"固定速度窗{w}上限{limit:g}":banks[:,j,k] for j,w in enumerate(WINDOWS) for k,limit in enumerate(LIMITS)}
        for k,limit in enumerate(LIMITS):
            cols[f"固定三窗中位上限{limit:g}"]=np.median(banks[:,:,k],axis=1)
        cols[PRIMARY]=banks[np.arange(len(y)),choices,1]
        return cols,choices
