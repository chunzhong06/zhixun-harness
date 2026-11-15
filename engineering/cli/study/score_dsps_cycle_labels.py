"""在相同样本、相同核分类器上分开核验原生决策、默认概率和按整组误差校准的预测。"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from score_dsps_cycle_transfer import DEST as PRIOR, PREVIOUS, SOURCE, read, acquire, LocalAcceptance, LocalStore, atomic_json, digest, tally

DEST=PRIOR.parent/"dsps-cycle-labels"
PRIMARY="整组核验后按置信度选择"
FEATURES=("进入与稳定期差周期内相对","进入与稳定期差多负载","进入与稳定期差单负载",
          "重复周期波动周期内相对","完整周期原读数单负载","同器件空气相对周期单负载")


def outputs(x,tr,te,y,feature,pooled):
    src=x[feature,0]; dest=x[feature,11]
    train=np.vstack((dest[tr],src[tr])) if pooled else dest[tr]
    target=np.tile(y[tr],2) if pooled else y[tr]
    est=make_pipeline(StandardScaler(),SVC(C=4,gamma="scale",probability=True,random_state=17)).fit(train,target)
    model=est[-1]
    if model.classes_.tolist()!=[1,2,3]:raise ValueError("核模型输出类别顺序不同")
    prob=est.predict_proba(dest[te]);scores=est.decision_function(dest[te]);native=est.predict(dest[te])
    return prob,scores,native,model.probA_.tolist()


def run():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("类别审计已经完成")
        ids,group,y,x=acquire()
        previous=read(PRIOR/"scores.json")
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        events=[];truth=[];cols={};details=[]
        for rep,parent in zip(np.unique(group),previous["requests"],strict=True):
            tr=group!=rep;te=~tr
            old=read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(rep),old])!=parent or old["ids"]!=ids[te].tolist():raise ValueError("先前回执错位")
            pred=dict(old["预测"]);bank={};rates={};foldlog={}
            for feat in FEATURES:
                for pool in (False,True):
                    arm=feat+("/双器件合训/核" if pool else "/目标固定/核")
                    prob,margins,native,coef=outputs(x,tr,te,y,feat,pool)
                    if not np.array_equal(1+prob.argmax(1),pred[arm]):
                        raise ValueError("相同核权重的默认概率未复现")
                    inner,ii,inner_y=[],[],[]
                    for other in np.unique(group[tr]):
                        ev=group==other
                        pp,mm,nn,_=outputs(x,tr&~ev,ev,y,feat,pool)
                        inner.append(mm);ii.append(pp);inner_y.extend(y[ev])
                    hh=np.vstack(inner); rawp=np.vstack(ii);yy=np.asarray(inner_y)
                    calibrator=LogisticRegression(C=1,max_iter=500).fit(hh,yy)
                    cal=calibrator.predict_proba(margins)
                    pred[arm+"/原生"]=native.tolist()
                    pred[arm+"/留组校准"]=(1+cal.argmax(1)).tolist()
                    # 再留一个源重复评价校准器本身，避免用同一组预测拟合又评分。
                    n=len(inner[0]); cval=np.empty_like(hh)
                    for left,right in ((np.arange(n),np.arange(n,len(hh))),(np.arange(n,len(hh)),np.arange(n))):
                        cval[right]=LogisticRegression(C=1,max_iter=500).fit(hh[left],yy[left]).predict_proba(hh[right])
                    for key,history,current in ((arm,rawp,prob),(arm+"/留组校准",cval,cal)):
                        rates[key]=f1_score(yy,1+history.argmax(1),average="macro")
                        bank[key]=current
                    foldlog[arm]={"原生与默认概率意见相同":int(np.sum(native==1+prob.argmax(1))),"总数":len(native),
                                  "两两概率校准系数":coef}
            best=max(rates,key=rates.get)
            keys=list(bank); prior=np.array([max(.001,rates[k])**4 for k in keys]); z=np.stack([bank[k] for k in keys],axis=1)
            conf=np.diff(np.sort(z,axis=2)[:,:,1:],axis=2).squeeze(2)
            weight=prior[None,:]*(.05+conf)
            pred[PRIMARY]=(1+(z*weight[:,:,None]).sum(1).argmax(1)).tolist()
            pred["留源核验固定权重"]=(1+(z*prior[None,:,None]).sum(1).argmax(1)).tolist()
            pred["留源核验选择单一读法"]=(1+bank[best].argmax(1)).tolist()
            # 论文PDS在最佳新特征上同样重训：目标每个连续传感器区块回归到源端再分类。
            from sklearn.linear_model import Ridge
            f=FEATURES[0]
            a=x[f,0].reshape(len(y),20,8).transpose(0,2,1)
            b=x[f,11].reshape(len(y),20,8).transpose(0,2,1)
            A,B=a.reshape(len(y),-1),b.reshape(len(y),-1)
            sa,sb=StandardScaler().fit(A[tr]),StandardScaler().fit(B[tr])
            source=sa.transform(A[tr]); target=sb.transform(B[tr]); service=sb.transform(B[te])
            blocks=[Ridge(alpha=1).fit(target[:,j:j+20],source[:,j:j+20]).predict(service[:,j:j+20]) for j in range(0,160,20)]
            svm=SVC(C=4,gamma="scale",probability=True,random_state=17).fit(source,y[tr])
            mapped=np.column_stack(blocks)
            pred["论文分段校准周期读法原生"]=svm.predict(mapped).tolist()
            pred["论文分段校准周期读法概率"]=(1+svm.predict_proba(mapped).argmax(1)).tolist()
            payload={"ids":ids[te].tolist(),"预测":pred,"源核验选择":best,"源留组分数":rates}
            key=digest([int(rep),payload]);events.append(key)
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("发放失败")
            sent=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),sent])!=key:raise ValueError("发放身份不符")
            truth.extend(y[te]);details.append(foldlog)
            for k,v in sent["预测"].items():cols.setdefault(k,[]).extend(v)
        scores={k:tally(np.array(truth),np.array(v)) for k,v in cols.items()}
        fixed=max([previous["decision"]["固定"],"留源核验固定权重","留源核验选择单一读法",
                   *[k for k in scores if k.endswith(("/原生","/留组校准"))]],key=lambda k:scores[k]["F1"])
        paper=max(["直接标准化","分段直接标准化","论文分段校准周期读法原生","论文分段校准周期读法概率"],key=lambda k:scores[k]["F1"])
        delta=scores[PRIMARY]["F1"]-max(scores[fixed]["F1"],scores[paper]["F1"])
        decision={"主候选":PRIMARY,"固定":fixed,"论文":paper,"领先":delta,"进独立":delta>=.01 and scores[PRIMARY]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"requests":events,"scores":scores,"各折核对":details,"decision":decision})
        print(json.dumps({"核对":{k:scores[k] for k in (PRIMARY,fixed,paper,previous["decision"]["固定"],previous["decision"]["固定"]+"/原生")},"判定":decision},ensure_ascii=False))


if __name__=="__main__":run()
