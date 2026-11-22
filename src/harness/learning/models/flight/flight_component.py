"""从可测站位组成的温压比和转速出发，核对健康参照与同工况重复测量。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from harness.learning.models.flight.flight_condition_response import ConditionReadout
from harness.learning.models.flight.flight_masked_readings import combine

SENSORS=("T24","T30","T48","T50","P15","P2","P21","P24","Ps30","P40","P50","Nf","Nc","Wf")
SETTINGS=("alt","Mach","TRA","T2")
VIEWS=("原读数","温压转速关系","相同部件工况的余差","合用原量与部件余差")
PRIMARY="依本航班重复测量的寿命分歧合用部件读法"


def relationships(rows):
    """各比值只涉及作者的实测字段。高压压气机下游仅有静压，相关读法为经验量。"""
    s=rows[:,:14]
    t2=rows[:,17]
    if np.any(s<=0) or np.any(t2<=0):raise ValueError("当前公式需要正的绝对温度、压力、转速和燃油流量")
    t24,t30,t48,t50,p15,p2,p21,p24,ps30,p40,p50,nf,nc,wf=s.T
    return np.log(np.column_stack((p21/p2,p24/p21,t24/t2,t30/t24,ps30/p24,
                                   t50/t48,p50/p40,p15/p2,nf/np.sqrt(t2),
                                   nc/np.sqrt(t24),wf/ps30,nc/nf)))


class ComponentFeatures:
    def fit(self,prep,points,healthy):
        self.prep=prep
        self.points=points
        z=prep[0].transform(healthy[:,14:])
        links=relationships(healthy)
        self.reference=self.regression(z,links)
        self.internal=[]
        # 低压、高压和膨胀段分别依赖风扇转速、核心转速以及燃油供给。
        for predictors,targets in (([8],[0,1,2,7]),([9],[3,4]),([8,9,10],[5,6]),([],[8,9,10,11])):
            x=np.column_stack((z,links[:,predictors]))
            self.internal.append((predictors,targets,self.regression(x,links[:,targets])))
        return self

    @staticmethod
    def regression(x,y):
        return make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=1.)).fit(x,y)

    def measurements(self,rows):
        z=self.prep[0].transform(rows[:,14:])
        raw=rows[:,:14]-self.prep[1].predict(z)
        links=relationships(rows)
        external=links-self.reference.predict(z)
        internal=np.empty_like(links)
        for predictors,targets,model in self.internal:
            internal[:,targets]=links[:,targets]-model.predict(np.column_stack((z,links[:,predictors])))
        matrix=np.column_stack((raw,external,internal))
        return z,matrix

    def read(self,rows):
        z,matrix=self.measurements(rows)
        neighbors=np.argsort(cdist(self.points,z),axis=1)[:,:max(2,int(np.ceil(len(z)/8)))]
        if neighbors.shape[1]<2:raise ValueError("同工况重复试验少于两个读数")
        # 交错的邻近距离排名给两个子组相近的工况覆盖，不会按真实寿命挑点。
        result=[]
        for subset in (neighbors,neighbors[:,::2],neighbors[:,1::2]):
            q1,mid,q3=np.percentile(matrix[subset],[25,50,75],axis=1)
            result.append(np.column_stack((mid,q3-q1,self.points)))
        return np.stack(result)


class ComponentReadings:
    def fit(self,x,y):
        self.readouts=[ConditionReadout().fit(view,y) for view in self.split(x[:,0])]
        return self

    @staticmethod
    def split(x):
        indices=[np.arange(14),np.arange(14,26),np.arange(26,38),np.r_[0:14,26:38]]
        return [x[:,:,np.r_[ix,38+ix,76:80]] for ix in indices]

    def trial(self,x):
        values=[]
        for j in range(3):
            predictions=[combine(*reader.readings(z)) for reader,z in zip(self.readouts,self.split(x[:,j]),strict=True)]
            values.append(np.column_stack(predictions))
        bank=values[0]
        disagreement=np.abs(values[1]-values[2])
        weights=(1+disagreement)**-2
        return bank,disagreement,(bank*weights).sum(1)/weights.sum(1)
