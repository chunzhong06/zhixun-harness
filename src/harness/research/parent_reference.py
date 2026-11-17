SOURCE='''from sklearn.base import BaseEstimator
from harness_task import load_data,evaluate_model
class PreserveParent(BaseEstimator):
    def fit(self,X,y):return self
    def predict(self,X):return X.iloc[:,-1].to_numpy()
X,y,V,T=load_data()
model=PreserveParent().fit(X,y)
evaluate_model(model,'完整父版保持')
'''
