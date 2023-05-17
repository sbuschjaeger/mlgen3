from models.model import Model
from sklearn.ensemble import RandomForestClassifier
from models.RandomForest.Forest import Forest
from sklearn.metrics import accuracy_score

class SKRandomForestClassifier(Model) : #Could be SKLearnRFC
    def __init__(self, max_depth, n_estimators):
        self.sklearn_model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)

    def fit(self):
        self.sklearn_model.fit(self.XTrain,self.YTrain)
        #TODO force existence in an abstract class
        self.internal_forest=Forest()
        self.internal_forest.fromSKLearn(self.sklearn_model)

        return self

    def score_model(self, x, y):
        prediction=self.internal_forest.predict_batch(x)
        accuracy = accuracy_score(y, prediction)
        
        #Compute some value
        return {"Accuracy": accuracy}