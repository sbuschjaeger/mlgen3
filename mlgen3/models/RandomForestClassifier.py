from .model import model

class RandomForestClassified(Model) : #Could be SKLearnRFC
    def __init__(self, max_depth, n_estimators):

    def fit(self, x, y):
        return #ourforeststructure

    def score_dataset(self,x,y, model=None):
        if(model==None):
            model=self._model
        model.predict(x)
        #Compute some value
        return 0, "Accuracy"