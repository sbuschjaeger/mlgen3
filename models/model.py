from abc import ABC

class Model(ABC):

    #Keep x,y, trainconfig, ... rememebered
    # x_train, y_train, x_test, y_test

    def __init__(self):
    @abstractmethod
    def fit(self,x,y):
        pass
    @abstractmethod
    def score_dataset(self,x,y, _model=None):
        pass
    
    def train_model(self, x,y, train_type="TestTrainSplit", test_size=0.25, random_state=42):
        #Splitting, prepare data
        #calls to fit
        self._model=self.fit()

    def score(self, x=None, y=None):
        if x==None:
            if self.x_test==None:
                print("You are dumb")
            else:
                x=self.x_test
    
