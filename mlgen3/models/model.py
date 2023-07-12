from abc import ABC,abstractmethod

class Model(ABC):

    #Keep x,y, trainconfig, ... rememebered
    # x_train, y_train, x_test, y_test

    def __init__(self):
        self.XTrain=None
        self.XTest=None
        self.YTrain=None
        self.YTest=None
        
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def score_model(self):
        pass

    def score(self, x=None, y=None):
        if x is None:
            if self.XTest is None:
                # TODO: refactor to assert
                print("You are dumb")
            else:
                x=self.XTest
                y=self.YTest
        return self.score_model(x,y)
    
