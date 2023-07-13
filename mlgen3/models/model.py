from abc import ABC,abstractmethod

class Model(ABC):

    #Keep x,y, trainconfig, ... rememebered
    # x_train, y_train, x_test, y_test

    def __init__(self):
        self.XTrain = None
        self.XTest = None
        self.YTrain = None
        self.YTest = None
        
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def score_model(self):
        pass

    def score(self, x=None, y=None):
        if x is None or y is None:
            if self.XTest is None or self.YTest is None:
                raise ValueError(f"Cannot score model, because there is no data to score it on. Make sure that either x and y are supplied for scoring or that XTest and YTest are set, i.e. by using a TestTrainSplit Trainer. X was {x} and Y was {y}. XTest was  {self.XTest} and YTest was {self.YTest}")
            else:
                x=self.XTest
                y=self.YTest
        return self.score_model(x,y)
    
