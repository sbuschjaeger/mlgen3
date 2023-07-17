from abc import ABC,abstractmethod

class Model(ABC):

    #Keep x,y, trainconfig, ... rememebered
    # x_train, y_train, x_test, y_test

    def __init__(self, original_model):
        self.original_model = original_model
        self.XTrain = None
        self.XTest = None
        self.YTrain = None
        self.YTest = None
        
    def fit(self):
        if self.XTrain is not None and self.YTrain is not None:
            self.original_model.fit(self.XTrain, self.YTrain)
        else:
            raise ValueError(f"Cannot fit model because XTrain and YTrain have not been set properly. Set these fields, e.g. by using a TestTrainSplit Trainer. XTrain was {self.XTrain} and YTrain was {self.YTrain}")

        self.init_from_fitted(self.original_model)

    @abstractmethod
    def score_model(self):
        # TODO: If we only distinguish between classification and regression and force a predict / predict_proba method, then we do not really need to implement this
        pass

    @abstractmethod
    def init_from_fitted(self, original_model):
        pass

    def score(self, X = None, y = None):
        if X is None or y is None:
            if self.XTest is None or self.YTest is None:
                raise ValueError(f"Cannot score model, because there is no data to score it on. Make sure that either X and y are supplied for scoring or that XTest and YTest are set, i.e. by using a TestTrainSplit Trainer. X was {X} and Y was {y}. XTest was  {self.XTest} and YTest was {self.YTest}")
            else:
                X = self.XTest
                y = self.YTest
        else:
            # TODO: This implies that we need to call score on a self-fitted model at some point?
            if self.XTest is None or self.YTest is None:
                self.XTest = X
                self.YTest = y

        return self.score_model(X,y)
    
