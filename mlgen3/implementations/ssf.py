import numpy as np

from mlgen3.implementations.implementation import Implementation

class SSF(Implementation):

    def __init__(self, model, feature_type="float", label_type="int", internal_type = "float"):
        super().__init__(model,feature_type,label_type)
        self.internal_type = internal_type