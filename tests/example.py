#!/usr/bin/env python3

#import sys
#sys.path.append("..")
#import mlgen3

import os
import tempfile

from sklearn.ensemble import RandomForestClassifier

from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.trainers.testtrainsplit import TestTrainSplit
#from mlgen3.trainers.testtrainsplit import TestTrainSplit

from mlgen3.models.tree_ensemble.forest import Forest
from mlgen3.implementations.cpp.ifelse import IfElse

from Datasets import get_dataset

'''
copy function on model class?
'''

# x,y=load_from_csv("adult.csv", yindex=0)
#Load adult data manually
x,y = get_dataset("adult")
#x,y=load_adult_dataset()

'''
types: TestTrainSplit, CorssValidation
'''


trainer = TestTrainSplit()
trainer = SKTrainer(RandomForestClassifier(max_depth=10,n_estimators=5))

model = trainer.train(RandomForestClassifier(max_depth=10,n_estimators=5))


model = Forest.from_sklearn(RandomForestClassifier(max_depth=10,n_estimators=5))
model = Forest.from_weka(...)

trainer = TestTrainSplit(x,y)
model = Forest(RandomForestClassifier(max_depth=10, n_estimators=5)) 
trainer.train(model)
# #Alternative, load mdoel
scores = trainer.score() #Can have another dataset, score is a call to the type object of the modelconfig
print("My model scrors to "+str(scores))
# #Modifications on the model object, prune, ...
implementation = IfElse(model, feature_type="int", label_type="float")
implementation.implement()
# print(implementation._header)
# print("#####")
# print(implementation._code)
# #Can already have test code / data arrays
materializer=LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "debug"))
materializer.deploy() # in this case, make
output=materializer.run() # output is a fancy python object {"accuracy":5, "time":2.2, "icachemisses":26378}
print(output)