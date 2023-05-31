#!/usr/bin/env python3

#import sys
#sys.path.append("..")
#import mlgen3

from mlgen3.materializer.linuxcppstandalone import LinuxCPPStandalone
from mlgen3.trainers.testtrainsplit import TestTrainSplit
#from mlgen3.trainers.testtrainsplit import TestTrainSplit

from mlgen3.models.SKRandomForestClassifier import SKRandomForestClassifier
from mlgen3.implemantations.cpp.IfElse import IfElse

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
trainer=TestTrainSplit(x,y)
model=SKRandomForestClassifier(max_depth=10, n_estimators=5)
trainer.train(model)
# #Alternative, load mdoel
scores=trainer.score() #Can have another dataset, score is a call to the type object of the modelconfig
print("My model scrors to "+str(scores))
# #Modifications on the model object, prune, ...
implementation=IfElse(model, feature_type="int", label_type="float")
implementation.implement()
# print(implementation._header)
# print("#####")
# print(implementation._code)
# #Can already have test code / data arrays
materializer=LinuxCPPStandalone(implementation, run_test=True, measure_time=True, run_in_perf=True)
materializer.materialize("mymodeltets")
materializer.deploy() # in this case, make
# output=materializer.run() # output is a fancy python object {"accuracy":5, "time":2.2, "icachemisses":26378}