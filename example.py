#!/usr/bin/env python3

import mlgen3

'''
copy function on model class?
'''

x,y=load_from_csv("adult.csv", yindex=0)

'''
types: TestTrainSplit, CorssValidation
'''
model=RandomforestClassifier(max_depth=10, n_estimators=5)
trainer=TestTrainSplit(tetest_sizest=0.25, random_state=5)
trainer.train(model, x, y)
#Alternative, load mdoel
score, description=model.score() #Can have another dataset, score is a call to the type object of the modelconfig
print("My model scrors to "+description+" of "+score)
#Modifications on the model object, prune, ...
implementation=IfElse(model, feature_type="int", label_type="int")
implementation.implement()
#Can already have test code / data arrays
materializer=LinuxCPPStandalone(implementation, run_test=True, measure_time=True, run_in_perf=True)
materializer.materialize("mymodel/")
materializer.deploy() # in this case, make
output=materializer.run() # output is a fancy python object {"accuracy":5, "time":2.2, "icachemisses":26378}