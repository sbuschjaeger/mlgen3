from sklearn import tree
import sklearn
import mlgen3
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from mlgen3.models.tree_ensemble.tree import Tree
from mlgen3.materializer.cpp.arduino import Arduino
from mlgen3.implemantations.tree.cpp.native import Native

if __name__ == "__main__":

    # Load data
    dataFrame = pd.read_csv("testing.csv");
    Y = dataFrame.pop("label")
    X = dataFrame

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)

    print("start generating testmodels")
    sktree = tree.DecisionTreeClassifier(max_depth=2)

    sktree.fit(X_train, Y_train)



    #tree = Tree()
    tree = Tree.from_sklearn(sktree)
    #tree.implement()

    native = Native(tree, feature_type="double", label_type="double")
    native.implement()
    
    materializer = Arduino(native, measure_time=True)
    native.model.XTest = X_test
    native.model.YTest = Y_test

    materializer.materialize("./testmodels/")

    materializer.deploy(board = "uno")


