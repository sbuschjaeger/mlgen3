from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


#generate sklearn models

#random forest
rf = RandomForestClassifier(n_estimators=16, max_depth=2).fit(X_train, y_train)

X_train_sk_leaves = [rf.apply(_train.reshape(1,-1))[0] for _train in X_train]

#logistic regression
lr = LogisticRegression().fit(X_train_sk_leaves, y_train)

X_test_sk_leaves = np.asarray([rf.apply(_test.reshape(1,-1))[0] for _test in X_test])


#print scores of models for fun

rf_acc = rf.score(X_test, y_test)
rf_lr_acc = lr.score(X_test_sk_leaves, y_test)


#port into mlgen3
import sys
import os
sys.path.append(os.path.abspath(".."))
from mlgen3.models.linear import Linear
from mlgen3.models.tree_ensemble.forest import Forest
from mlgen3.models.ssf import SSF


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

#generate mlgen3 models
ssf = SSF.from_sklearn(rf, lr)


#compare leafe indices
X_test_ssf_leaves = np.squeeze([ssf.forest.apply(_test) for _test in X_test], axis=-1)

assert(np.equal(X_test_sk_leaves, X_test_ssf_leaves), "Leaf indices are not equal")

#compare scores
correct_predictions = 0
for i, x in enumerate(X_test):
    if ssf.predict_proba(x.reshape(1, -1)) == y_test[i]:
        correct_predictions += 1
ssf_acc = correct_predictions / len(y_test)

assert(rf_acc == ssf_acc, "Scores are not equal")

print("everything is fine!")

