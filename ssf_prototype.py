from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


#generate some random ass scikit-learn random forest

rf = RandomForestClassifier(n_estimators=16, max_depth=2)
rf.fit(X_train, y_train)

X_train_leaves = [rf.apply(_train.reshape(1,-1))[0] for _train in X_train]

#generate a logistic regression model

lr = LogisticRegression().fit(X_train_leaves, y_train)

X_test_leaves = [rf.apply(_test.reshape(1,-1))[0] for _test in X_test]


#print scores of models for fun

print(f"RF Accuracy: {rf.score(X_test, y_test)}")
print(f"RF + Log-Reg Accuracy: {lr.score(X_test_leaves, y_test)}")
