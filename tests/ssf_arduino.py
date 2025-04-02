from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlgen3.materializer.cpp.arduino import Arduino
from mlgen3.models.ssf import SSF
from mlgen3.implementations.ssf.cpp.ssf import Native as SSFImplementation
from tests import read_data
import math

# datasets: shopping, spambase, adult, drybean, letter, rice, room, magic,
X,y = read_data.readData('letter')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=12, max_depth=3)
lr = LogisticRegression(max_iter=10000)
test_length = 10
d = 3
for t in [0.4]:

    rf.fit(X_train,y_train)
    print('Score: ' + str(rf.score(X_test,y_test)))
    ssf, forest, lr, stumps = SSF.from_data(X_train, y_train, rf, lr, t, d)
    X_test = (math.pow(10, d) * X_test).astype(int)
    ssf_mlgen_acc = ssf.score(X_test, y_test)
    print(ssf_mlgen_acc)

    implementation = SSFImplementation(ssf, feature_type="float", label_type="float")
    implementation.implement()

    materializer = Arduino(implementation, measure_time=True, amount_features=X.shape[1])
    implementation.model.XTest = X_test[:test_length, :]  # Just some data points  X_test[:10, :]
    implementation.model.YTest = y_test[:test_length]  # Just some data points [:10]
    
    materializer.materialize("./testmodels/")
    materializer.deploy(board="megaatmega2560", mcu="atmega2560")
