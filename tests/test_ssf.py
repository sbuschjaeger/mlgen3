import os
import tempfile
import unittest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

from mlgen3.implementations.ssf.cpp.ssf import Native
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone
from mlgen3.models.ssf import SSF
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


from sklearn.tree._tree import TREE_LEAF

def leaf_mapping_from_sklearn(sk_tree):
    node_ids = [0]
    mapping = {}
    
    leaf_cnt = 0
    while(len(node_ids) > 0):
        cur_node = node_ids.pop(0)

        if sk_tree.children_left[cur_node] == TREE_LEAF and sk_tree.children_right[cur_node] == TREE_LEAF:
            mapping[cur_node] = leaf_cnt
            leaf_cnt += 1
        else:
            leftChild = sk_tree.children_left[cur_node]
            node_ids.append(leftChild)

            rightChild = sk_tree.children_right[cur_node]
            node_ids.append(rightChild)

    return mapping

class TestSSF(unittest.TestCase):

    def setUp(self):
        self.test_cases = [] 

        for nc in [2,3,8]:
            X, y = make_classification(n_samples=1000, n_features=16, random_state=42, n_classes=nc, n_informative=8)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
            rf = RandomForestClassifier(n_estimators=16, max_depth=12).fit(X_train, y_train)

            X_train_one_hot = []
            for e in rf.estimators_:
                node_map = leaf_mapping_from_sklearn(e.tree_)
                adjusted_idx = [node_map[idx] for idx in e.apply(X_train)] 
                
                one_hot_leaves = np.zeros( (X_train.shape[0], e.get_n_leaves()) )
                one_hot_leaves[np.arange(len(X_train)),adjusted_idx] = 1 
                X_train_one_hot.append(one_hot_leaves)
            
            X_train_one_hot = np.concatenate(X_train_one_hot,axis=1)
            lr = LogisticRegression().fit(X_train_one_hot, y_train)

            self.test_cases.append(
                (f"n_classes = {nc}",rf,lr,X_test,y_test)
            )
 
    def test_from_scikitlearn(self):
        for name, rf, lr, X_test, y_test in self.test_cases:
            with self.subTest(params=name):
                X_test_one_hot = []
                for e in rf.estimators_:
                    node_map = leaf_mapping_from_sklearn(e.tree_)
                    adjusted_idx = [node_map[idx] for idx in e.apply(X_test)] 
                    
                    one_hot_leaves = np.zeros( (X_test.shape[0], e.get_n_leaves()) )
                    one_hot_leaves[np.arange(len(X_test)),adjusted_idx] = 1 
                    X_test_one_hot.append(one_hot_leaves)

                X_test_one_hot = np.concatenate(X_test_one_hot,axis=1)
                lr_acc = accuracy_score(lr.predict(X_test_one_hot), y_test)
                        
                ssf = SSF.from_sklearn(rf, lr)
                ssf_mlgen_acc = ssf.score(X_test, y_test)
                self.assertAlmostEqual(lr_acc, ssf_mlgen_acc["Accuracy"], places=3)

    def test_native_linuxstandalone(self):
        for name, rf, lr, X_test, y_test in self.test_cases:
            with self.subTest(params=name):
                ssf = SSF.from_sklearn(rf, lr)
                ssf_mlgen_acc = ssf.score(X_test, y_test)
                implementation = Native(ssf, feature_type="float", label_type="float")
                implementation.implement()
                
                materializer = LinuxStandalone(implementation, measure_accuracy=True, measure_time=True, measure_perf=False)
                materializer.materialize(os.path.join(tempfile.gettempdir(), "mlgen3", "TestSSFNative"))
                materializer.deploy() 
                output = materializer.run() 
                materializer.clean()
                self.assertAlmostEqual(
                    float(output["Accuracy"]), ssf_mlgen_acc["Accuracy"] * 100.0, places=3
                )


if __name__ == '__main__':
    unittest.main()
