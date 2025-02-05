import numpy as np
import textwrap
import re
from mlgen3.implementations.implementation import Implementation
from mlgen3.implementations.linear.cpp.native import Native as LinearNative
from mlgen3.implementations.tree.cpp.native import Native as TreeNative

class SSF(Implementation):

    def __init__(self, model, feature_type="float", label_type="float"):
        super().__init__(model,feature_type,label_type)

    def implement(self):
        # Implement forest
        forest = TreeNative(self.model.forest, feature_type=self.feature_type, label_type="leaf_index")
        forest.implement()

        code = forest.code
        code = "\n\n//code for random forest\n" + code
        code = code.replace("predict(std::vector", "predict_forest(std::vector")
        forest.code = code

        header = forest.header
        header = "\n\n//header for random forest\n" + header
        header = header.replace("predict(std::vector", "predict_forest(std::vector")
        forest.header = header

        # Implement logistic regression
        lr = LinearNative(self.model.lr, feature_type="int", label_type=self.label_type)
        lr.implement()

        code = lr.code
        code = "//code for logistic regression\n\n\n\n" + code
        code = code.replace("predict(std::vector", "predict_lr(std::vector")
        lr.code = code

        header = lr.header
        header = "//header for logistic regression\n" + header
        header = header.replace("predict(std::vector", "predict_lr(std::vector")
        lr.header = header

        # Combine the two implementations
        self.code = forest.code + lr.code
        self.header = forest.header + lr.header

        self.code += "//combining Random Forest and Logistic Regression\n"
        self.code += f"std::vector<{lr.label_type}> predict(std::vector<{forest.feature_type}> features) {{\n"
        if len(forest.model.trees) > 1:
            self.code += "    std::vector<int> leaf_indices = predict_leaf_indices(features);\n"
        else:
            self.code += "    std::vector<int> leaf_indices;\n"
            self.code += "    leaf_indices[0] = predict_leaf_index(features);\n"
        self.code += "    return predict_lr(leaf_indices);\n"
        self.code += f"}}\n"

        self.header += "\n//combining Random Forest and Logistic Regression\n"
        self.header += f"std::vector<{lr.label_type}> predict(std::vector<{forest.feature_type}> features);\n"

        self.code = textwrap.dedent(self.code)
        self.header = textwrap.dedent(self.header)
        self.header = re.sub(r'^[ \t]+', '', self.header, flags=re.MULTILINE) 
    