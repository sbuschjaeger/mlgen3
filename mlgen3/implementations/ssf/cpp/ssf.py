from abc import abstractmethod
import numpy as np
import textwrap
import re
from mlgen3.implementations.implementation import Implementation
from mlgen3.implementations.linear.cpp.native import Native as LinearNative
from mlgen3.implementations.tree.cpp.native import Native as TreeNative

class Native(Implementation):

    def __init__(self, model, feature_type="float", label_type="float"):
        super().__init__(model,feature_type,label_type)

    def implement(self):
        #the_forest = self.model.forest.copy()
        # TODO COPY MODEL?
        for e, node_map in zip(self.model.forest.trees, self.model.node_mapping):
            #leaves = e.get_leaf_nodes()
            #inner_node_cnt = min([n.id for n in leaves])
            e.n_classes = len(node_map)
            for node in e.get_leaf_nodes():
                node.prediction = [0 for _ in range(len(node_map))] #np.zeros(len(leaves), dtype=np.int32)
                node.prediction[node_map[node.id]] = 1

        tree_code = ""
        ensemble_code = ""
        tree_headers = ""

        # TODO CONFIGURE THIS
        native_tree = TreeNative(self.model.forest, feature_type=self.feature_type, label_type="unsigned int", int_type=None, reorder_nodes = False, set_size = 8, force_cacheline = False)
        for n_tree in range(len(self.model.forest.trees)):
            header, code = native_tree.implement_member(n_tree)
            tree_code += code
            tree_headers += "\t\t" + header + "\n" 
            
            # Add tab indentation here already so the code looks somewhat nice
            if n_tree == 0:
                ensemble_code+=f"\tstd::vector<unsigned int> one_hot = predict_{n_tree}(pX);\n"
            else:
                if n_tree == 1:
                    ensemble_code += f"\tstd::vector<unsigned int> result_temp;\n"  
                ensemble_code+=f"\tresult_temp = predict_{n_tree}(pX);\n"
                ensemble_code+=f"\tone_hot.insert(one_hot.end(), result_temp.begin(), result_temp.end());\n"
                # ensemble_code+=f"\tstd::(result.begin(), result.end(), result_temp.begin(),result.begin(), std::plus<{self.label_type}>());\n"
        
        native_linear = LinearNative(self.model.lr, feature_type="unsigned int", label_type="float")
        native_linear.implement()

        native_linear.code = native_linear.code.replace("predict", "predict_ssf")
        native_linear.code = native_linear.code.replace("#include \"model.h\"", "")

        native_linear.header = native_linear.header.replace("predict", "predict_ssf")
        native_linear.header = native_linear.header.replace("#pragma once", "")
        native_linear.header = native_linear.header.replace("#include <vector>", "")


        # For readability, we use f-strings which, unfortunatley, introduces tabs which look messy the end. Hence
        # we use inspect.cleandoc to remove tabs, but preserve the general indentation. Note that since tree_headers
        # are already properly aligned, we add them _after_ the call to cleandoc. Same goes for ensemble_code

        # TODO NAME IS REQUIRED HERE!
        self.header = f"""
            #pragma once
            #include <vector>
            #include <algorithm>

            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);
            {tree_headers}
            {native_linear.header}
        """ 

        predict = f"""
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX){{
                {ensemble_code}
                return predict_ssf(one_hot);
            }}
        """
        
        self.code = f"""
            #include "model.h"
            {tree_code}
            {native_linear.code}
            {predict}
        """ 



        # # Implement forest
        # forest = TreeNative(self.model.forest, feature_type=self.feature_type, label_type="bool")
        # forest.implement()

        # code = forest.code
        # code = "\n\n//code for random forest\n" + code
        # code = code.replace("predict(std::vector", "predict_forest(std::vector")
        # forest.code = code

        # header = forest.header
        # header = "\n\n//header for random forest\n" + header
        # header = header.replace("predict(std::vector", "predict_forest(std::vector")
        # forest.header = header

        # # Implement logistic regression
        # lr = LinearNative(self.model.lr, feature_type="int", label_type=self.label_type)
        # lr.implement()

        # code = lr.code
        # code = "//code for logistic regression\n\n\n\n" + code
        # code = code.replace("predict(std::vector", "predict_lr(std::vector")
        # lr.code = code

        # header = lr.header
        # header = "//header for logistic regression\n" + header
        # header = header.replace("predict(std::vector", "predict_lr(std::vector")
        # lr.header = header

        # # Combine the two implementations
        # self.code = forest.code + lr.code
        # self.header = forest.header + lr.header

        # self.code += "//combining Random Forest and Logistic Regression\n"
        # self.code += f"std::vector<{lr.label_type}> predict(std::vector<{forest.feature_type}> features) {{\n"
        # if len(forest.model.trees) > 1:
        #     self.code += "    std::vector<int> leaf_indices = predict_leaf_indices(features);\n"
        # else:
        #     self.code += "    std::vector<int> leaf_indices;\n"
        #     self.code += "    leaf_indices[0] = predict_leaf_index(features);\n"
        # self.code += "    return predict_lr(leaf_indices);\n"
        # self.code += f"}}\n"

        # self.header += "\n//combining Random Forest and Logistic Regression\n"
        # self.header += f"std::vector<{lr.label_type}> predict(std::vector<{forest.feature_type}> features);\n"

        # self.code = textwrap.dedent(self.code)
        # self.header = textwrap.dedent(self.header)
        # self.header = re.sub(r'^[ \t]+', '', self.header, flags=re.MULTILINE) 
    