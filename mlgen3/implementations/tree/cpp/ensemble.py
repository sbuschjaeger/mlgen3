from abc import ABC,abstractmethod
import inspect
import os
from textwrap import dedent
import textwrap

from mlgen3.models.tree_ensemble.forest import Forest
from mlgen3.models.tree_ensemble.tree import Tree

from ...implementation import Implementation

class Ensemble(Implementation):

    @abstractmethod
    def implement_member(self, number): #returned
        pass

    def __init__(self, model, feature_type="int", label_type="int"):
        super().__init__(model, feature_type, label_type)

        if not isinstance(model, (Tree,Forest)):
            raise ValueError(f"Ensemble currently only supportes Tree and Forest models, but you supplied {model}")

        if isinstance(model, Tree):
            self.model = Forest()
            self.model.trees = [model]
            self.model.weights = [1.0]

            self.original_model = model.original_model
            self.model.XTest = model.XTest
            self.model.YTest = model.YTest
            self.model.XTrain = model.XTrain
            self.model.YTrain = model.YTrain
        else:
            self.model = model

        self.code=""
        self.header=""
    
    def merge_weights(self):
        for w, t in zip(self.model.weights, self.model.trees):
            for n in t.nodes:
                if n.prediction is not None:
                    n.prediction *= w
        self.model.weights = [1.0 for _ in range(len(self.model.weights))]

    def implement(self):
        tree_code = ""
        ensemble_code = ""
        tree_headers = ""

        self.merge_weights()
        if len(self.model.trees) > 1:
            for n_tree in range(len(self.model.trees)):
                header, code = self.implement_member(n_tree)
                tree_code += code
                tree_headers += "\t\t" + header + "\n" 
                
                # Add tab indentation here already so the code looks somewhat nice
                if n_tree == 0:
                    ensemble_code+=f"\tstd::vector<{self.label_type}> result = predict_{n_tree}(pX);\n"
                else:
                    if n_tree == 1:
                        ensemble_code += f"\tstd::vector<{self.label_type}> result_temp;\n"  
                    ensemble_code+=f"\tresult_temp = predict_{n_tree}(pX);\n"
                    ensemble_code+=f"\tstd::transform(result.begin(), result.end(), result_temp.begin(),result.begin(), std::plus<{self.label_type}>());\n"
            
            # For readability, we use f-strings which, unfortunatley, introduces tabs which look messy the end. Hence
            # we use inspect.cleandoc to remove tabs, but preserve the general indentation. Note that since tree_headers
            # are already properly aligned, we add them _after_ the call to cleandoc. Same goes for ensemble_code

            # TODO NAME IS REQUIRED HERE!
            self.header = f"""
                #pragma once
                #include <vector>
                #include <algorithm>
                {tree_headers}
            """ 

            predict = f"""
                std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX){{
                    {ensemble_code}
                    return result;
                }}
            """

            self.code = f"""
                #include "model.h"
                {tree_code}
                {predict}
            """ 
        else:
            _, code = self.implement_member(None)
            self.header = f"""
                #pragma once
                #include <vector>
                std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);
            """

            self.code = f"""
                #include "model.h"
                {code}
            """.strip()

        

        