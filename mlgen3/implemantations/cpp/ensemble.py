from abc import ABC,abstractmethod

from ..implementation import Implementation

class Ensemble(Implementation):

    @abstractmethod
    def implement_member(self, number): #returned
        pass

    def __init__(self, model, feature_type="int", label_type="int"):
        super().__init__(feature_type, label_type)
        self.model=model

        self._code=""
        self._header=""
    
    def implement(self):
        self._header=f"""
            #pragma once
            #include <vector>
            #include <algorithm>
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);
        """

        ensemble_code=f"""
            std::vector<{self.label_type}> result;
            std::vector<{self.label_type}> result_temp;
        """
        tree_code=""

        for n_tree in range(len(self.model.internal_forest.trees)):
            header, code=self.implement_member(n_tree)
            tree_code += code
            self._header += header
            if n_tree==0:
                ensemble_code+=f"result=predict_{n_tree}(pX);\n"
            else:
                ensemble_code+=f"result_temp=predict_{n_tree}(pX);\n"
                ensemble_code+="std::transform(result.begin(), result.end(), result_temp.begin(),result.begin(), std::plus<{label_type}>());\n"

        # TODO NAME IS REQUIRED HERE!
        self._code=f"""
            #include "model.h"
            {tree_code}
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX){{
                {ensemble_code}
                return result;
            }}
        """