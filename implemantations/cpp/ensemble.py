from abc import ABC,abstractmethod

from implemantations.implementation import Implementation

class Ensemble(Implementation):

    @abstractmethod
    def implement_member(self, number): #returned
        pass

    def __init__(self, model, feature_type="int", label_type="int"):
        self.model=model
        self.feature_type=feature_type
        self.label_type=label_type

        self._code=""
        self._header=""
    
    def implement(self):
        self._header="""
        #include <vector>
        std::vector<{label_type}> predict(std::vector<{feature_type}> &pX);
        """.replace("{label_type}",self.label_type).replace("{feature_type}",self.feature_type)

        ensemble_code=""
        tree_code=""

        for n_tree in range(len(self.model.internal_forest.trees)):
            header, code=self.implement_member(n_tree)
            tree_code += code
            self._header += header
            ensemble_code+="predict_{num}(pX);\n".replace("{num}",str(n_tree))

        self._code="""
        {tree_code}
        std::vector<{label_type}> predict(std::vector<{feature_type}> &pX){
            {ensemble_code}
        }
        """.replace("{ensemble_code}",ensemble_code).replace("{label_type}",self.label_type).replace("{feature_type}",self.feature_type).replace("{tree_code}", tree_code)