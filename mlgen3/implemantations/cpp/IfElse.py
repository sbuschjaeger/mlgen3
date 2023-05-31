from abc import ABC

from mlgen3.models.RandomForest.Tree import Tree
from .ensemble import Ensemble

class IfElse(Ensemble):

    def __init__(self, model, feature_type="int", label_type="int"):
        # TODO assert that model is either Tree or Forest object
        super().__init__(model,feature_type,label_type)

    def implement_member(self, number): #returned
        tree=self.model.internal_forest.trees[number]
        # TODO use format instead of replace
        header="""
        std::vector<{label_type}> predict_{number}(std::vector<{feature_type}> &pX);
        """.replace("{label_type}",self.label_type).replace("{feature_type}",self.feature_type).replace("{number}",str(number))

        def implement_node(node):
            if node.prediction is not None:
                return_code="std::vector<"+self.label_type+">{"+",".join([str(s) for s in node.prediction])+"}"
                return """return {prediction};""".replace("{prediction}",return_code)
            return """
            if (pX[{fi}] <= {split}){
                {leftChild}
            }
            else{
                {rightChild}
            }
            """.replace("{fi}",str(node.feature)).replace("{split}",str(node.split)).replace("{leftChild}",implement_node(node.leftChild)).replace("{rightChild}",implement_node(node.rightChild))

        code="""
        std::vector<{label_type}> predict_{number}(std::vector<{feature_type}> &pX){
            {tree_code}
        }
        """.replace("{tree_code}",implement_node(tree.head)).replace("{label_type}",self.label_type).replace("{feature_type}",self.feature_type).replace("{number}",str(number))

        return header, code
    