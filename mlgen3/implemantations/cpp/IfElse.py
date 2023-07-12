from abc import ABC

from mlgen3.models.RandomForest.Tree import Tree
from .ensemble import Ensemble

class IfElse(Ensemble):

    def __init__(self, model, feature_type="int", label_type="int"):
        # TODO assert that model is either Tree or Forest object
        super().__init__(model,feature_type,label_type)

    def implement_member(self, number): #returned
        tree=self.model.internal_forest.trees[number]
        header=f"std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &pX);"

        def implement_node(node, indentation=""):
            if node.prediction is not None:
                arr = "{" + ",".join([str(s) for s in node.prediction]) + "}"
                return f"return std::vector<{self.label_type}>({arr})"
                #return_code="std::vector<"+self.label_type+">({"+",".join([str(s) for s in node.prediction])+"})"
                # return """return {prediction};""".replace("{prediction}",return_code)
            
            return f"""
                {indentation}if (pX[{node.feature}] <= {node.split}){{
                {indentation}    {implement_node(node.leftChild, indentation+"   ")}
                {indentation}}}else{{
                {indentation}    {implement_node(node.rightChild, indentation+"   ")}
                {indentation}}}
            """

        code=f"""
            std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &pX){{
            {implement_node(tree.head)}
        }}
        """

        return header, code
    