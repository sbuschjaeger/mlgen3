from abc import ABC
import inspect

from mlgen3.models.tree_ensemble.tree import Tree
from .ensemble import Ensemble

class IfElse(Ensemble):

    def __init__(self, model, feature_type="int", label_type="int"):
        # TODO assert that model is either Tree or Forest object
        super().__init__(model,feature_type,label_type)

    def implement_member(self, number): #returned
        if number is None:
            tree = self.model.trees[0]
            header = f"std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);"
        else:
            tree = self.model.trees[number]
            header = f"std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &pX);"

        def implement_node(node, indentation = ""):
            if node.prediction is not None:
                arr = "{" + ",".join([str(s) for s in node.prediction]) + "}"
                return f"return std::vector<{self.label_type}>({arr});"
            
            # This string should not have any whitespaces/newlines at the beginning/end of it, because it would 
            # mess-up the recursion at bit => strip it
            return f"""
                {indentation}if (pX[{node.feature}] <= {node.split}){{
                {indentation}    {implement_node(node.leftChild, indentation+"    ")}
                {indentation}}} else {{
                {indentation}    {implement_node(node.rightChild, indentation+"    ")}
                {indentation}}}
            """.strip()
        
        if number is None:
            code = f"""
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX){{
                {implement_node(tree.head)}
            }}
        """
        else:
            code = f"""
                std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &pX){{
                    {implement_node(tree.head)}
                }}
            """

        return header, code
    