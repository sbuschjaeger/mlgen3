from abc import ABC
from .ensemble import Ensemble


class IfElse(Ensemble):

    def __init__(self, model, feature_type="int", label_type="int"):
        super().__init__(model,feature_type,label_type)
        self.__amount_leafs = 0

    def implement_member(self, number): #returned
        if number is None:
            tree = self.model.trees[0]
            if self.label_type != "leaf_index":
                header = f"std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);"
            else:
                header = f"int predict_leaf_index(std::vector<{self.feature_type}> &pX);"
        else:
            tree = self.model.trees[number]
            if self.label_type != "leaf_index":
                header = f"std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &pX);"
            else:
                header = f"int predict_{number}_leaf_index(std::vector<{self.feature_type}> &pX);"

        def implement_node(node, indentation = ""):
            if node.prediction is not None:
                arr = "{" + ",".join([str(s) for s in node.prediction]) + "}"
                return f"return std::vector<{self.label_type}>({arr});"
            
            # This string should not have any whitespaces/newlines at the beginning/end of it, because it would 
            # mess-up the recursion at bit => strip it
            return f"""
                if (pX[{node.feature}] <= {node.split}){{
                    {implement_node(node.leftChild, indentation+"    ")}
                }} else {{
                    {implement_node(node.rightChild, indentation+"    ")}
                }}
            """
        
        def implement_node_leaf_index(node, indentation = ""):
            if node.prediction is not None:
                ret = f"return {self.__amount_leafs};"
                self.__amount_leafs += 1 #recursion is in preoder, so we can just count up the leafs and numerate them with the current state
                #after completed recursion, self.__amount_leafs == number of leafs, thats why it is named like that
                #nodes do not have any information regarding index of the leafs, so we do it like that
                return ret
            
            # This string should not have any whitespaces/newlines at the beginning/end of it, because it would 
            # mess-up the recursion at bit => strip it
            return f"""
                if (pX[{node.feature}] <= {node.split}){{
                    {implement_node_leaf_index(node.leftChild, indentation+"    ")}
                }} else {{
                    {implement_node_leaf_index(node.rightChild, indentation+"    ")}
                }}
            """
        
        if number is None:
            if self.label_type == "leaf_index":
                code = f"""
                    int predict_leaf_index(std::vector<{self.feature_type}> &pX){{
                    {implement_node_leaf_index(tree.head)}
                }}
                """
            else:
                code = f"""
                    std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX){{
                        {implement_node(tree.head)}
                    }}
                    """
        else:
            if self.label_type == "leaf_index":
                code = f"""
                    int predict_{number}_leaf_index(std::vector<{self.feature_type}> &pX){{
                        {implement_node_leaf_index(tree.head)}
                    }}
                """
            else:
                code = f"""
                    std::vector<{self.label_type}> predict_{number}(std::vector<{self.feature_type}> &pX){{
                        {implement_node(tree.head)}
                    }}
                """


        return header, code
    