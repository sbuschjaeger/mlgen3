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

    def shared_definitions(self):
        # C++ definitions that every member source file needs. They are written into the header, so each file sees them once.
        return ""

    def __init__(self, model, feature_type="int", label_type="int", n_files=1):
        super().__init__(model, feature_type, label_type)

        if not isinstance(model, (Tree,Forest)):
            raise ValueError(f"Ensemble currently only supportes Tree and Forest models, but you supplied {model}")

        if n_files < 1:
            raise ValueError(f"n_files must be at least 1, but you supplied {n_files}")
        self.n_files = n_files

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
        # The source files holding the trees, one entry per file. See self.split_member_codes
        self.member_codes=[]

    def merge_weights(self):
        for w, t in zip(self.model.weights, self.model.trees):
            for n in t.nodes:
                if n.prediction is not None:
                    n.prediction *= w
        self.model.weights = [1.0 for _ in range(len(self.model.weights))]

    def split_member_codes(self, tree_codes):
        """Splits the code of the trees into the requested number of source files, each a complete file that holds a
        contiguous chunk of trees. Chunks are as equal as possible, so a model with fewer trees than files yields one
        file per tree.

        Args:
            tree_codes (list): The code of every tree, in tree order.

        Returns:
            list: A list of complete C++ source files.
        """
        n_chunks = min(self.n_files, len(tree_codes))
        chunk_size, rest = divmod(len(tree_codes), n_chunks)

        files = []
        start = 0
        for i in range(n_chunks):
            end = start + chunk_size + (1 if i < rest else 0)
            files.append('#include "model.h"\n' + "".join(tree_codes[start:end]))
            start = end
        return files

    def implement_predict_vector(self):
        """The vector interface of the ensemble. It allocates the output once and forwards to the array version."""
        n_classes = self.model.trees[0].n_classes
        return f"""
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX){{
                std::vector<{self.label_type}> out({n_classes}, 0);
                predict(pX.data(), out.data());
                return out;
            }}
        """

    def implement(self):
        self.merge_weights()

        if len(self.model.trees) > 1:
            tree_codes = []
            tree_headers = ""

            for n_tree in range(len(self.model.trees)):
                header, code = self.implement_member(n_tree)
                tree_codes.append(code)
                tree_headers += "\t\t" + header + "\n"

            self.member_codes = self.split_member_codes(tree_codes)

            # Every tree adds its own leaf values, so the ensemble only has to zero the output and call all trees
            ensemble_code = "".join([f"\t\tout[{k}] = 0;\n" for k in range(self.model.trees[0].n_classes)])
            ensemble_code += "".join([f"\t\tpredict_{n}(pX, out);\n" for n in range(len(tree_codes))])

            # For readability, we use f-strings which, unfortunatley, introduces tabs which look messy the end. Hence
            # we use inspect.cleandoc to remove tabs, but preserve the general indentation. Note that since tree_headers
            # are already properly aligned, we add them _after_ the call to cleandoc. Same goes for ensemble_code

            self.header = f"""
                #pragma once
                #include <vector>
                {self.shared_definitions()}
                {tree_headers}
                void predict(const {self.feature_type} *pX, {self.label_type} *out);
                std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);
            """

            predict = f"""
                void predict(const {self.feature_type} *pX, {self.label_type} *out){{
                    {ensemble_code}
                }}
            """

            self.code = f"""
                #include "model.h"
                {predict}
                {self.implement_predict_vector()}
            """
        else:
            _, code = self.implement_member(None)
            self.member_codes = self.split_member_codes([code])

            if self.label_type == "leaf_index":
                # The leaf index is not summed up, so the single tree is the whole model and needs no predict function
                self.header = f"""
                    #pragma once
                    #include <vector>
                    int predict_leaf_index(std::vector<{self.feature_type}> &pX);
                """
                self.code = '#include "model.h"'
            else:
                # The single tree implements the array version of predict by itself
                self.header = f"""
                    #pragma once
                    #include <vector>
                    {self.shared_definitions()}
                    void predict(const {self.feature_type} *pX, {self.label_type} *out);
                    std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &pX);
                """

                self.code = f"""
                    #include "model.h"
                    {self.implement_predict_vector()}
                """


