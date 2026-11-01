#!/usr/bin/env python3

import os
import subprocess
import tempfile
import unittest

import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from mlgen3.implementations.tree.cpp.ifelse import IfElse
from mlgen3.implementations.tree.cpp.native import Native
from mlgen3.materializer.cpp.linuxstandalone import LinuxStandalone

from mlgen3.models.tree_ensemble.forest import Forest

# The main of the materializer only prints accuracy and latency, so this one prints the raw output of every
# observation instead. It reads the testing.csv that the materializer writes and skips its label column.
MAIN_CPP = """
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "model.h"

int main(int argc, char *argv[]) {
	std::ifstream file(argv[1]);
	std::string header, line, column, entry;
	std::getline(file, header);

	std::cout << std::setprecision(10);
	while (std::getline(file, line)) {
		if (line.empty()) {
			continue;
		}

		std::stringstream columns(header), values(line);
		std::vector<float> x;
		while (std::getline(columns, column, ',') && std::getline(values, entry, ',')) {
			if (column == "label") {
				break;
			}
			x.push_back(std::stof(entry));
		}

		std::vector<float> out = predict(x);
		for (unsigned int k = 0; k < out.size(); ++k) {
			std::cout << out[k] << (k + 1 == out.size() ? '\\n' : ' ');
		}
	}
	return 0;
}
"""


class TestGradientBoosting(unittest.TestCase):

    def setUp(self):
        # init="zero" is the only init the Forest import supports, 20 trees of depth 3 keep the compile time small
        self.models = []

        X, y = datasets.load_diabetes(return_X_y=True)
        reg = GradientBoostingRegressor(n_estimators=20, max_depth=3, init="zero", random_state=0)
        reg.fit(X, y)
        self.models.append(("regressor", reg, X))

        X, y = datasets.load_breast_cancer(return_X_y=True)
        binary = GradientBoostingClassifier(n_estimators=20, max_depth=3, init="zero", random_state=0)
        binary.fit(X, y)
        self.models.append(("binary", binary, X))

        X, y = datasets.load_wine(return_X_y=True)
        multi = GradientBoostingClassifier(n_estimators=20, max_depth=3, init="zero", random_state=0)
        multi.fit(X, y)
        self.models.append(("multiclass", multi, X))

    def check(self, name, model, X, implementation_class, n_files):
        forest = Forest.from_sklearn(model)
        # score() stores X as the test set of the forest, which the materializer writes to testing.csv
        forest.score(X, model.predict(X))

        implementation = implementation_class(forest, feature_type="float", label_type="float", n_files=n_files)
        implementation.implement()

        materializer = LinuxStandalone(implementation, measure_time=True)
        path = os.path.join(tempfile.gettempdir(), "mlgen3", "TestGradientBoosting",
                           f"{name}_{type(implementation).__name__}_{n_files}")
        materializer.materialize(path)
        materializer.deploy()

        with open(os.path.join(path, "main.cpp"), "w") as f:
            f.write(MAIN_CPP)

        build = subprocess.run(f"cd {path} && make -j4", capture_output=True, text=True, shell=True)
        self.assertEqual(build.returncode, 0, build.stderr)

        run = subprocess.run(f"cd {path} && ./model testing.csv", capture_output=True, text=True, shell=True)
        self.assertEqual(run.returncode, 0, run.stderr)

        out = np.array([[float(v) for v in line.split()] for line in run.stdout.splitlines()])
        reference = self.reference(model, X)
        self.assertEqual(out.shape, reference.shape)
        np.testing.assert_allclose(out, reference, atol=1e-4)

        materializer.clean()

    def reference(self, model, X):
        # The forest sums the raw values of the trees, which is what decision_function returns. Regression and binary
        # classification return one column, so it is reshaped to match the multiclass output.
        if isinstance(model, GradientBoostingRegressor):
            raw = model.predict(X)
        else:
            raw = model.decision_function(X)
        return np.asarray(raw).reshape(len(X), -1)

    def test_gb_sklearn_import(self):
        for name, model, X in self.models:
            with self.subTest(f"Importing the {name} model"):
                forest = Forest.from_sklearn(model)
                np.testing.assert_allclose(forest.predict_proba(X), self.reference(model, X), atol=1e-4)

    def test_ifelse_linuxstandalone(self):
        for name, model, X in self.models:
            for n_files in [1, 3]:
                msg = f"Running test_ifelse_linuxstandalone on GB {name} with n_files={n_files}"
                with self.subTest(msg):
                    self.check(name, model, X, IfElse, n_files)

    def test_native_linuxstandalone(self):
        for name, model, X in self.models:
            for n_files in [1, 3]:
                msg = f"Running test_native_linuxstandalone on GB {name} with n_files={n_files}"
                with self.subTest(msg):
                    self.check(name, model, X, Native, n_files)


if __name__ == '__main__':
    unittest.main()
