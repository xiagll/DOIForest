import unittest

import pandas as pd

from detectors.DOI_forest import *
from detectors.sampling import *
from detectors import EuclideanLSHFamily

class TestOptForest(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/ad.csv', header=None)
        self._X = data.values[:, :-1]

    def test_decision_function(self):
        num_of_tree = 100
        DOI_forest = DOIForest(num_of_tree,
                                       VSSampling(num_of_tree), EuclideanLSHFamily(norm=2, bin_width=4))
        DOI_forest.fit(self._X)
        # l2hash_forest.display()

        x = self._X[0:10, :]
        score = DOI_forest.decision_function(x)
        print("score: ", score)
        print("avg isolation efficiency: ", DOI_forest.get_avg_isolation_efficiency())
        print("avg branch factor: ", DOI_forest.get_avg_branch_factor())
        self.assertTrue(DOI_forest is not None)