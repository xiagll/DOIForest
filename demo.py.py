import os
import sys
import time

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from detectors import DOIForest
from detectors import VSSampling
from detectors import EuclideanLSHFamily

import argparse

""""""
ap = argparse.ArgumentParser()
ap.add_argument("-StringA", "--dataset", required=True, help="dataset")
ap.add_argument("-IntB", "--iteration", required=True, help="iteration")
args = vars(ap.parse_args())
filename = args["dataset"]
iteration = int(args["iteration"])
print("Dataset:",filename)

""""""
num_ensemblers = 100

glass_df = pd.read_csv('data/' + filename + '.csv', header=None)  # pandas.DataFrame is returned by pandas.read_csv()
X = glass_df.values[:, :-1]  # numpy.ndarray is returned by pandas.DataFrame.values()
ground_truth = glass_df.values[:, -1]

detectors = [("DOIForest", DOIForest(num_ensemblers,
                                       VSSampling(num_ensemblers), EuclideanLSHFamily(norm=2, bin_width=4)))]


for i, (dtc_name, dtc) in enumerate(detectors):
    print("\n" + dtc_name + ":")
    AUC = []
    PR_AUC = []
    Traintime = []
    Testtime = []
    for j in range(15):
        start_time = time.time()

        dtc.fit(X)

        for j in range(iteration):
            dtc.gen2_algorithm(X, num_ensemblers, change_rate=0.2, MUTATION_RATE=0.5)

        train_time = time.time() - start_time

        y_pred = dtc.decision_function(X)

        test_time = time.time() - start_time - train_time

        auc = roc_auc_score(ground_truth, y_pred)
        AUC.append(auc)
        pr_auc = average_precision_score(ground_truth, y_pred)
        PR_AUC.append(pr_auc)

        Traintime.append(train_time)
        Testtime.append(test_time)

    mean_auc = np.mean(AUC)
    std_auc = np.std(AUC)
    mean_pr = np.mean(PR_AUC)
    std_pr = np.std(PR_AUC)
    mean_traintime = np.mean(Traintime)
    mean_testtime = np.mean(Testtime)

    print("\tAUC score:\t", mean_auc)
    print("\tAUC std:\t", std_auc)
    print("\tPR score:\t", mean_pr)
    print("\tPR std:\t", std_pr)
    print("\tTraining time:\t", mean_traintime)
    print("\tTesting time:\t", mean_testtime)