# DOIForest
This repository contains the code for the experiments of the paper "Deep Optimal Isolation Forest with Genetic Algorithm for Anomaly Detection".

# Requirement
- numpy==1.20.1
- sklearn==0.22.1
- pandas==1.4.1

# Dataset
We evaluate all methods on 20 real-world benchmark datasets, which are available in public [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php), [Kaggle Repository](https://www.kaggle.com/datasets), and [ADRepository](https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets).

# Experiment
### You can try different `--iteration` (the layers) to see how the AUC performance changes. 
### An example for running the code:
    python demo.py --dataset=wine --iteration=80
