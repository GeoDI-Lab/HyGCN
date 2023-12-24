# HyGCN

# Title:
A Hypergraph-based Hybrid Graph Convolutional Network for Intracity Human Activity Intensity Prediction and Geographic Relationship Interpretation

# Introduction:
This is a Pytorch implementation of HyGCN. Our code is based on ASTGCN (https://github.com/guoshnBJTU/ASTGCN-r-pytorch) and Pytorch Geometric (https://github.com/pyg-team/pytorch_geometric).

# Pre:
Step1: Clone the code of ASTGCN.

Step2: Put HyGCN.py into model and MN_astgcn.yaml into configurations.

# Datasets:
Step1: Download the demo dataset provided by (). And Put it into folder data (If not, please create it).

Step2: Process dataset like ASTGCN. 
```shell
python prepareData.py --config configurations/MN_astgcn.conf
  ```

# Train and Test:
Please refer to ASTGCN's Run and Test (https://github.com/guoshnBJTU/ASTGCN-r-pytorch).

# Reference:
Wang, Yi, and Di Zhu. "A Hypergraph-based Hybrid Graph Convolutional Network for Intracity Human Activity Intensity Prediction and Geographic Relationship Interpretation." Information Fusion (2023): 102149.
