# duplicate-detection
This repository contains all resources that produced the results for the paper "Scalable Product Duplicate Detection". 
The main file is the python file named "paper.py" and takes the json file named "TVs-all-merged" as input data. 
The goal is to provide a solution to the duplicate indentifying problem. 
The python file is seperated by cells, where each cell serves a distinct step in the proposed method:

## Import packages and data preparation
Import the data and transform the input data from dictionary to lists to make it easier to work with.

## Define functions
User defined function used later in the file are declared

## Data cleaning
Data cleaning is performed

## Find Model Words from titles
Model words are extracted from the titles of the TVs

## Bootstrap
Function"bootstrap": creates bootstrap samples based on the input data

## Create Binary Vector Representation
Function "bin_vec": creates binary vector based on the model words and give the characteristic matrix as output

## MinHash
Function "minhash": creates minhash signatures from the binary matrix

## LSH
Function "lsh": performs Locality-Sensitive Hashing on the Signature matrix and produces the list of candidate pairs

## LSH Evaluation
Function "f1_star": calculates the pair quality, pair completeness, f1* and the number of candidate pairs from the candidate pairs list

## Dissimilarity Matrix
Function "dissim": creates the dissimilarity matrix based on the candidate pair list

## Clustering
Function "cluster": performs the Agglomerative Clustering algorithm based on the dissimilarity matrix

## Cluster Evaluation
Function "f1": evaluates the clusters and returns the precision, recall, f1 and the number of comparisons made

## Optimization
The optimal distance threshold is calculated based on the f1 score

## Final
Final results are computed for the paper

Function are described by docstrings that explain the function well.
