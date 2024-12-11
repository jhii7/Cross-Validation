
# Cross Validation Evaluation: R^2/ F1 Scoring Tool

## Overview

#### What is Cross-Validation?

Cross-validation is a method used in statistical analysis and machine learning to assess and compare the performance of predictive models. It works by splitting the data into two or more subsets. In this process, the data is divided into complementary parts: one part is used to train the model, while another is used for validating its performance.
Cross-validation is mainly used to determine how well a predictive model can perform on new datasets. Its accuracy is crucial in real-world scenarios.

#### Why Is Cross Validation Important?
Cross-validation has indispensable role in developing and evaluating predictive models and offers several important benefits below:
1. Cross-validation can provide a detailed model performance assessment across training and validation datasets, which helps uncover any variance in results. 

2. Cross-validation enhances the model's reliability by ensuring that success isn't simply a result of overfitting a particular dataset, leading to better predictions for new, independent datasets.

3. Cross-validation can be applied to compare different models based on their performance to assist in model selection. It allows us to identify the one that performs best across various data splits.

4. Cross-validation aids in tuning parameters: we can test different hyperparameter settings to choose the best one, which enhances the model's predictive accuracy. Cross-validation is an essential tool for building and validating strong predictive models.

## Tool Goals

The goal of our Cross Validation evaluation tool is to run both Single train-test split and K-folds Cross Validation on an input dataset, and generate both R^2 and F1 scores to evaluate the performance between the validation techniques. 

## Instructions

#### 1. Install Packages

To install necessary packages, paste the following command in the terminal:
```
pip install imbalanced-learn numpy pandas scikit-learn matplotlib
```
#### 2. Running the Script

To run the Python script in VSC/Github, enter the following command in the terminal:
```
python F1R2figsINPUT.py
```
After this, the script will prompt the user for a file path. Enter the input file's path into the terminal. 
```
"Enter path to the CSV: "
```
The following example provides the filepath to csv file dataset1.csv, located inside TestingSets folder within the directory.
```
# Example:
TestingSets/dataset1.csv
```

Jupyterhub allows the user to view generated plots from evaluation. Copy script into Jupyter notebook and run to observe output plots if needed.


#### 3. Evaluating the Output
The following output will record the R^2 and F1 values for each train-test split and k-fold generated from the input dataset. The following is an example output derived from imput file dataset1.csv:
```
$ python F1R2figsINPUT.py
Enter path to the CSV: TestingSets/dataset1.csv
 
 
R^2 Evaluation:
 
Train-Test Split, Linear Regression R^2 Training Data: 
0.00013016622712969106
Train-Test Split, Linear Regression R^2 Testing Data: 
-0.03326427777288421
 
 
K-folds, Linear Regression: 
R² Score for each fold: [np.float64(-0.002), np.float64(-0.0049), np.float64(-0.0092), np.float64(-0.1091), np.float64(-0.019)]
Average R² across 5 folds: -0.03
 
 
F1 Evaluation:
 
Best threshold: -0.9226599999999999, Best F1 Score: 0.97
Train-Test Split, Logistic Regression (F1):
F1 Score (Training Data): 0.5149
F1 Score (Testing Data): 0.5106
 
 

K-folds, Logistic Regression (F1, Representing Testing Sets):
F1 Score for each fold: [0.9529, 0.9899, 0.9691, 0.9899, 0.9688]
Average F1 Score across 5 folds: 0.9741

```