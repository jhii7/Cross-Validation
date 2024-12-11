
# Cross Validation Evaluation: R^2/ F1 Scoring Tool

## Overview

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