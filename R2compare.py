import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


## TO RUN FILE ON dataset1:


#       python R2compare.py

#       TestingSets/dataset1.csv

# !!!
# Here, we get the R^2 (coefficient of determination) values for training and testing data.
# "R-squared is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model."
# In essence, the higher the score, the better predictive performance. 
# !!!

# Train-Test-Split #

# Can someone get this part working? Use X and y below in the meantime as a test set. 
# data1 = pd.read_csv("dataset1.csv", sep='/t', header=None, engine='python')
# X = data1.iloc[:, 0]
# y = data1.iloc[:, -1]

###X = np.arange(20).reshape(-1, 1)
###y = np.array([5, 12, 11, 19, 30, 29, 23, 40, 51, 54, 74, 62, 68, 73, 89, 84, 89, 101, 99, 106])


file_path = input("Enter the path to the CSV file: ")
data = pd.read_csv(file_path, header=None)

print(data.head()) 
# Print column names
print(data.columns)  


X = data[0].values.reshape(-1, 1)  # First col =  feature
y = data[1].values  # Second col =  target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression().fit(X_train, y_train)
print("Train-Test Split, Linear Regression R^2 Training Data: ")
trainingScore = model.score(X_train, y_train)
print(trainingScore)
print("Train-Test Split, Linear Regression R^2 Testing Data: ")
testingScore = model.score(X_test, y_test)
print(testingScore)




# K-Folds (5) #

###X = np.arange(20).reshape(-1, 1)
###y = np.array([5, 12, 11, 19, 30, 29, 23, 40, 51, 54, 74, 62, 68, 73, 89, 84, 89, 101, 99, 106])

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

average_r2 = np.mean(scores) 
print("K-folds, Linear Regression: ")
print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average R² across {k} folds: {average_r2:.2f}")
