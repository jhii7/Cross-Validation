import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Here, we get the R^2 (coefficient of determination) values for training and testing data.
# "R-squared is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model."
# In essence, the higher the score, the better predictive performance. 

filepath = input("Enter the path to the CSV: ")
data = pd.read_csv(filepath, header=None)
X = data[0].values.reshape(-1, 1)  # First col =  feature
y = data[1].values  # Second col =  target
print(" ")


# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression().fit(X_train, y_train)
print("Train-Test Split, Linear Regression R^2 Training Data: ")
trainingScore = model.score(X_train, y_train)
print(trainingScore)
print("Train-Test Split, Linear Regression R^2 Testing Data: ")
testingScore = model.score(X_test, y_test)
print(testingScore)



# K-Folds (5) 
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

average_r2 = np.mean(scores) 
print("K-folds, Linear Regression: ")
print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average R² across {k} folds: {average_r2:.2f}")
