import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import PredictionErrorDisplay

# !!!
# Here, we get the R^2 (coefficient of determination) values for training and testing data.
# "R-squared is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model."
# In essence, the higher the score, the better predictive performance. 
# !!!

# USING SKLEARN DIABETES DATASET, COMMENT OUT IF USING OWN DATASET
# BMI VS. BLOOD PRESSURE
X, y = load_diabetes(return_X_y=True)
# END

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression().fit(X_train, y_train)
print("Train-Test Split, Linear Regression R^2 Training Data: ")
trainingScore = model.score(X_train, y_train)
print(trainingScore)
print("Train-Test Split, Linear Regression R^2 Testing Data: ")
testingScore = model.score(X_test, y_test)
print(testingScore)

# PLOT #
y_pred = model.predict(X_train)
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(
    y[:len(y_pred)],
    y_pred=y_pred,
    kind="actual_vs_predicted",
    subsample=100,
    ax=axs[0],
    random_state=0,
)
axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    y[:len(y_pred)],
    y_pred=y_pred,
    kind="residual_vs_predicted",
    subsample=100,
    ax=axs[1],
    random_state=0,
)
axs[1].set_title("Residuals vs. Predicted Values")
fig.suptitle("Plotting Train-Test Split predictions")
plt.tight_layout()
plt.show()

# K-Folds (5) #

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

average_r2 = np.mean(scores) 
print("K-folds, Linear Regression: ")
print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average R² across {k} folds: {average_r2:.2f}")

# PLOT #
y_pred = cross_val_predict(model, X, y, cv=kf)
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(
    y,
    y_pred=y_pred,
    kind="actual_vs_predicted",
    subsample=100,
    ax=axs[0],
    random_state=0,
)
axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    y,
    y_pred=y_pred,
    kind="residual_vs_predicted",
    subsample=100,
    ax=axs[1],
    random_state=0,
)
axs[1].set_title("Residuals vs. Predicted Values")
fig.suptitle("Plotting cross-validated predictions")
plt.tight_layout()
plt.show()
