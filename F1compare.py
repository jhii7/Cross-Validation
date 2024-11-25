
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer


 ##### NOT FINISHED wil not run
## TO RUN FILE ON dataset1:


#       python F1compare.py

#       TestingSets/dataset1.csv


file_path = input("Enter the path to the CSV file: ")
data = pd.read_csv(file_path, header=None)

print(data.head()) 
# Print column names
print(data.columns)  

X = data[0].values.reshape(-1, 1)  # First col = feature
y = data[1].values


## Single Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train-Test Split, Logistic Regression F1 Score Training Data:")
training_f1 = f1_score(y_train, y_pred_train, average='weighted')
print(training_f1)
print("Train-Test Split, Logistic Regression F1 Score Testing Data:")
testing_f1 = f1_score(y_test, y_pred_test, average='weighted')
print(testing_f1)



## K FOLD


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
f1_scoreE = make_scorer(f1_score, average='weighted')
scores = cross_val_score(model, X, y, cv=kf, scoring=f1_scoreE)


## Results
average_f1 = np.mean(scores)
print("K-folds, Logistic Regression F1 Scores: ")
print(f"F1 Score for each fold: {[round(score, 4) for score in scores]}")
print(f"Average F1 Score across {k} folds: {average_f1:.4f}")