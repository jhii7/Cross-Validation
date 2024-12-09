import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Here, we compute the F1 scores for both training/testing sets for Train Test split and testing sets across all k-folds in Cross Validation.
# Higher F1 score indicates a higher recall and precision, resulting in better and more accurate classification.
#

filepath = input("Enter path to the CSV: ")
data = pd.read_csv(filepath, header=None)
X = data[0].values.reshape(-1, 1) 
y = data[1].values

# Adjusted threshold
thresh = np.median(y)
yclass = (y > thresh).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
Xresample, yresample = smote.fit_resample(X, yclass)



# Train Test Split F1
X_train, X_test, y_train, y_test = train_test_split(Xresample, yresample, test_size=0.2, random_state=0)
model = LogisticRegression().fit(X_train, y_train)

ytrainpred = model.predict(X_train)
ytestpred = model.predict(X_test)
trainf1 = f1_score(y_train, ytrainpred)
testf1 = f1_score(y_test, ytestpred)

print("Train-Test Split, Logistic Regression (F1):")
print(f"F1 Score (Training Data): {trainf1:.2f}")
print(f"F1 Score (Testing Data): {testf1:.2f}")



# K-Fold Cross Validation F1 (5)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = LogisticRegression()

f1_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = yclass[train_index], yclass[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_scores.append(float(f1_score(y_test, y_pred)))  # Convert to Python float

average_f1 = np.mean(f1_scores)
print("\nK-folds, Logistic Regression (F1, Representing Testing Sets):")
print(f"F1 Score for each fold: {[round(score, 4) for score in f1_scores]}")
print(f"Average F1 Score across {k} folds: {average_f1:.2f}")
