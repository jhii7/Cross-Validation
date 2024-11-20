# Cross-Validation

https://scikit-learn.org/1.5/modules/cross_validation.html

```
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_validate, cross_val_predict

X, y = datasets.load_iris(return_X_y=True)


clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits = 5)

# scores = cross_val_score(clf, X, y, cv = k_folds)
# print1 = cross_validate(clf, X, y, return_train_score = True, cv = k_folds)


# # print("Cross Validation Scores: ", scores)
# # print("Average CV Score: ", scores.mean())
# # print("Number of CV Scores used in Average: ", len(scores))
# # print(print1)

lasso = linear_model.Lasso()
y_pred = cross_val_predict(lasso, X, y, cv=k_folds)

fig, axs = plt.subplots(2)
fig.suptitle('Plots')
axs[0].plot(irisdata['data'])
axs[1].plot(y_pred)
```
