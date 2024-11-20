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

Goal:
To evaluate the effectiveness of 5-fold cross-validation compared to using a single train-test split (i.e., without cross-validation), you can follow these steps:

Model Training Without Cross-Validation:

Split your dataset into a training set and a testing set (commonly a 70/30 or 80/20 split).
Train your model on the training set.
Evaluate the model on the testing set using appropriate metrics (e.g., accuracy, precision, recall, F1 score).
Model Training With 5-Fold Cross-Validation:

Divide your dataset into 5 equally sized folds.
For each fold:
Use it as a validation set while the other 4 folds are used for training.
Train the model and evaluate it on the validation fold.
Calculate the average performance metrics across all folds.
Comparison of Results:

Compare the average metrics obtained from 5-fold cross-validation with the metrics from the train-test split.
Analyze the variance in performance across the folds; lower variance often indicates that the model is stable and performs consistently.
Considerations:

Overfitting and Underfitting: Cross-validation helps in understanding how well your model generalizes to unseen data, while a single train-test split can give a biased estimate if the split is not representative.
Data Size: If your dataset is small, cross-validation can provide a better estimate of model performance because it allows all data to be used for training and validation.
Metrics Sensitivity: Depending on which metrics you are using, you may notice differences in the evaluation of model performance.
By comparing the metrics from both methods, you can assess the reliability and robustness of your model's performance. If cross-validation shows significantly better or more consistent results, it may be a better method to use for performance evaluation on your dataset.
