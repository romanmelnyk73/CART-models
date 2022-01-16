#  Breast Canccer Dataset from Kaggle is used for classification
# Import pandas
import pandas as pd
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Import models: DecisionTreeClassifier + LogisticRegressionClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Import ensembles: BaggingClassifier + AdaboostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Import additional functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Import metrics: accuracy_score + roc_aus_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


le = LabelEncoder()
data = pd.read_csv('data.csv')
X = data.iloc[:, [2,9]]
y = le.fit_transform(data["diagnosis"])

# data["diag"] = data["diagnosis"].apply(lambda val: 1 if val == "M" else 0)
# # print(data[['diagnosis','diag']])
# y = data['diag']

SEED=1
# Split the dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8,
                                                    stratify=y,
                                                    random_state=SEED)
#________________Decision Tree  ____________
dt = DecisionTreeClassifier(max_depth=3, max_features=0.2, min_samples_leaf=0.06, random_state=SEED)
# best params: 'max_depth': 3, 'max_features': 0.2, 'min_samples_leaf': 0.06
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_pred, y_test)
print("Test set accuracy of DT: {:.2f}".format(acc))

# ____________Logistic Regression___________________
logreg = LogisticRegression(random_state=SEED)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

acc_lr = accuracy_score(y_pred_lr, y_test)
print("Test set accuracy of Logreg: {:.2f}".format(acc_lr))

#__________ Bagging of dt______________
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, oob_score=True, random_state=1)
bc.fit(X_train, y_train)
y_pred_bc = bc.predict(X_test)
acc_bc = accuracy_score(y_pred_bc, y_test)
# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy of BaggingClassifier: {:.3f}, OOB accuracy: {:.3f}'.format(acc_bc, acc_oob))

# ___________Adaboost: Adaptive Boosting + ROC AUC_______________
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict_proba(X_test)[:,1]

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_ada)
print('ROC AUC score of AdaBoost: {:.2f}'.format(ada_roc_auc))

#__________GRID SEARCH__________

# Define the grid of hyperparameters 'params_dt'
params_dt = {'max_depth': [3, 4,5, 6],
             'min_samples_leaf': [0.04, 0.06, 0.08],
             'max_features': [0.2, 0.4,0.6, 0.8]
             }

# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='accuracy',
                       cv=10,
                       n_jobs=-1)
# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy: {:.3f}'.format(best_CV_score))

# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_
# Evaluate test set accuracy
test_acc = best_model.score(X_test,y_test)
# Print test set accuracy
print("Test set accuracy of best model: {:.3f}".format(test_acc))


