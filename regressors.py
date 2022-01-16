# Auto-mpg dataset from Kaggle is used for regression task
# Import pandas
import pandas as pd
# Import LinearRegressor
from sklearn.linear_model import LinearRegression
# Import RandomForestRegressor + GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Import MSE
from sklearn.metrics import mean_squared_error as MSE
# Import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Import mathplotlib
import matplotlib.pyplot as plt
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv('auto-mpg.csv')

data["horsepower"] = data["horsepower"].apply(lambda val: '103' if val == "?" else val)
data["horsepower"] = data["horsepower"].astype('float')
# print(data["horsepower"])

X = data.iloc[:,[1,2,3,4,5,6,7]]
y = data.iloc[:,0]


SEED=2
# Split the dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8,
                                                     random_state=SEED)
# ______________LinearRegression___________________
# Instantiate lr
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
rmse_lr = (MSE(y_pred_lr, y_test))**(1/2)
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# ____________________Decision Tree Regression____________________
# Instantiate dt
dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=0.16, random_state=SEED)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Compute rmse_dt
rmse_dt = (MSE(y_pred, y_test))**(1/2)

# Print rmse_dt
print("DecisionTree test set RMSE of dt: {:.2f}".format(rmse_dt))

# ______________________Random Forest______________________________
#Instantiate rf
rf = RandomForestRegressor(n_estimators=25, random_state=SEED)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# Evaluate the test set RMSE
rmse_test = (MSE(y_test, y_pred))**(1/2)

# Print rmse_test
print('RandomForest test set RMSE: {:.2f}'.format(rmse_test))

  # __________FEATURE IMPORTANCES__________
 # Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                         index= X_train.columns)

 # Sort importances
importances_sorted = importances.sort_values()

 # Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# ________________GRADIENT BOOSTING REGRESSOR___________________
gb = GradientBoostingRegressor(max_depth=1, n_estimators=100, random_state=SEED)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
rmse_gb = (MSE(y_pred_gb, y_test))**(1/2)

print('GradientBoosting RMSE: {:.2f}'.format(rmse_gb))

# _________Stochastic GB regressor____________
sgbr = GradientBoostingRegressor(max_depth=1, subsample=0.8, max_features=0.2,
                                 n_estimators=300, random_state=SEED)
sgbr.fit(X_train, y_train)
y_pred_sgbr = sgbr.predict(X_test)
rmse_sgbr = (MSE(y_pred_sgbr, y_test))**(1/2)

print('Stochastic GradientBoosting RMSE: {:.2f}'.format(rmse_sgbr))


#__________GRID SEARCH of RANDOM Forest__________
# rf = RandomForestRegressor(random_state=SEED)

# Define the grid of hyperparameters 'params_rf'
params_rf = {'n_estimators': [300, 400, 500],
             'max_depth': [4, 6, 8],
             'min_samples_leaf': [0.1, 0.2],
             'max_features': ['log2', 'sqrt']}

# Instantiate  'grid_rf' with CV
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       cv=3,
                       scoring='neg_mean_squared_error',
                       verbose=1,
                       n_jobs=-1)

# Fit 'grid_dt' to the training data
grid_rf.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract best model from 'grid_dt'
best_model = grid_rf.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
print('GridSearch RMSE of RF: {:.2f}'.format(rmse_test))
