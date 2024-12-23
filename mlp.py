import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor

import pandas as p

class userWorkoutModel:
    def __init__(self, x, y, max_depth=5, learning_rate=0.1, n_estimators=150, subsample=1, colsample_bytree=1, reg_alpha=0):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        model = MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", 
                                                  max_depth=self.max_depth, 
                                                  learning_rate=self.learning_rate, 
                                                  n_estimators=self.n_estimators, 
                                                  subsample=self.subsample, 
                                                  colsample_bytree=self.colsample_bytree, 
                                                  reg_alpha=self.reg_alpha))
        model.fit(X_train, y_train)
        return model, X_test, y_test

# #test/predict
# y_pred = model.predict(X_test)
# print("Predictions for regression:", y_pred)

# mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
# print("Mean Squared Errors for each target:", mse)

# import matplotlib.pyplot as plt

# xgb.plot_importance(model.estimators_[0])
# plt.show()

# # to tune hyperparameters

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'estimator__subsample': [0.5, 0.8, 1],
#     'estimator__colsample_bytree': [0.5, 0.8, 1],
#     'estimator__reg_alpha': [0, 0.1, 0.5, 1, 5],
# }

# # Use GridSearchCV with MultiOutputRegressor
# grid_search = GridSearchCV(
#     estimator=MultiOutputRegressor(XGBRegressor(objective="reg:squarederror")),
#     param_grid=param_grid,
#     scoring="neg_mean_squared_error",
#     cv=3,
#     verbose=1
# )

# grid_search.fit(X_train, y_train)
# print("Best Parameters:", grid_search.best_params_)