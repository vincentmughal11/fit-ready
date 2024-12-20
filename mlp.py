import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

import pandas as pd

#set the model up
data = pd.read_csv("data.csv")

X = data.drop(columns=["Age", "Gender", "Weight (kg)", "Height (m)", "Session_Duration (hours)", "Workout_Type"])
y = data[["Max_BPM", "Avg_BPM", "Resting_BPM"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBRegressor

model = MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", max_depth=6, learning_rate=0.1, n_estimators=100))

#train
model.fit(X_train, y_train)

#test/predict
y_pred = model.predict(X_test)
print("Predictions for regression:", y_pred)

y_pred = model.predict(X_test)
print("Predictions for classification:", y_pred)

mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
print("Mean Squared Errors for each target:", mse)

import matplotlib.pyplot as plt

xgb.plot_importance(model.estimators_[0])
plt.show()

# to tune hyperparameters

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     "estimator__max_depth": [3, 5, 7],
#     "estimator__learning_rate": [0.01, 0.1, 0.2],
#     "estimator__n_estimators": [50, 100, 150],
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




