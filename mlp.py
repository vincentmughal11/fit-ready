import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor

import pandas as pd

class UserWorkoutModel:
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

from sklearn.ensemble import RandomForestClassifier

class ExerciseRecommender:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.model = None

    def preprocess_data(self):
        # Convert categorical data to numerical
        self.data['BodyPart'] = self.data['BodyPart'].astype('category').cat.codes
        self.data['Equipment'] = self.data['Equipment'].astype('category').cat.codes
        self.data['Rating'] = self.data['Rating'].fillna(0)  # Fill missing ratings with 0

    def train_model(self):
        self.preprocess_data()
        X = self.data[['BodyPart', 'Equipment']]
        y = self.data['Title']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def recommend_exercises(self, target_muscles, target_equipment, num_exercises=5):
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        
        # Filter exercises by target muscles and equipment
        filtered_data = self.data[self.data['BodyPart'].isin(target_muscles) & self.data['Equipment'].isin(target_equipment)]
        if filtered_data.empty:
            return []

        # Predict and recommend exercises
        X = filtered_data[['Type', 'BodyPart', 'Equipment', 'Level']]
        predictions = self.model.predict(X)
        recommended_exercises = filtered_data[filtered_data['Title'].isin(predictions)].head(num_exercises)
        
        return recommended_exercises[['Title', 'Desc', 'BodyPart', 'Equipment', 'Rating']]

# Example usage:
# recommender = ExerciseRecommender('/Users/vincentmughal/Downloads/fitready/data/exercises.csv')
# recommender.train_model()
# recommendations = recommender.recommend_exercises(['Abdominals'], num_exercises=5)
# print(recommendations)