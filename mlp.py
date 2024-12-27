'''
This file states the two classes UserWorkoutModel and ExerciseRecommender.
These classes are the machine learning models powering the app.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

class ModelNotTrainedException(Exception):
    """Custom exception for untrained model errors."""
    def __init__(self, message="Model not trained. Call train_model() first."):
        self.message = message
        super().__init__(self.message)


class UserWorkoutModel:
    '''
    This class is used to train a model that predicts the user's average, maximum,
    and resting heart rate based on their metrics.
    '''
    def __init__(
        self,
        x,
        y,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=150,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=0,
    ):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha

    def train(self):
        '''
        This functions trains the ML model and returns the model, X_test, and y_test.
        '''
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )
        model = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
            )
        )
        model.fit(x_train, y_train)
        return model, x_test, y_test


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

class ExerciseRecommender:
    '''
    This class is used to recommend exercises based on user input.
    '''
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.model = None

    def preprocess_data(self):
        '''
        Preprocess data into usable format for the model.
        Achieves this by converting categorical data to numerical.
        '''
        # Convert categorical data to numerical
        self.data["Type"] = self.data["Type"].astype("category").cat.codes
        self.data["BodyPart"] = self.data["BodyPart"].astype("category")
        self.data["BodyPart_codes"] = self.data["BodyPart"].cat.codes
        self.data["Equipment"] = self.data["Equipment"].astype("category")
        self.data["Equipment_codes"] = self.data["Equipment"].cat.codes
        self.data["Level"] = self.data["Level"].astype("category").cat.codes
        self.data["Rating"] = self.data.groupby(["BodyPart", "Equipment"])["Rating"].transform(
            lambda x: x.fillna(x.mean())
        )

    def train_model(self):
        '''
        Trains the model using the preprocessed data.
        '''
        self.preprocess_data()
        x = self.data[["Type", "BodyPart_codes", "Equipment_codes", "Level"]]
        y = self.data["Title"]
        X_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def recommend_exercises(self, target_muscles, target_equipment, num_exercises=5):
        '''
        Recommends exercises based on target muscles and equipment.
        '''
        if self.model is None:
            raise ModelNotTrainedException()

        target_muscles_codes = [
            self.data["BodyPart"].cat.categories.get_loc(muscle)
            for muscle in target_muscles
        ]
        target_equipment_codes = [
            self.data["Equipment"].cat.categories.get_loc(equipment)
            for equipment in target_equipment
        ]

        # Filter exercises by target muscles and equipment
        filtered_data = self.data[
            self.data["BodyPart_codes"].isin(target_muscles_codes)
            & self.data["Equipment_codes"].isin(target_equipment_codes)
        ]
        if filtered_data.empty:
            return []

        # Predict and recommend exercises
        x = filtered_data[["Type", "BodyPart_codes", "Equipment_codes", "Level"]]
        predictions = self.model.predict(x)
        recommended_exercises = filtered_data[
            filtered_data["Title"].isin(predictions)
        ].head(num_exercises)

        return recommended_exercises[
            ["Title", "Desc", "BodyPart", "Equipment", "Rating"]
        ]