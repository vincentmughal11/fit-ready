"""
This file dictates the layout of the streamlit app. 
It loads the user workout model and exercise recommender model 
and displays the recommendations based on the user input.
"""

import streamlit as st
import numpy as np
import pandas as pd
from mlp import UserWorkoutModel, ExerciseRecommender


@st.cache_resource
def load_user_workout_model():
    """
    This function loads the user workout model, 
    preprocess the data from data/user_workout.csv, 
    and then trains the model and returns it.
    """
    user_workout_data = pd.read_csv("data/user_workout.csv")
    user_workout_data = pd.get_dummies(
        user_workout_data, columns=["Gender", "Workout_Type"]
    )
    x1 = user_workout_data[
        [
            "Age",
            "Gender_Female",
            "Gender_Male",
            "Weight (kg)",
            "Height (m)",
            "Session_Duration (hours)",
            "Workout_Type_Yoga",
            "Workout_Type_HIIT",
            "Workout_Type_Cardio",
            "Workout_Type_Strength",
        ]
    ]
    y1 = user_workout_data[["Max_BPM", "Avg_BPM", "Resting_BPM"]]
    user_workout_model_creator = UserWorkoutModel(x1, y1)
    return user_workout_model_creator.train()


userWorkoutModel, X_test, y_test = load_user_workout_model()


@st.cache_resource
def load_exercise_recommender():
    """
    This funciton loads the exercise recommender model 
    and processes the data from data/exercises.csv, 
    trains the model, and then returns it.
    """
    recommender = ExerciseRecommender("data/exercises.csv")
    recommender.preprocess_data()
    recommender.train_model()
    return recommender


exercise_recommender = load_exercise_recommender()


# App Title
st.title("FitReady")


class UserInfoWorkout:
    """
    This class stores the user information and workout information 
    and encodes it to be used in the model.
    """

    def __init__(self, age, gender, weight, height, duration, workout):
        self.age = age
        self.gender = gender
        self.weight = weight
        self.height = height
        self.duration = duration
        self.workout = workout

    def encode(self):
        """
        This function encodes the user information
         and workout information to be used in the model.
         """
        if self.workout == "Cardio":
            workout_encode = [False, False, True, False]
        elif self.workout == "Strength":
            workout_encode = [False, False, False, True]
        elif self.workout == "HIIT":
            workout_encode = [False, True, False, False]
        elif self.workout == "Yoga":
            workout_encode = [True, False, False, False]
        return np.array(
            [
                self.age,
                0 if self.gender == "Male" else 1,
                1 if self.gender == "Male" else 0,
                self.weight,
                self.height,
                self.duration,
                *workout_encode,
            ]
        )


# User Input
st.sidebar.write("## User/Workout information")
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_age = st.sidebar.slider("Age", 18, 65)
user_height = st.sidebar.slider("Height (m)", 1.5, 2.2, step=0.01)
user_weight = st.sidebar.slider("Weight (kg)", 50, 150)
user_duration = st.sidebar.slider("Duration (hours)", 1.00, 3.00, step=0.01)
workout_type = st.sidebar.selectbox(
    "Workout Type", ["Cardio", "Strength", "HIIT", "Yoga"]
)


# Workout input
st.sidebar.write("## Exercises")
st.sidebar.write("only for Strength workouts")
muscle_groups = st.sidebar.multiselect(
    "Target Muscles",
    [
        "Abdominals",
        "Adductors",
        "Abductors",
        "Biceps",
        "Calves",
        "Chest",
        "Forearms",
        "Glutes",
        "Hamstrings",
        "Lats",
        "Lower Back",
        "Middle Back",
        "Traps",
        "Neck",
        "Quadriceps",
        "Shoulders",
        "Triceps",
    ],
)
equipment = st.sidebar.multiselect(
    "Equipment",
    [
        "Bands",
        "Barbell",
        "Kettlebells",
        "Dumbbell",
        "Other",
        "Cable",
        "Machine",
        "Body Only",
        "Medicine Ball",
        "Exercise Ball",
        "Foam Roll",
        "E-Z Curl Bar",
    ],
)


# Display Recommendations
st.subheader("Recommendations", divider="grey")

user_info_workout = UserInfoWorkout(
    user_age, user_gender, user_weight, user_height, user_duration, workout_type
)
user_input = user_info_workout.encode()
max_bpm, avg_bpm, rest_bpm = userWorkoutModel.predict([user_input])[0]

st.metric(label="Max BPM", value=round(max_bpm))
st.metric(label="Avg BPM", value=round(avg_bpm))
st.metric(label="Resting BPM", value=round(rest_bpm))

if workout_type == "Strength":
    exercises = exercise_recommender.recommend_exercises(muscle_groups, equipment)
    if not exercises:
        st.write("No exercises found for the selected muscle groups and equipment.")
    else:
        st.dataframe(exercises)
