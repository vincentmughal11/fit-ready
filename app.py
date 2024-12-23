import streamlit as st
import numpy as np
from mlp import userWorkoutModel
import pandas as pd
import math

data = pd.read_csv("data/user_workout.csv")

data = pd.get_dummies(data, columns=["Gender", "Workout_Type"])
print(data.head())

x = data[["Age", "Gender_Female", "Gender_Male", "Weight (kg)", "Height (m)", "Session_Duration (hours)", "Workout_Type_Yoga", "Workout_Type_HIIT", "Workout_Type_Cardio", "Workout_Type_Strength"]]
y = data[["Max_BPM", "Avg_BPM", "Resting_BPM"]]

modelCreator = userWorkoutModel(x, y)
model, X_test, y_test = modelCreator.train()

# App Title
st.title("FitReady")

class UserInfoWorkout:
    def __init__(self, age, gender, weight, height, duration, workout):
        self.age = age
        self.gender = gender
        self.weight = weight
        self.height = height
        self.duration = duration
        self.workout = workout

    def encode(self):
        if self.workout == "Cardio":
            workoutArray = [False, False, True, False]
        elif self.workout == "Strength":
            workoutArray= [False, False, False, True]
        elif self.workout == "HIIT":
            workoutArray = [False, True, False, False]
        elif self.workout == "Yoga":
            workoutArray = [True, False, False, False]
        return np.array([self.age, 0 if self.gender == "Male" else 1, 1 if self.gender == "Male" else 0,  self.weight, self.height, self.duration, *workoutArray])

# User Input
st.sidebar.write("## User/Workout information")
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_age = st.sidebar.slider("Age", 18, 65)
user_height = st.sidebar.slider("Height (cm)", 150, 220)
user_weight = st.sidebar.slider("Weight (kg)", 50, 150)
duration = st.sidebar.slider("Duration (hours)", 1.00, 3.00, step=0.01)
workout_type = st.sidebar.selectbox("Workout Type", ["Cardio", "Strength", "Flexibility"])

st.sidebar.write('## Exercises')
st.sidebar.write('only for Strength workouts')
muscle_groups = st.sidebar.multiselect("Target Muscles", ["Chest", "Back", "Legs", "Core", "Arms",])
intensity = st.sidebar.slider("Intensity Level", 1, 10)

# Display Recommendations
st.subheader("Recommendations", divider="grey")

def get_BPM(user_gender, user_age, user_height, user_weight, duration, workout_type):
    x = np.array([user_age, ])
    x = x.reshape(1, -1)
    print(x)
    return model.predict(x)
    

bpm = get_BPM(user_gender, user_age, user_height, user_weight, duration, workout_type)
st.metric(label="Max BPM", value=math.round(max_bpm))
st.metric(label="Avg BPM", value=math.round(avg_bpm))
st.metric(label="Resting BPM", value=math.round(rest_bpm))
# selection = st.pills("Warm up suggestions", warm_up, selection_mode="multi")
# selection = st.pills("Wind down suggestions", wind_down, selection_mode="multi")

