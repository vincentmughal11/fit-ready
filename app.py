import streamlit as st
import numpy as np
from mlp import userWorkoutModel
import pandas as pd
import math

data = pd.read_csv("data/user_workout.csv")

x = data[["Age", "Gender", "Weight (kg)", "Height (m)", "Session_Duration (hours)", "Workout_Type"]]
y = data[["Max_BPM", "Avg_BPM", "Resting_BPM"]]

modelCreator = userWorkoutModel(x, y)
model, X_test, y_test = modelCreator.train()

# App Title
st.title("FitReady")

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
    x = np.array([user_age, user_gender, user_weight, user_height, duration, workout_type])
    x = x.reshape(1, -1)
    print(x)
    return model.predict(x)
    

bpm = get_BPM(user_gender, user_age, user_height, user_weight, duration, workout_type)
st.metric(label="Max BPM", value=math.round(max_bpm))
st.metric(label="Avg BPM", value=math.round(avg_bpm))
st.metric(label="Resting BPM", value=math.round(rest_bpm))
# selection = st.pills("Warm up suggestions", warm_up, selection_mode="multi")
# selection = st.pills("Wind down suggestions", wind_down, selection_mode="multi")

