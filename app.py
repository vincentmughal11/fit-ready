import streamlit as st
import numpy as np
from mlp import MultiLayerPerceptron

# App Title
st.title("FitReady")

# User Input
st.sidebar.write("## User/Workout information")
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_age = st.sidebar.slider("Age", 18, 65)
user_height = st.sidebar.slider("Height (cm)", 150, 220)
user_weight = st.sidebar.slider("Weight (kg)", 50, 150)
duration = st.sidebar.slider("Duration (hours)", 0.00, 3.00, step=0.01)
workout_type = st.sidebar.selectbox("Workout Type", ["Cardio", "Strength", "Flexibility"])

st.sidebar.write('## Exercises')
st.sidebar.write('only for Strength workouts')
muscle_groups = st.sidebar.multiselect("Target Muscles", ["Chest", "Back", "Legs", "Core", "Arms",])
intensity = st.sidebar.slider("Intensity Level", 1, 10)

# Display Recommendations
st.subheader("Recommendations", divider="grey")

# import csv

# def train_model(epochs, layers):
#     mlp = MultiLayerPerceptron(layers)

#     with open("data/data.csv", "r") as data, open("data/labels.csv", "r") as labels:
#         dataReader = csv.reader(data)
#         next(dataReader, None)

#         labelsReader = csv.reader(labels)
#         next(labelsReader, None)

#         reader = zip(dataReader, labelsReader)
#         for i in range(epochs):
#             mse = 0
#             for row in reader:
#                 mse += mlp.bp(row[1], row[2])


# train_model(15, [5, 5, 2])


# def get_routines(workout_type, intensity):
#     with open("data/data.csv", "r") as file:
#         reader = csv.reader(file)
#         next(reader, None)

#         for row in reader:

import json

def get_routines(workout_type, intensity):
    with open("data/temp.json", "r") as file:
        data = json.load(file)
    level = "high" if intensity > 5 else "low"
    return data[workout_type][level]

warm_up, wind_down = get_routines(workout_type, intensity)
selection = st.pills("Warm up suggestions", warm_up, selection_mode="multi")
selection = st.pills("Wind down suggestions", wind_down, selection_mode="multi")



