import streamlit as st
import numpy as np
from mlp import MultiLayerPerceptron

# App Title
st.title("FitReady")

# User Input
workout_type = st.sidebar.selectbox("Workout Type", ["Cardio", "Strength", "Flexibility"])
muscle_groups = st.sidebar.multiselect("Muscle Groups", ["Chest", "Back", "Legs", "Core", "Arms",])
intensity = st.sidebar.slider("Intensity Level", 1, 10)
duration = st.sidebar.slider("Duration (minutes)", 10, 120)
st.sidebar.write("How did the workout feel?")
user_satisfaction = st.sidebar.feedback("stars")

# Display Recommendations
st.subheader("Recommendations", divider="grey")

import csv

def train_model(epochs, layers):
    mlp = MultiLayerPerceptron(layers)

    with open("data/data.csv", "r") as data, open("data/labels.csv", "r") as labels:
        dataReader = csv.reader(data)
        next(dataReader, None)

        labelsReader = csv.reader(labels)
        next(labelsReader, None)

        reader = zip(dataReader, labelsReader)
        for i in range(epochs):
            mse = 0
            for row in reader:
                mse += mlp.bp(row[1], row[2])


train_model(15, [5, 5, 2])


def get_routines(workout_type, intensity):
    with open("data/data.csv", "r") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:


    

warm_up, wind_down = get_routines(workout_type, intensity)
selection = st.pills("Warm up suggestions", warm_up, selection_mode="multi")
selection = st.pills("Wind down suggestions", wind_down, selection_mode="multi")



