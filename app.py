import streamlit as st
import numpy as np
import mlp

# App Title
st.title("FitReady")

# User Input
workout_type = st.sidebar.selectbox("Workout Type", ["Cardio", "Strength", "Flexibility"])
muscle_groups = st.sidebar.multiselect("Muscle Groups", ["Chest", "Back", "Legs", "Shoulders", "Arms", "Abs", "Calves", "Glutes", "Full Body"])
intensity = st.sidebar.slider("Intensity Level", 1, 10)
duration = st.sidebar.slider("Duration (minutes)", 10, 120)
st.sidebar.write("How did the workout feel?")
user_satisfaction = st.sidebar.feedback("stars")

# Display Recommendations
st.subheader("Recommendations", divider="grey")

import csv

def train_model()

def get_routines(workout_type, intensity):
    with open("data/data.csv", "r") as file:
        reader = csv.reader(file)
        next(reader, None)

        for row in reader:

    

warm_up, wind_down = get_routines(workout_type, intensity)
selection = st.pills("Warm up suggestions", warm_up, selection_mode="multi")
selection = st.pills("Wind down suggestions", wind_down, selection_mode="multi")



