import streamlit as st

# App Title
st.title("FitReady")

# User Input
workout_type = st.sidebar.selectbox("Workout Type", ["Cardio", "Strength", "Flexibility"])
intensity = st.sidebar.slider("Intensity Level", 1, 10)
duration = st.sidebar.slider("Duration (minutes)", 10, 120)
user_satisfaction = st.sidebar.slider("How did the workout feel?", 1, 10)

# Display Recommendations
st.subheader("Recommendations", divider="grey")

import json

def get_routines(workout_type, intensity):
    with open("data/routines.json", "r") as file:
        data = json.load(file)
    level = "high" if intensity > 5 else "low"
    return data[workout_type][level]

warm_up, wind_down = get_routines(workout_type, intensity)
st.write(f"Warm-Up Routine: {warm_up}")
st.write(f"Wind-Down Routine: {wind_down}")

