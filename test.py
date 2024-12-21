import pandas as pd

#set the model up
data = pd.read_csv("data/user_workout.csv")


# Print the first 5 values of "Avg_BPM"
data["Avg_BPM"] = data["Avg_BPM"].astype(int)
print(data["Avg_BPM"].head(5))
