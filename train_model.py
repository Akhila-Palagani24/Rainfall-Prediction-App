import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("data/flood_data.csv")

# Features and target
X = data[["rainfall", "river_level", "temperature", "humidity"]]
y = data["flood"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully and saved!")