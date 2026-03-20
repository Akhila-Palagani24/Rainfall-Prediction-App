from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Flood Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    rainfall = data["rainfall"]
    river_level = data["river_level"]
    temperature = data["temperature"]
    humidity = data["humidity"]

    features = np.array([[rainfall, river_level, temperature, humidity]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Flood Likely"
    else:
        result = "No Flood"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
    