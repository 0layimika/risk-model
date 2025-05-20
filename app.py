from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("risk_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print(data)
    features = np.array([[
        data["is_new_device"],
        data["is_new_location"],
        data["hour"],
        data["day_of_the_week"],
        data["distance_km"]
    ]])
    prediction = int(model.predict(features)[0])
    print(prediction)
    return jsonify({"risk":prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
