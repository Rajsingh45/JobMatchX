from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

# Expected feature order
FEATURE_ORDER = [
    "CGPA", "Internships", "Projects", "Workshops/Certifications", 
    "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks", 
    "ExtraActivities", "PlacementTrain"
]

# Utility function to convert yes/no text to binary
def yes_no_to_binary(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "yes":
            return 1
        elif value == "no":
            return 0
    # If already numeric or not matching, try converting to int or default to 0
    try:
        return int(value)
    except:
        return 0

@app.route("/")
def home():
    return render_template("index.html")  # If you're serving HTML via Flask

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debugging
        
        # Check that all required fields are provided
        if not all(key in data for key in FEATURE_ORDER):
            return jsonify({"error": "Missing required input fields"}), 400
        
        # Create a list for each feature with appropriate conversions
        processed_data = [
            float(data["CGPA"]),
            int(data["Internships"]),
            int(data["Projects"]),
            int(data["Workshops/Certifications"]),
            int(data["AptitudeTestScore"]),
            float(data["SoftSkillsRating"]),
            int(data["SSC_Marks"]),
            int(data["HSC_Marks"]),
            yes_no_to_binary(data["ExtraActivities"]),
            yes_no_to_binary(data["PlacementTrain"])
        ]
        
        # Convert to DataFrame with the expected column names
        input_data = pd.DataFrame([processed_data], columns=FEATURE_ORDER)
        
        prediction = model.predict(input_data)[0]
        return jsonify({"placement_prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
