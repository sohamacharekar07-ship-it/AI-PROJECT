
# Diabetes Prediction System
# Step 4: Flask backend

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            # Get form values
            for feature in FEATURES:
                values[feature] = float(request.form[feature])

            # Basic validation
            if any(value < 0 for value in values.values()):
                raise ValueError("Values cannot be negative.")

            if values["Age"] <= 0:
                raise ValueError("Age must be greater than zero.")

            if values["BMI"] <= 0:
                raise ValueError("BMI must be greater than zero.")

            if values["Glucose"] <= 0:
                raise ValueError("Glucose must be greater than zero.")

            # Prepare input in the same column order as training
            input_data = pd.DataFrame(
                [[values[f] for f in FEATURES]],
                columns=FEATURES
            )

            # Predict class
            prediction = int(model.predict(input_data)[0])

            if prediction == 1:
                result = (
                    "Positive class predicted by the model. "
                    "This is not a diagnosis; please consult "
                    "a qualified healthcare professional."
                )
            else:
                result = (
                    "Negative class predicted by the model. "
                    "This does not rule out diabetes. "
                    "Consult a healthcare professional if concerned."
                )

        except (ValueError, KeyError):
            error = "Please enter valid numerical values in all fields."

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)