from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("house_price_model (1).pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Overall Qual"]),
            float(request.form["Gr Liv Area"]),
            float(request.form["Garage Cars"]),
            float(request.form["Total Bsmt SF"]),
            float(request.form["Full Bath"]),
            float(request.form["Year Built"])
        ]
        input_df = pd.DataFrame([features], columns=[
            "Overall Qual", "Gr Liv Area", "Garage Cars",
            "Total Bsmt SF", "Full Bath", "Year Built"
        ])
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)
        return render_template("result.html", prediction=prediction)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
