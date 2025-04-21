from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("house_price_model.pkl")

df = pd.read_csv("AmesHousing.csv")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
if 'Order' in df.columns:
    df.drop('Order', axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
model_columns = df.drop('SalePrice', axis=1).columns

@app.route('/')
def home():
    return render_template("index.html", columns=model_columns)

@app.route('/predict', methods=["POST"])
def predict():
    input_data = {}
    for col in model_columns:
        val = request.form.get(col, 0)
        try:
            input_data[col] = float(val)
        except:
            input_data[col] = 0
    input_df = pd.DataFrame([input_data], columns=model_columns)
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
