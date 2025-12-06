from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
MODEL_PATH = "model.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except:
    raise FileNotFoundError(f"Please place '{MODEL_PATH}' in the project root directory.")

df = pd.read_csv("processed_data.csv")

features = df.drop("Price", axis=1).columns.tolist()


@app.route("/")
def home():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect values from form inputs
        input_values = []

        for col in features:
            value = request.form[col]

            # Convert numeric values
            try:
                value = float(value)
            except:
                pass   # if string keep as string

            input_values.append(value)

        # Convert to dataframe (for correct column order)
        input_df = pd.DataFrame([input_values], columns=features)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template("result.html", predicted_price=prediction)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
