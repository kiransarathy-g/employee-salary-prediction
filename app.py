from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and preprocessing tools
model = joblib.load('model/salary_model.pkl')
encoders = joblib.load('model/label_encoders.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')  # ✅ load correct features

# Load dataset to populate dropdowns for categorical fields
df = pd.read_csv("adult 3.csv").dropna()
dropdowns = {
    col: sorted(df[col].unique())
    for col in feature_names if col in df.columns and df[col].dtype == 'object'
}

@app.route('/')
def home():
    return render_template('index.html', dropdowns=dropdowns, feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for field in feature_names:
        value = request.form[field]
        if field in encoders:
            input_data[field] = encoders[field].transform([value])[0]
        else:
            input_data[field] = float(value)

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    predicted_salary = round(model.predict(input_scaled)[0], 2)
    formatted_salary = f"₹{predicted_salary:,.2f}"
    return render_template('index.html', dropdowns=dropdowns, feature_names=feature_names,result=formatted_salary, predicted_value=predicted_salary)


if __name__ == '__main__':
    app.run(debug=True)
