from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ============================
# Load trained model and encoders
# ============================
model = joblib.load("xgboost_model.pkl")

# Load dataset to fit encoders
df = pd.read_csv(r"Churn-in-Telecom Dataset (3333).csv")
df['international plan'] = df['international plan'].str.lower()
df['voice mail plan'] = df['voice mail plan'].str.lower()

state_enc = LabelEncoder()
state_enc.fit(df['state'])

international_plan_enc = LabelEncoder()
international_plan_enc.fit(df['international plan'])

voice_mail_plan_enc = LabelEncoder()
voice_mail_plan_enc.fit(df['voice mail plan'])

# ============================
# Home page
# ============================
@app.route('/')
def home():
    return render_template('home.html')

# ============================
# About page
# ============================
@app.route('/about')
def about():
    return render_template('about.html')

# ============================
# Predict page
# ============================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            state = request.form['state']
            account_length = float(request.form['account_length'])
            area_code = int(request.form['area_code'])
            international_plan = request.form['international_plan'].lower()
            voice_mail_plan = request.form['voice_mail_plan'].lower()
            number_vmail_messages = float(request.form['number_vmail_messages'])
            total_day_minutes = float(request.form['total_day_minutes'])
            total_day_calls = int(request.form['total_day_calls'])
            total_day_charge = float(request.form['total_day_charge'])
            total_eve_minutes = float(request.form['total_eve_minutes'])
            total_eve_calls = int(request.form['total_eve_calls'])
            total_eve_charge = float(request.form['total_eve_charge'])
            total_night_minutes = float(request.form['total_night_minutes'])
            total_night_calls = int(request.form['total_night_calls'])
            total_night_charge = float(request.form['total_night_charge'])
            total_intl_minutes = float(request.form['total_intl_minutes'])
            total_intl_calls = int(request.form['total_intl_calls'])
            total_intl_charge = float(request.form['total_intl_charge'])
            customer_service_calls = int(request.form['customer_service_calls'])

            # Encode categorical features
            state_encoded = state_enc.transform([state])[0]
            international_plan_encoded = international_plan_enc.transform([international_plan])[0]
            voice_mail_plan_encoded = voice_mail_plan_enc.transform([voice_mail_plan])[0]

            # Prepare input for model
            sample = np.array([
                state_encoded, account_length, area_code, international_plan_encoded, voice_mail_plan_encoded,
                number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
                total_eve_minutes, total_eve_calls, total_eve_charge, total_night_minutes,
                total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls,
                total_intl_charge, customer_service_calls
            ]).reshape(1, -1)

            # Make prediction
            prediction = model.predict(sample)[0]
            result = "Churn" if prediction == 1 else "Non Churn"

            return render_template('result.html', prediction=result)

        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')

# ============================
# Run Flask app
# ============================
if __name__ == '__main__':
    app.run(debug=True)
