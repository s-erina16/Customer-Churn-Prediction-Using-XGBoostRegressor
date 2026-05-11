import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ============================
# 1. Load trained model
# ============================
model = joblib.load("xgboost_model.pkl")
print("✅ Model loaded successfully!")

# ============================
# 2. Load dataset to fit encoders
# ============================
file_path = r"E:\priya\2025 - 2026\CSI College Ooty\ME CSE\Sowndarya M\churn\code\Churn-in-Telecom Dataset (3333).csv"
df = pd.read_csv(file_path)

# ============================
# 3. Prepare encoders
# ============================
state_enc = LabelEncoder()
international_plan_enc = LabelEncoder()
voice_mail_plan_enc = LabelEncoder()

# Convert to lowercase for consistent mapping
df['international plan'] = df['international plan'].str.lower()
df['voice mail plan'] = df['voice mail plan'].str.lower()

state_enc.fit(df['state'])
international_plan_enc.fit(df['international plan'])
voice_mail_plan_enc.fit(df['voice mail plan'])

# ============================
# 4. Get manual input from user
# ============================
state = input("Enter State (e.g., KS, NJ, CA): ").strip()
account_length = float(input("Enter Account Length: "))
area_code = int(input("Enter Area Code: "))
international_plan = input("International Plan (Yes/No): ").strip().lower()
voice_mail_plan = input("Voice Mail Plan (Yes/No): ").strip().lower()
number_vmail_messages = float(input("Number of Voice Mail Messages: "))
total_day_minutes = float(input("Total Day Minutes: "))
total_day_calls = int(input("Total Day Calls: "))
total_day_charge = float(input("Total Day Charge: "))
total_eve_minutes = float(input("Total Eve Minutes: "))
total_eve_calls = int(input("Total Eve Calls: "))
total_eve_charge = float(input("Total Eve Charge: "))
total_night_minutes = float(input("Total Night Minutes: "))
total_night_calls = int(input("Total Night Calls: "))
total_night_charge = float(input("Total Night Charge: "))
total_intl_minutes = float(input("Total Intl Minutes: "))
total_intl_calls = int(input("Total Intl Calls: "))
total_intl_charge = float(input("Total Intl Charge: "))
customer_service_calls = int(input("Customer Service Calls: "))

# ============================
# 5. Encode categorical inputs
# ============================
state_encoded = state_enc.transform([state])[0]
international_plan_encoded = international_plan_enc.transform([international_plan])[0]
voice_mail_plan_encoded = voice_mail_plan_enc.transform([voice_mail_plan])[0]

# ============================
# 6. Create input array
# ============================
sample = np.array([
    state_encoded, account_length, area_code, international_plan_encoded, voice_mail_plan_encoded,
    number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
    total_eve_minutes, total_eve_calls, total_eve_charge, total_night_minutes,
    total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls,
    total_intl_charge, customer_service_calls
]).reshape(1, -1)

# ============================
# 7. Make prediction
# ============================
prediction = model.predict(sample)

# ============================
# 8. Print Churn or Non Churn
# ============================
if prediction[0] == 1:
    print("\n🔮 Prediction: Churn")
else:
    print("\n🔮 Prediction: Non Churn")
