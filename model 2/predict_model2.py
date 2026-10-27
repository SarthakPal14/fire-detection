
import joblib
import pandas as pd

print("Enter Data")


import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf_model_path = os.path.join(BASE_DIR, "model_rf.pkl")
xgb_model_path = os.path.join(BASE_DIR, "model_xgb.pkl")

rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

print("Models loaded successfully!")



temp = float(input("Enter Temperature (°C): "))
rh = float(input("Enter Humidity (%): "))
ws = float(input("Enter Wind Speed (km/h): "))
rain = float(input("Enter Rain (mm): "))

ffmc = float(input("Enter FFMC: "))
dmc = float(input("Enter DMC: "))
dc = float(input("Enter DC: "))
isi = float(input("Enter ISI: "))
bui = float(input("Enter BUI: "))


columns = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]

data = pd.DataFrame([[temp, rh, ws, rain, ffmc, dmc, dc, isi, bui]], columns=columns)



rf_pred = rf_model.predict(data)[0]
xgb_pred = xgb_model.predict(data)[0]

combined_pred = (rf_pred + xgb_pred) / 2



def get_risk(value):
    if value < 5:
        return "Low"
    elif value < 10:
        return "Medium"
    else:
        return "High"

rf_risk = get_risk(rf_pred)
xgb_risk = get_risk(xgb_pred)
combined_risk = get_risk(combined_pred)



print("\n🔥 FIRE RISK ANALYSIS")

print("\n🌳 Random Forest:")
print(f"FWI: {round(rf_pred, 2)} | Risk: {rf_risk}")

print("\n⚡ XGBoost:")
print(f"FWI: {round(xgb_pred, 2)} | Risk: {xgb_risk}")

print("\n🤝 Combined Prediction:")
print(f"FWI: {round(combined_pred, 2)} | Risk: {combined_risk}")



if combined_risk == "High":
    print("\n🚨 ALERT: High Fire Risk!")
elif combined_risk == "Medium":
    print("\n⚠️ Warning: Moderate Risk")
else:
    print("\n✅ Safe Conditions")