
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor



data = pd.read_csv("Dataset2.csv")

print("Dataset Loaded Successfully")
print(data.head())



# Remove unnecessary columns
data = data.drop(columns=["day", "month", "year"], errors='ignore')

# Handle missing values (if any)
data = data.dropna()

# Convert 'Classes' column (if exists)
if "Classes" in data.columns:
    data["Classes"] = data["Classes"].map({"fire": 1, "not fire": 0})



# Input Features
X = data[["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]]

# Target
y = data["FWI"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)



xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Performance:")
    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("R2 Score:", round(r2, 3))


evaluate(rf_model, X_test, y_test, "Random Forest")
evaluate(xgb_model, X_test, y_test, "XGBoost")


joblib.dump(rf_model, "model_rf.pkl")
joblib.dump(xgb_model, "model_xgb.pkl")

print("\nModels saved successfully!")