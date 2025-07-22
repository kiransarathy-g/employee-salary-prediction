import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import joblib
import os

# Load and clean dataset
df = pd.read_csv("adult 3.csv").dropna()
# Convert income to numeric target
df['income'] = df['income'].astype(str).str.strip().str.lower().str.replace('.', '', regex=False)

# Print unique values BEFORE filtering
print("🔍 Unique raw income values:", df['income'].unique())

# Filter to only rows with known values
valid_income_map = {
    '<=50k': 25000,
    '50k': 50000,
    '>50k': 85000
}


# Replace with numeric values
df['income'] = df['income'].map(valid_income_map)

# Remove outliers from income
'''Q1, Q3 = df['income'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['income'] >= Q1 - 1.5 * IQR) & (df['income'] <= Q3 + 1.5 * IQR)]
'''
# ✅ Selected important features including race and sex
selected_features = ['education', 'occupation', 'hours-per-week', 'age',
                     'capital-gain', 'workclass', 'marital-status', 'race', 'gender']
X = df[selected_features].copy()
y = df['income']

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost with GridSearchCV
model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Save model and artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/salary_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")  # ✅ Save features separately

print("✅ Trained and saved model using features:", X.columns.tolist())
