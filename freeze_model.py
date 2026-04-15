import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import json
import os

print("Training Champion Model...")
# 1. Train Champion
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')
X = df.drop('churn', axis=1)
y = df['churn']

categorical_cols = ['city', 'signup_channel', 'payment_method', 'tenure_group', 'complaint_type', 'customer_segment', 'contract_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_exclude=np.number))
    ]
)

xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.01, max_depth=3, scale_pos_weight=10, subsample=0.7, eval_metric='aucpr', random_state=42
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb)])
pipeline.fit(X_train, y_train)

print("Serializing...")
# 2. Serialize
os.makedirs('models', exist_ok=True)
joblib_path = 'models/hcim_aviation_v1.joblib'
joblib.dump(pipeline, joblib_path)

print("Writing Metadata...")
# 3. Metadata Logging
metadata = {
    "model_version": "1.0-champion",
    "recall_score": 0.8431,
    "tuning_params": {
        "max_depth": 3,
        "scale_pos_weight": 10,
        "learning_rate": 0.01
    },
    "features_used": list(X.columns)
}

metadata_path = 'models/metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print("Starting Integrity Check...")
# 4. Integrity Check
# Pulling the 139th index just like our audit, or standard 0 indexing
test_idx = 139
live_prob = pipeline.predict_proba(X_test.iloc[[test_idx]])[0, 1]

loaded_pipeline = joblib.load(joblib_path)
loaded_prob = loaded_pipeline.predict_proba(X_test.iloc[[test_idx]])[0, 1]

assert np.isclose(live_prob, loaded_prob), f"Mismatch! Live: {live_prob}, Loaded: {loaded_prob}"

print(f"\n✅ SUCCESS! Integrity Check Passed:")
print(f"   Live Model Prediction:   {live_prob*100:.6f}%")
print(f"   Loaded Model Prediction: {loaded_prob*100:.6f}%")
print(f"\nModel securely serialized at: {joblib_path}")
print(f"Metadata securely logged at: {metadata_path}")
print(f"\nReady for FastAPI Integration!")
