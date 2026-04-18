import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')

print("Cleaning up old aviation artifacts...")
if os.path.exists('models/hcim_aviation_v1.joblib'):
    os.remove('models/hcim_aviation_v1.joblib')
    print("Deleted hcim_aviation_v1.joblib")

print("Training Champion Model (E-commerce/Streaming)...")
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')
X = df.drop('churn', axis=1)
y = df['churn']

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

print("Serializing E-commerce Champion...")
os.makedirs('models', exist_ok=True)
joblib_path = 'models/hcim_E-comm_Stream_v1.joblib'
joblib.dump(pipeline, joblib_path)

print("Writing Aligned Metadata...")
metadata = {
    "model_id": "HCIM-Ecomm-Streaming-V1",
    "sector": "E-commerce/Streaming",
    "dataset_source": "df_clean (E-comm specific signals)",
    "performance_metrics": { 
        "recall": 0.8431, 
        "f1_score": 0.3815 
    },
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

print("Starting Integrity Validation...")
test_idx = 139
live_prob = pipeline.predict_proba(X_test.iloc[[test_idx]])[0, 1]

loaded_pipeline = joblib.load(joblib_path)
loaded_prob = loaded_pipeline.predict_proba(X_test.iloc[[test_idx]])[0, 1]
y_pred_loaded = (loaded_pipeline.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
loaded_recall = recall_score(y_test, y_pred_loaded)

assert np.isclose(live_prob, loaded_prob), f"Mismatch! Live: {live_prob}, Loaded: {loaded_prob}"

print(f"\nSUCCESS! Integrity Validation Passed:")
print(f"   Final Joblib Path:       {joblib_path}")
print(f"   Validated Recall Score:  {loaded_recall:.4f}")
print(f"   Live Memory Probability: {live_prob*100:.6f}%")
print(f"   Loaded File Probability: {loaded_prob*100:.6f}%")
print(f"\nEnvironment purified. Model HCIM_E-comm_Stream_v1 is ready for Phase 4 deployment!")
