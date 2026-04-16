import os
import joblib
import pandas as pd
import numpy as np
import shap
import uuid
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# 1. Configuration & Security
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Neral AI: Hybrid Churn Intelligence", version="1.0")

# 2. Security Layer
def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Invalid or Missing API Key")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 3. Global Registry
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(df: pd.DataFrame, sector: str):
    """Surgical feature replication with total Type Isolation."""
    
    # Helper to get raw values regardless of naming convention
    def get_raw(key, default=None):
        return df.get(key, df.get(key.replace('_', ' '), df.get(key.replace(' ', '_'), default)))

    # Schema Definitions
    cat_features = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }
    cats = cat_features.get(sector, [])

    # 1. Feature Engineering: Aviation
    if sector == 'aviation':
        df['distance_log'] = np.log1p(float(pd.to_numeric(get_raw('Flight_Distance'), errors='coerce') or 0))
        df['delay_intensity_log'] = np.log1p(float(pd.to_numeric(get_raw('Departure_Delay_Minutes'), errors='coerce') or 0) + 
                                            float(pd.to_numeric(get_raw('Arrival_Delay_Minutes'), errors='coerce') or 0))
        df['is_business_travel'] = 1 if str(get_raw('Type of Travel')) == 'Business travel' else 0
        df['is_disloyal_customer'] = 1 if str(get_raw('Customer Type')) == 'disloyal Customer' else 0
        
        # Ratings
        rating_cols = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for col in rating_cols: df[col] = pd.to_numeric(get_raw(col), errors='coerce') or 3.0
        df['csat_score'] = df[rating_cols].mean(axis=1)
        df['loyalty_shock_score'] = df['csat_score'] * df['is_disloyal_customer']
        df['service_friction_score'] = (5 - df['Inflight wifi service']) + (5 - df['Online boarding'])

    # 2. Feature Engineering: E-commerce
    elif sector == 'ecommerce':
        df['monthly_fee_log'] = np.log1p(float(pd.to_numeric(get_raw('Monthly_Charges'), errors='coerce') or 30))
        df['value_score_log'] = np.log1p(float(pd.to_numeric(get_raw('Total_Usage_GB'), errors='coerce') or 100) * float(pd.to_numeric(get_raw('Tenure'), errors='coerce') or 1))
        df['last_login_days_log'] = np.log1p(float(pd.to_numeric(get_raw('Last_Login_Days'), errors='coerce') or 5))
        
        df['is_zombie_user'] = 1 if (pd.to_numeric(get_raw('monthly_logins'), errors='coerce') or 1) < 2 else 0
        df['csat_score'] = float(pd.to_numeric(get_raw('csat'), errors='coerce') or 3)
        df['nps_normalized'] = (pd.to_numeric(get_raw('nps'), errors='coerce') or 7) / 10.0
        df['loyalty_shock_score'] = (1 - df['nps_normalized']) * (1 / ((pd.to_numeric(get_raw('Tenure'), errors='coerce') or 1) + 1))
        df['loyalty_resilience'] = (pd.to_numeric(get_raw('Tenure'), errors='coerce') or 1) * df['csat_score']

    # 3. TYPE-LOCKING SHIELD: Final alignment with the Preprocessor's expectations
    required_cols = PREPROCESSORS[sector].feature_names_in_
    for col in required_cols:
        if col not in df.columns:
            # Categorical missing = "unknown", Numerical missing = 0.0
            df[col] = "unknown" if col in cats else 0.0
        
        if col in cats:
            # Force to string and handle NaN strings
            df[col] = df[col].astype(str).replace(['nan', 'None', 'NaN', '0.0', '0'], 'unknown')
        else:
            # Force to float
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
                
    return df[required_cols]

@app.on_event("startup")
def load_models():
    sectors = {'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 'aviation': 'models/hcim_Aviation_v1.joblib'}
    for sector, path in sectors.items():
        if os.path.exists(path):
            pipeline = joblib.load(path)
            MODELS[sector] = pipeline.named_steps['classifier']
            PREPROCESSORS[sector] = pipeline.named_steps['preprocessor']
            EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
            print(f"Loaded {sector} logic...")
    print("Unified Core Online. 16GB Memory Engine Engaged.")

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

def extract_top_driver(sector: str, df_final: pd.DataFrame) -> str:
    transformed = PREPROCESSORS[sector].transform(df_final)
    all_feat = PREPROCESSORS[sector].get_feature_names_out()
    features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
    df_shap = pd.DataFrame(transformed, columns=features_clean)
    shap_vals = EXPLAINERS[sector](df_shap)
    top_idx = np.argmax(np.abs(shap_vals.values[0]))
    return features_clean[top_idx]

@app.post("/v1/predict")
@app.post("/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid sector")
    
    df_raw = pd.DataFrame([payload.features])
    df_ready = apply_feature_engineering(df_raw, sector)
    
    # The moment of truth: Transform and Predict
    processed_data = PREPROCESSORS[sector].transform(df_ready)
    prob = MODELS[sector].predict_proba(processed_data)[0, 1]
    top_driver = extract_top_driver(sector, df_ready)
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": round(float(prob), 4),
        "trigger_diagnosis": top_driver,
        "prescriptive_rescue": "High Engagement Intervention" if prob > 0.5 else "Standard Review",
        "status": "success"
    }