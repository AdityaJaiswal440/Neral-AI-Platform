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
    """Surgical feature replication with total Type Isolation to prevent isnan errors."""
    
    def get_raw(key):
        # Universal lookup: check key, underscore-key, space-key, and lowercase
        return df.get(key, df.get(key.replace(' ', '_'), df.get(key.replace('_', ' '), df.get(key.lower(), None))))

    def f_num(key, default=0.0):
        val = get_raw(key)
        if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else default
        try: return float(val) if val is not None else default
        except: return default

    def f_str(key, default="unknown"):
        val = get_raw(key)
        if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else default
        res = str(val).strip()
        return res if (res.lower() not in ['nan', 'none', 'null', '0', '0.0']) else default

    # Define Sector-Specific Schemas
    if sector == 'aviation':
        cat_cols = ['Class', 'Customer Type', 'Type of Travel']
        # Engineering
        df['distance_log'] = np.log1p(f_num('Flight_Distance'))
        df['delay_intensity_log'] = np.log1p(f_num('Departure_Delay_Minutes') + f_num('Arrival_Delay_Minutes'))
        df['is_business_travel'] = 1 if f_str('Type of Travel') == 'Business travel' else 0
        df['is_disloyal_customer'] = 1 if f_str('Customer Type') == 'disloyal Customer' else 0
        
        rating_cols = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for col in rating_cols: df[col] = f_num(col, 3.0)
        df['csat_score'] = df[rating_cols].mean(axis=1)
        df['loyalty_shock_score'] = df['csat_score'] * df['is_disloyal_customer']
        df['service_friction_score'] = (5 - df['Inflight wifi service']) + (5 - df['Online boarding'])
        
        std_cols = ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
                    'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']
        for col in std_cols: df[col] = f_num(col, 3.0)
        for col in cat_cols: df[col] = f_str(col)

    elif sector == 'ecommerce':
        cat_cols = ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
        
        # Numeric Engineering
        df['monthly_fee_log'] = np.log1p(f_num('Monthly_Charges', 30.0))
        df['value_score_log'] = np.log1p(f_num('Total_Usage_GB', 100.0) * f_num('Tenure', 1.0))
        df['last_login_days_log'] = np.log1p(f_num('Last_Login_Days', 5.0))
        df['feature_intensity_log'] = np.log1p(f_num('features_used', 2.0))
        df['support_intensity_log'] = np.log1p(f_num('support_tickets', 0.0))
        df['session_strength_log'] = np.log1p(f_num('total_monthly_time', 500.0) / (f_num('monthly_logins', 1.0) + 1.0))
        
        df['csat_score'] = f_num('csat', 3.0)
        df['nps_normalized'] = f_num('nps', 7.0) / 10.0
        df['loyalty_shock_score'] = (1 - df['nps_normalized']) * (1 / (f_num('Tenure', 1.0) + 1))
        df['loyalty_resilience'] = f_num('Tenure', 1.0) * df['csat_score']
        
        # Binary & Standard Numerical
        num_cols = ['support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped',
                    'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 'weekly_active_days', 
                    'usage_density', 'usage_growth_rate', 'email_open_rate_fixed', 'is_zombie_user', 'is_slow_ghost', 
                    'is_passive_promoter', 'is_advocate', 'is_recency_danger', 'is_high_friction_payment', 
                    'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk']
        
        for col in num_cols:
            clean_key = col.split('_')[0] if '_' in col else col
            df[col] = f_num(clean_key)
        for col in cat_cols: df[col] = f_str(col)

    # FINAL TYPE ENFORCEMENT SHIELD
    required_cols = PREPROCESSORS[sector].feature_names_in_
    for col in required_cols:
        if col not in df.columns:
            df[col] = "unknown" if (sector == 'ecommerce' and col in cat_cols) else 0.0
        
        # Explicitly cast to prevent mixed-type isnan crashes
        if col in cat_cols or (sector == 'aviation' and col in ['Class', 'Customer Type', 'Type of Travel']):
            df[col] = df[col].astype(str)
        else:
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
    df_raw = pd.DataFrame([payload.features])
    df_ready = apply_feature_engineering(df_raw, sector)
    
    # Inference
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)