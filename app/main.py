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
    """Defensive feature replication with Strict Type-Locking."""
    
    def get_raw(key, default=None):
        return df.get(key, df.get(key.replace(' ', '_'), df.get(key.replace('_', ' '), default)))

    def f_num(key, default=0.0):
        val = get_raw(key, default)
        if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else default
        try: return float(val) if val is not None else default
        except: return default

    def f_str(key, default="unknown"):
        val = get_raw(key, default)
        if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else default
        return str(val) if val is not None else default

    if sector == 'aviation':
        df['distance_log'] = np.log1p(f_num('Flight_Distance'))
        df['delay_intensity_log'] = np.log1p(f_num('Departure_Delay_Minutes') + f_num('Arrival_Delay_Minutes'))
        df['is_business_travel'] = 1 if f_str('Type_of_Travel') == 'Business travel' else 0
        df['is_disloyal_customer'] = 1 if f_str('Customer_Type') == 'disloyal Customer' else 0
        
        core_ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for col in core_ratings: df[col] = f_num(col, 3.0)

        df['csat_score'] = df[core_ratings].mean(axis=1)
        df['loyalty_shock_score'] = df['csat_score'] * df['is_disloyal_customer']
        df['service_friction_score'] = (5 - df['Inflight wifi service']) + (5 - df['Online boarding'])
        
        std_cols = ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
                    'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']
        for col in std_cols: df[col] = f_num(col, 3.0)

    elif sector == 'ecommerce':
        # Numerical Blocks
        df['monthly_fee_log'] = np.log1p(f_num('Monthly_Charges', 30.0))
        df['value_score_log'] = np.log1p(f_num('Total_Usage_GB', 100.0) * f_num('Tenure', 1.0))
        df['last_login_days_log'] = np.log1p(f_num('Last_Login_Days', 5.0))
        df['feature_intensity_log'] = np.log1p(f_num('features_used', 2.0))
        df['support_intensity_log'] = np.log1p(f_num('support_tickets', 0.0))
        df['session_strength_log'] = np.log1p(f_num('total_monthly_time', 500.0) / (f_num('monthly_logins', 1.0) + 1.0))
        
        for col in ['support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped']:
            df[col] = f_num(col.replace('_clipped', ''))

        # Boolean Flags
        df['is_zombie_user'] = 1 if f_num('monthly_logins') < 2 else 0
        df['is_slow_ghost'] = 1 if f_num('total_monthly_time') < 100 and f_num('Tenure') > 6 else 0
        df['is_passive_promoter'] = 1 if f_num('nps', 7) in [7, 8] else 0
        df['is_advocate'] = 1 if f_num('nps', 7) > 8 else 0
        df['is_recency_danger'] = 1 if f_num('Last_Login_Days') > 15 else 0
        df['is_high_friction_payment'] = 1 if f_str('payment_method') == 'Mailed Check' else 0
        df['is_bouncer'] = 1 if f_num('email_open_rate') < 0.05 else 0
        df['is_hidden_dissatisfaction'] = 1 if f_num('csat', 3) < 3 and f_num('support_tickets') == 0 else 0

        # Calculated Scores
        df['csat_score'] = f_num('csat', 3.0)
        df['nps_normalized'] = f_num('nps', 7.0) / 10.0
        df['loyalty_shock_score'] = (1 - df['nps_normalized']) * (1 / (f_num('Tenure', 1.0) + 1))
        df['engagement_efficiency'] = f_num('total_monthly_time') / (f_num('features_used') + 1)
        df['usage_growth_rate'] = f_num('usage_delta') 
        df['loyalty_resilience'] = f_num('Tenure', 1.0) * df['csat_score']
        df['email_open_rate_fixed'] = f_num('email_open_rate', 0.2)
        df['payment_structural_risk'] = 1 if f_num('payment_failures') > 1 else 0

        # Categorical Block (Type-Locked as Strings)
        cat_cols = ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
        for col in cat_cols: df[col] = f_str(col)
        
        # Raw Numerical Block
        raw_num = ['discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 'weekly_active_days', 'usage_density']
        for col in raw_num: df[col] = f_num(col)

    # Final Schema Shield
    required = PREPROCESSORS[sector].feature_names_in_
    for col in required:
        if col not in df.columns:
            df[col] = 0.0 # Numeric default to avoid isnan crash
                
    return df

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