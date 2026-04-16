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
# Ensure NERAL_SECRET is set in your Hugging Face Space Settings
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Neral AI: Unified Churn Core", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Security Layer
def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Access Denied. Invalid API Key.")

# 3. Global Model Registry
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(df_in: pd.DataFrame, sector: str):
    """The Iron Mask: Total type segregation to eliminate isnan TypeErrors."""
    
    # Initialize empty dictionary for the new row
    # Reconstructing from scratch prevents Pandas 'object' type bleeding
    engineered = {}

    def get_raw(key, default=None):
        # Universal lookup: check key, underscore, space, or lowercase variant
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in df_in.columns:
                val = df_in[k].iloc[0]
                return val if pd.notna(val) else default
        return default

    # A. SECTOR LOGIC: Aviation
    if sector == 'aviation':
        dist = float(pd.to_numeric(get_raw('Flight_Distance'), errors='coerce') or 0)
        dep_d = float(pd.to_numeric(get_raw('Departure_Delay_Minutes'), errors='coerce') or 0)
        arr_d = float(pd.to_numeric(get_raw('Arrival_Delay_Minutes'), errors='coerce') or 0)
        
        engineered['distance_log'] = np.log1p(dist)
        engineered['delay_intensity_log'] = np.log1p(dep_d + arr_d)
        engineered['is_business_travel'] = 1.0 if str(get_raw('Type of Travel')).strip() == 'Business travel' else 0.0
        engineered['is_disloyal_customer'] = 1.0 if str(get_raw('Customer Type')).strip() == 'disloyal Customer' else 0.0
        
        ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for r in ratings: engineered[r] = float(pd.to_numeric(get_raw(r), errors='coerce') or 3.0)
        
        engineered['csat_score'] = sum(engineered[r] for r in ratings) / 4.0
        engineered['loyalty_shock_score'] = engineered['csat_score'] * engineered['is_disloyal_customer']
        engineered['service_friction_score'] = (5.0 - engineered['Inflight wifi service']) + (5.0 - engineered['Online boarding'])
        
        std = ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
               'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']
        for s in std: engineered[s] = float(pd.to_numeric(get_raw(s), errors='coerce') or 3.0)
        
        # Categoricals (Force to String)
        for c in ['Class', 'Customer Type', 'Type of Travel']:
            val = get_raw(c)
            engineered[c] = str(val) if val is not None else "unknown"

    # B. SECTOR LOGIC: E-commerce
    elif sector == 'ecommerce':
        m_charges = float(pd.to_numeric(get_raw('Monthly_Charges'), errors='coerce') or 30.0)
        tenure = float(pd.to_numeric(get_raw('Tenure'), errors='coerce') or 1.0)
        usage = float(pd.to_numeric(get_raw('Total_Usage_GB'), errors='coerce') or 100.0)
        logins = float(pd.to_numeric(get_raw('monthly_logins'), errors='coerce') or 1.0)
        
        engineered['monthly_fee_log'] = np.log1p(m_charges)
        engineered['value_score_log'] = np.log1p(usage * tenure)
        engineered['last_login_days_log'] = np.log1p(float(pd.to_numeric(get_raw('Last_Login_Days'), errors='coerce') or 5.0))
        engineered['feature_intensity_log'] = np.log1p(float(pd.to_numeric(get_raw('features_used'), errors='coerce') or 2.0))
        engineered['support_intensity_log'] = np.log1p(float(pd.to_numeric(get_raw('support_tickets'), errors='coerce') or 0.0))
        engineered['session_strength_log'] = np.log1p(500.0 / (logins + 1.0))
        
        csat = float(pd.to_numeric(get_raw('csat', 3.0), errors='coerce') or 3.0)
        engineered['csat_score'] = csat
        nps = float(pd.to_numeric(get_raw('nps', 7.0), errors='coerce') or 7.0)
        engineered['nps_normalized'] = nps / 10.0
        engineered['loyalty_shock_score'] = (1.0 - (nps/10.0)) * (1.0 / (tenure + 1.0))
        engineered['loyalty_resilience'] = tenure * csat
        engineered['is_zombie_user'] = 1.0 if logins < 2.0 else 0.0

        # Map all 38 required numerical features
        num_feats = ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 
                     'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 
                     'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk',
                     'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 
                     'usage_growth_rate', 'email_open_rate_fixed', 'engagement_efficiency']
        for n in num_feats:
            clean_key = n.split('_')[0]
            engineered[n] = float(pd.to_numeric(get_raw(clean_key), errors='coerce') or 0.0)
            
        # Categoricals (Force to String)
        cat_cols = ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
        for c in cat_cols:
            val = get_raw(c)
            engineered[c] = str(val) if val is not None else "unknown"

    # C. FINAL HARDENING: Reconstruct DataFrame with Immutable Types
    required = PREPROCESSORS[sector].feature_names_in_
    final_row = {}
    
    for col in required:
        val = engineered.get(col)
        # Type-check based on content
        if isinstance(val, str):
            final_row[col] = val if val.lower() != 'nan' else 'unknown'
        else:
            final_row[col] = float(val) if val is not None else 0.0

    # Build the final DF and lock dtypes
    final_df = pd.DataFrame([final_row])
    for col in required:
        if isinstance(final_row[col], str):
            final_df[col] = final_df[col].astype(str)
        else:
            final_df[col] = final_df[col].astype(np.float64)
            
    return final_df[required]

@app.on_event("startup")
def load_models():
    paths = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 
        'aviation': 'models/hcim_Aviation_v1.joblib'
    }
    for sector, path in paths.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            MODELS[sector] = pipe.named_steps['classifier']
            PREPROCESSORS[sector] = pipe.named_steps['preprocessor']
            EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
            print(f"Loaded {sector} logic...")
    print("Unified Core Online. 16GB Memory Engine Engaged.")

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

@app.post("/v1/predict")
@app.post("/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid Sector. Valid options: aviation, ecommerce")
    
    # 1. Surgical Engineering
    df_ready = apply_feature_engineering(pd.DataFrame([payload.features]), sector)
    
    # 2. Inference
    processed = PREPROCESSORS[sector].transform(df_ready)
    prob = MODELS[sector].predict_proba(processed)[0, 1]
    
    # 3. Explainability (SHAP)
    all_feat = PREPROCESSORS[sector].get_feature_names_out()
    features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
    shap_vals = EXPLAINERS[sector](pd.DataFrame(processed, columns=features_clean))
    top_idx = np.argmax(np.abs(shap_vals.values[0]))
    top_driver = features_clean[top_idx]
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": round(float(prob), 4),
        "trigger_diagnosis": top_driver,
        "prescriptive_rescue": "High Engagement Intervention" if prob > 0.5 else "Standard Review",
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    # Listening on 8000 for Hugging Face internal mapping
    uvicorn.run(app, host="0.0.0.0", port=8000)