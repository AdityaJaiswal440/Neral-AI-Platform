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

# 1. Config & Security
# ENSURE: Hugging Face Secret NERAL_SECRET = NERAL_SECRET_2026
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
app = FastAPI(title="Neral AI: Absolute Core", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET and x-api-key")

# 2. Global Models
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(df_in: pd.DataFrame, sector: str):
    """Total Type Segregation: The only way to stop ufunc 'isnan' errors."""
    raw = df_in.iloc[0].to_dict()
    
    def get_val(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in raw: return raw[k] if pd.notna(raw[k]) else default
        return default

    # Mapping categorical columns per sector
    cat_lookup = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    engineered = {}

    if sector == 'aviation':
        dist = float(pd.to_numeric(get_val('Flight_Distance'), errors='coerce') or 0)
        dep_d = float(pd.to_numeric(get_val('Departure_Delay_Minutes'), errors='coerce') or 0)
        arr_d = float(pd.to_numeric(get_val('Arrival_Delay_Minutes'), errors='coerce') or 0)
        engineered.update({
            'distance_log': np.log1p(dist),
            'delay_intensity_log': np.log1p(dep_d + arr_d),
            'is_business_travel': 1.0 if str(get_val('Type of Travel')) == 'Business travel' else 0.0,
            'is_disloyal_customer': 1.0 if str(get_val('Customer Type')) == 'disloyal Customer' else 0.0
        })
        ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for r in ratings: engineered[r] = float(pd.to_numeric(get_val(r), errors='coerce') or 3.0)
        engineered['csat_score'] = sum(engineered[r] for r in ratings) / 4.0
        engineered['loyalty_shock_score'] = engineered['csat_score'] * engineered['is_disloyal_customer']
        engineered['service_friction_score'] = (5.0 - engineered['Inflight wifi service']) + (5.0 - engineered['Online boarding'])
        std = ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
               'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']
        for s in std: engineered[s] = float(pd.to_numeric(get_val(s), errors='coerce') or 3.0)

    elif sector == 'ecommerce':
        m_charges = float(pd.to_numeric(get_val('Monthly_Charges'), errors='coerce') or 30.0)
        tenure = float(pd.to_numeric(get_val('Tenure'), errors='coerce') or 1.0)
        usage = float(pd.to_numeric(get_val('Total_Usage_GB'), errors='coerce') or 100.0)
        logins = float(pd.to_numeric(get_val('monthly_logins'), errors='coerce') or 1.0)
        time = float(pd.to_numeric(get_val('total_monthly_time'), errors='coerce') or 500.0)
        feats = float(pd.to_numeric(get_val('features_used'), errors='coerce') or 2.0)
        
        engineered.update({
            'monthly_fee_log': np.log1p(m_charges),
            'value_score_log': np.log1p(usage * tenure),
            'last_login_days_log': np.log1p(float(pd.to_numeric(get_val('Last_Login_Days'), errors='coerce') or 5.0)),
            'feature_intensity_log': np.log1p(feats),
            'support_intensity_log': np.log1p(float(pd.to_numeric(get_val('support_tickets'), errors='coerce') or 0.0)),
            'session_strength_log': np.log1p(time / (logins + 1.0)),
            'csat_score': float(pd.to_numeric(get_val('csat'), errors='coerce') or 3.0),
            'nps_normalized': (pd.to_numeric(get_val('nps'), errors='coerce') or 7.0) / 10.0,
            'is_zombie_user': 1.0 if logins < 2.0 else 0.0,
            'engagement_efficiency': time / (feats + 1.0)
        })
        engineered['loyalty_shock_score'] = (1.0 - engineered['nps_normalized']) * (1 / (tenure + 1))
        engineered['loyalty_resilience'] = tenure * engineered['csat_score']

        num_feats = ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 
                     'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 
                     'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk',
                     'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 
                     'usage_growth_rate', 'email_open_rate_fixed']
        for n in num_feats:
            clean = n.split('_')[0]
            engineered[n] = float(pd.to_numeric(get_val(clean), errors='coerce') or 0.0)

    # Reconstruct the Dataframe to guarantee no mixed-types
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        if col in cat_lookup:
            val = get_val(col, "unknown")
            # The kill-switch: force everything to string and strip spaces
            final_dict[col] = str(val).strip() if val is not None else "unknown"
        else:
            val = engineered.get(col, get_val(col))
            final_dict[col] = float(pd.to_numeric(val, errors='coerce') or 0.0)

    # FINAL BARRIER: Re-create DF with explicit types
    final_df = pd.DataFrame([final_dict])
    for col in required:
        if col in cat_lookup:
            final_df[col] = final_df[col].astype(object) # Forces sklearn to treat as categorical
        else:
            final_df[col] = final_df[col].astype(np.float64)
            
    return final_df[required]

@app.on_event("startup")
def load_models():
    paths = {'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 'aviation': 'models/hcim_Aviation_v1.joblib'}
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
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    
    # 1. Engineering Barrier
    df_ready = apply_feature_engineering(pd.DataFrame([payload.features]), sector)
    
    # 2. Transformation with extra safety
    try:
        processed = PREPROCESSORS[sector].transform(df_ready)
    except Exception as e:
        # Emergency cast to handle unexpected sklearn state
        print(f"DEBUG: Transformation failed, applying emergency cast. Error: {str(e)}")
        for col in df_ready.columns:
            if df_ready[col].dtype == object: df_ready[col] = df_ready[col].astype(str)
        processed = PREPROCESSORS[sector].transform(df_ready)

    # 3. Inference & Explainability
    prob = MODELS[sector].predict_proba(processed)[0, 1]
    all_feat = PREPROCESSORS[sector].get_feature_names_out()
    features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
    shap_vals = EXPLAINERS[sector](pd.DataFrame(processed, columns=features_clean))
    top_driver = features_clean[np.argmax(np.abs(shap_vals.values[0]))]
    
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