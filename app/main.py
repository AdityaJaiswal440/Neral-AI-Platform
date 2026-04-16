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

# 1. Security & App Initialization
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
app = FastAPI(title="Neral AI: Unified Churn Core", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Invalid Security Key")

# 2. Global Registry
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(payload: dict, sector: str):
    """The Iron Mask: Rebuilding data with absolute type isolation."""
    
    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload:
                val = payload[k]
                return val if val is not None and str(val).lower() != 'nan' else default
        return default

    # Categorical Map
    cat_cols = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    # Dictionary for engineered values
    eng = {}
    if sector == 'aviation':
        dist = float(pd.to_numeric(get_raw('Flight_Distance'), errors='coerce') or 0)
        dep_d = float(pd.to_numeric(get_raw('Departure_Delay_Minutes'), errors='coerce') or 0)
        arr_d = float(pd.to_numeric(get_raw('Arrival_Delay_Minutes'), errors='coerce') or 0)
        eng.update({
            'distance_log': np.log1p(dist),
            'delay_intensity_log': np.log1p(dep_d + arr_d),
            'is_business_travel': 1.0 if str(get_raw('Type of Travel')) == 'Business travel' else 0.0,
            'is_disloyal_customer': 1.0 if str(get_raw('Customer Type')) == 'disloyal Customer' else 0.0,
            'csat_score': 3.0 # Default
        })
        # Add ratings
        for r in ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']:
            eng[r] = float(pd.to_numeric(get_raw(r), errors='coerce') or 3.0)
        eng['csat_score'] = sum(eng[r] for r in ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']) / 4.0
        eng['loyalty_shock_score'] = eng['csat_score'] * eng['is_disloyal_customer']
        eng['service_friction_score'] = (10.0 - eng['Inflight wifi service'] - eng['Online boarding'])
        for s in ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']:
            eng[s] = float(pd.to_numeric(get_raw(s), errors='coerce') or 3.0)

    elif sector == 'ecommerce':
        m_charges = float(pd.to_numeric(get_raw('Monthly_Charges'), errors='coerce') or 30.0)
        tenure = float(pd.to_numeric(get_raw('Tenure'), errors='coerce') or 1.0)
        usage = float(pd.to_numeric(get_raw('Total_Usage_GB'), errors='coerce') or 100.0)
        logins = float(pd.to_numeric(get_raw('monthly_logins'), errors='coerce') or 1.0)
        time = float(pd.to_numeric(get_raw('total_monthly_time'), errors='coerce') or 500.0)
        feats = float(pd.to_numeric(get_raw('features_used'), errors='coerce') or 2.0)
        eng.update({
            'monthly_fee_log': np.log1p(m_charges),
            'value_score_log': np.log1p(usage * tenure),
            'last_login_days_log': np.log1p(float(pd.to_numeric(get_raw('Last_Login_Days'), errors='coerce') or 5.0)),
            'feature_intensity_log': np.log1p(feats),
            'support_intensity_log': np.log1p(float(pd.to_numeric(get_raw('support_tickets'), errors='coerce') or 0.0)),
            'session_strength_log': np.log1p(time / (logins + 1.0)),
            'csat_score': float(pd.to_numeric(get_raw('csat'), errors='coerce') or 3.0),
            'nps_normalized': (float(pd.to_numeric(get_raw('nps'), errors='coerce') or 7.0) / 10.0),
            'is_zombie_user': 1.0 if logins < 2.0 else 0.0,
            'engagement_efficiency': time / (feats + 1.0)
        })
        eng['loyalty_shock_score'] = (1.0 - eng['nps_normalized']) * (1.0 / (tenure + 1.0))
        eng['loyalty_resilience'] = tenure * eng['csat_score']
        for n in ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk', 'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 'usage_growth_rate', 'email_open_rate_fixed']:
            eng[n] = float(pd.to_numeric(get_raw(n.split('_')[0]), errors='coerce') or 0.0)

    # Reconstruct Final Row
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        if col in cat_cols:
            val = get_raw(col, "unknown")
            final_dict[col] = str(val).strip()
        else:
            val = eng.get(col, get_raw(col, 0.0))
            final_dict[col] = float(pd.to_numeric(val, errors='coerce') or 0.0)

    # FINAL TYPE LOCKING
    final_df = pd.DataFrame([final_dict])
    for col in required:
        if col in cat_cols:
            final_df[col] = final_df[col].astype(object) # Forces NumPy string-check bypass
        else:
            final_df[col] = final_df[col].astype(np.float64)
            
    return final_df

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
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    
    # Process
    df_ready = apply_feature_engineering(payload.features, sector)
    
    try:
        processed = PREPROCESSORS[sector].transform(df_ready)
        prob = MODELS[sector].predict_proba(processed)[0, 1]
        
        # Diagnosis
        all_feat = PREPROCESSORS[sector].get_feature_names_out()
        features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
        shap_vals = EXPLAINERS[sector](pd.DataFrame(processed, columns=features_clean))
        top_driver = features_clean[np.argmax(np.abs(shap_vals.values[0]))]
        
        return {
            "prediction_id": str(uuid.uuid4()),
            "probability": round(float(prob), 4),
            "trigger_diagnosis": top_driver,
            "status": "success"
        }
    except Exception as e:
        # Final debugging line if this still fails
        print(f"TRANSFORM ERROR ON COLUMNS: {df_ready.dtypes}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)