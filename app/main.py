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
    raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# 3. Global Registry
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(df_in: pd.DataFrame, sector: str):
    """Total Type Segregation. Kills the isnan TypeError at the source."""
    
    # Initialize dictionary to build a clean, typed row
    new_row = {}

    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in df_in.columns:
                val = df_in[k].iloc[0]
                return val if pd.notna(val) else default
        return default

    # Define strict categorical sets for each sector
    CAT_COLS = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    # Sector Engineering
    if sector == 'aviation':
        dist = float(pd.to_numeric(get_raw('Flight_Distance'), errors='coerce') or 0)
        dep_d = float(pd.to_numeric(get_raw('Departure_Delay_Minutes'), errors='coerce') or 0)
        arr_d = float(pd.to_numeric(get_raw('Arrival_Delay_Minutes'), errors='coerce') or 0)
        
        new_row['distance_log'] = np.log1p(dist)
        new_row['delay_intensity_log'] = np.log1p(dep_d + arr_d)
        new_row['is_business_travel'] = 1.0 if str(get_raw('Type of Travel')).strip() == 'Business travel' else 0.0
        new_row['is_disloyal_customer'] = 1.0 if str(get_raw('Customer Type')).strip() == 'disloyal Customer' else 0.0
        
        ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for r in ratings: new_row[r] = float(pd.to_numeric(get_raw(r), errors='coerce') or 3.0)
        
        new_row['csat_score'] = sum(new_row[r] for r in ratings) / 4.0
        new_row['loyalty_shock_score'] = new_row['csat_score'] * new_row['is_disloyal_customer']
        new_row['service_friction_score'] = (5.0 - new_row['Inflight wifi service']) + (5.0 - new_row['Online boarding'])
        
        std = ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
               'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']
        for s in std: new_row[s] = float(pd.to_numeric(get_raw(s), errors='coerce') or 3.0)

    elif sector == 'ecommerce':
        m_charges = float(pd.to_numeric(get_raw('Monthly_Charges'), errors='coerce') or 30.0)
        tenure = float(pd.to_numeric(get_raw('Tenure'), errors='coerce') or 1.0)
        usage = float(pd.to_numeric(get_raw('Total_Usage_GB'), errors='coerce') or 100.0)
        logins = float(pd.to_numeric(get_raw('monthly_logins'), errors='coerce') or 1.0)
        
        new_row['monthly_fee_log'] = np.log1p(m_charges)
        new_row['value_score_log'] = np.log1p(usage * tenure)
        new_row['last_login_days_log'] = np.log1p(float(pd.to_numeric(get_raw('Last_Login_Days'), errors='coerce') or 5.0))
        new_row['feature_intensity_log'] = np.log1p(float(pd.to_numeric(get_raw('features_used'), errors='coerce') or 2.0))
        new_row['support_intensity_log'] = np.log1p(float(pd.to_numeric(get_raw('support_tickets'), errors='coerce') or 0.0))
        new_row['session_strength_log'] = np.log1p(500.0 / (logins + 1.0))
        
        csat = float(pd.to_numeric(get_raw('csat'), errors='coerce') or 3.0)
        new_row['csat_score'] = csat
        nps = float(pd.to_numeric(get_raw('nps'), errors='coerce') or 7.0)
        new_row['nps_normalized'] = nps / 10.0
        new_row['loyalty_shock_score'] = (1.0 - (nps/10.0)) * (1.0 / (tenure + 1.0))
        new_row['loyalty_resilience'] = tenure * csat
        new_row['is_zombie_user'] = 1.0 if logins < 2.0 else 0.0

        num_feats = ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 
                     'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 
                     'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk',
                     'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 
                     'usage_growth_rate', 'email_open_rate_fixed', 'engagement_efficiency']
        for n in num_feats:
            clean = n.split('_')[0]
            new_row[n] = float(pd.to_numeric(get_raw(clean), errors='coerce') or 0.0)

    # Final Barrier: Merge Categoricals and Enforce Dtypes
    required = PREPROCESSORS[sector].feature_names_in_
    final_data = {}
    
    for col in required:
        if col in CAT_COLS:
            raw_cat = get_raw(col, "unknown")
            final_data[col] = str(raw_cat) if raw_cat is not None else "unknown"
        else:
            val = new_row.get(col, get_raw(col))
            final_data[col] = float(pd.to_numeric(val, errors='coerce') or 0.0)

    # Reconstruct DF from dictionary to ensure clean slate
    final_df = pd.DataFrame([final_data])
    
    # HARD CASTING: categorical as 'object', numerical as 'float64'
    for col in required:
        if col in CAT_COLS:
            final_df[col] = final_df[col].astype(str).replace('nan', 'unknown')
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
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    
    # 1. Total Engineering Isolation
    df_ready = apply_feature_engineering(pd.DataFrame([payload.features]), sector)
    
    # 2. Inference
    processed = PREPROCESSORS[sector].transform(df_ready)
    prob = MODELS[sector].predict_proba(processed)[0, 1]
    
    # 3. Explainability
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