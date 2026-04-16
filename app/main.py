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

# 1. Security & App Config
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
app = FastAPI(title="Neral AI: Unified Churn Core", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")

# 2. Global Registry
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(df: pd.DataFrame, sector: str):
    """Immutable Feature Factory. Zero mixed-types allowed."""
    
    def get_val(key, default=None):
        # Look for exact, underscore, space, or lowercase variant
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in df.columns:
                val = df[k].iloc[0]
                return val if pd.notna(val) else default
        return default

    # Define Schema Maps
    cat_map = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }
    cats = cat_map.get(sector, [])

    # Create local DF to avoid fragmentation warnings
    new_data = {}

    if sector == 'aviation':
        dist = float(get_val('Flight_Distance', 0))
        dep_delay = float(get_val('Departure_Delay_Minutes', 0))
        arr_delay = float(get_val('Arrival_Delay_Minutes', 0))
        
        new_data['distance_log'] = np.log1p(dist)
        new_data['delay_intensity_log'] = np.log1p(dep_delay + arr_delay)
        new_data['is_business_travel'] = 1 if str(get_val('Type of Travel')) == 'Business travel' else 0
        new_data['is_disloyal_customer'] = 1 if str(get_val('Customer Type')) == 'disloyal Customer' else 0
        
        ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for r in ratings: new_data[r] = float(get_val(r, 3.0))
        
        new_data['csat_score'] = sum(new_data[r] for r in ratings) / 4
        new_data['loyalty_shock_score'] = new_data['csat_score'] * new_data['is_disloyal_customer']
        new_data['service_friction_score'] = (5 - new_data['Inflight wifi service']) + (5 - new_data['Online boarding'])
        
        std = ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
               'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']
        for s in std: new_data[s] = float(get_val(s, 3.0))
        for c in cats: new_data[c] = str(get_val(c, 'unknown'))

    elif sector == 'ecommerce':
        # 38-feature engineering block
        m_charges = float(get_val('Monthly_Charges', 30.0))
        tenure = float(get_val('Tenure', 1.0))
        usage = float(get_val('Total_Usage_GB', 100.0))
        logins = float(get_val('monthly_logins', 1.0))
        time = float(get_val('total_monthly_time', 500.0))
        feats = float(get_val('features_used', 2.0))
        
        new_data['monthly_fee_log'] = np.log1p(m_charges)
        new_data['value_score_log'] = np.log1p(usage * tenure)
        new_data['last_login_days_log'] = np.log1p(float(get_val('Last_Login_Days', 5.0)))
        new_data['feature_intensity_log'] = np.log1p(feats)
        new_data['support_intensity_log'] = np.log1p(float(get_val('support_tickets', 0.0)))
        new_data['session_strength_log'] = np.log1p(time / (logins + 1))
        
        new_data['csat_score'] = float(get_val('csat', 3.0))
        nps = float(get_val('nps', 7.0))
        new_data['nps_normalized'] = nps / 10.0
        new_data['loyalty_shock_score'] = (1 - (nps/10.0)) * (1 / (tenure + 1))
        new_data['loyalty_resilience'] = tenure * new_data['csat_score']
        new_data['is_zombie_user'] = 1 if logins < 2 else 0
        new_data['engagement_efficiency'] = time / (feats + 1)
        
        # Mapping remaining 20+ features
        num_cols = ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 
                    'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 
                    'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk',
                    'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 
                    'usage_growth_rate', 'email_open_rate_fixed']
        for n in num_cols:
            clean = n.split('_')[0]
            new_data[n] = float(get_val(clean, 0.0))
        for c in cats: new_data[c] = str(get_val(c, 'unknown'))

    # Final Schema Alignment
    final_df = pd.DataFrame([new_data])
    required = PREPROCESSORS[sector].feature_names_in_
    
    for col in required:
        if col not in final_df.columns:
            final_df[col] = "unknown" if col in cats else 0.0
        
        # FORCE CASTING - This kills the isnan error
        if col in cats:
            final_df[col] = final_df[col].astype(str).replace(['nan', 'None'], 'unknown')
        else:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0).astype(float)
            
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
    
    df_ready = apply_feature_engineering(pd.DataFrame([payload.features]), sector)
    
    # Inference
    processed = PREPROCESSORS[sector].transform(df_ready)
    prob = MODELS[sector].predict_proba(processed)[0, 1]
    
    # Diagnosis (SHAP)
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