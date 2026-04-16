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
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Neral AI: Absolute Core", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET vs x-api-key")

# 2. Global Registry
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

def apply_feature_engineering(payload_dict: dict, sector: str):
    """Rebuilds data from a dictionary to guarantee type purity."""
    
    def get_raw(key, default=None):
        # Universal lookup: check key, underscore, space, and lowercase
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload_dict:
                val = payload_dict[k]
                return val if val is not None and str(val).lower() != 'nan' else default
        return default

    # Define strict categorical and numerical maps
    sector_cats = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    # The blueprint for the new row
    row = {}

    if sector == 'aviation':
        dist = float(get_raw('Flight_Distance', 0))
        dep_d = float(get_raw('Departure_Delay_Minutes', 0))
        arr_d = float(get_raw('Arrival_Delay_Minutes', 0))
        row.update({
            'distance_log': np.log1p(dist),
            'delay_intensity_log': np.log1p(dep_d + arr_d),
            'is_business_travel': 1.0 if str(get_raw('Type of Travel')) == 'Business travel' else 0.0,
            'is_disloyal_customer': 1.0 if str(get_raw('Customer Type')) == 'disloyal Customer' else 0.0
        })
        ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for r in ratings: row[r] = float(get_raw(r, 3.0))
        row['csat_score'] = sum(row[r] for r in ratings) / 4.0
        row['loyalty_shock_score'] = row['csat_score'] * row['is_disloyal_customer']
        row['service_friction_score'] = (5.0 - row['Inflight wifi service']) + (5.0 - row['Online boarding'])
        for s in ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 
                  'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']:
            row[s] = float(get_raw(s, 3.0))

    elif sector == 'ecommerce':
        m_charges = float(get_raw('Monthly_Charges', 30.0))
        tenure = float(get_raw('Tenure', 1.0))
        usage = float(get_raw('Total_Usage_GB', 100.0))
        logins = float(get_raw('monthly_logins', 1.0))
        time = float(get_raw('total_monthly_time', 500.0))
        feats = float(get_raw('features_used', 2.0))
        
        row.update({
            'monthly_fee_log': np.log1p(m_charges),
            'value_score_log': np.log1p(usage * tenure),
            'last_login_days_log': np.log1p(float(get_raw('Last_Login_Days', 5.0))),
            'feature_intensity_log': np.log1p(feats),
            'support_intensity_log': np.log1p(float(get_raw('support_tickets', 0.0))),
            'session_strength_log': np.log1p(time / (logins + 1.0)),
            'csat_score': float(get_raw('csat', 3.0)),
            'nps_normalized': (float(get_raw('nps', 7.0)) / 10.0),
            'is_zombie_user': 1.0 if logins < 2.0 else 0.0,
            'engagement_efficiency': time / (feats + 1.0)
        })
        row['loyalty_shock_score'] = (1.0 - row['nps_normalized']) * (1.0 / (tenure + 1.0))
        row['loyalty_resilience'] = tenure * row['csat_score']

        for n in ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 
                  'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 
                  'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk',
                  'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 
                  'usage_growth_rate', 'email_open_rate_fixed']:
            clean = n.split('_')[0]
            row[n] = float(get_raw(clean, 0.0))

    # Final Alignment with Preprocessor
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        if col in sector_cats:
            val = get_raw(col, "unknown")
            final_dict[col] = str(val).strip()
        else:
            final_dict[col] = float(row.get(col, get_raw(col, 0.0)))

    # Reconstruct DF from dictionary with absolute type segregation
    final_df = pd.DataFrame([final_dict])
    for col in required:
        if col in sector_cats:
            # Force NumPy to treat this as a pure string object
            final_df[col] = final_df[col].astype(str)
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
@app.post("/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    
    # 1. Total Reconstruction
    df_ready = apply_feature_engineering(payload.features, sector)
    
    # 2. Inference & Diagnosis
    try:
        processed = PREPROCESSORS[sector].transform(df_ready)
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
    except Exception as e:
        print(f"CRITICAL TRANSFORM ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)