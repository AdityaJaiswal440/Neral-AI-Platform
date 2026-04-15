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
    """Surgical feature replication with defensive schema alignment."""
    
    def f_val(key, default=0):
        # Checks for underscore, space, and lowercase variants
        val = df.get(key, df.get(key.replace('_', ' '), df.get(key.lower(), None)))
        if isinstance(val, pd.Series):
            return val.iloc[0] if not val.empty else default
        return val if val is not None else default

    if sector == 'aviation':
        # Aviation Logic (Verified)
        df['distance_log'] = np.log1p(float(f_val('Flight_Distance', 0)))
        df['delay_intensity_log'] = np.log1p(float(f_val('Departure_Delay_Minutes', 0)) + float(f_val('Arrival_Delay_Minutes', 0)))
        df['is_business_travel'] = 1 if str(f_val('Type_of_Travel')) == 'Business travel' else 0
        df['is_disloyal_customer'] = 1 if str(f_val('Customer_Type')) == 'disloyal Customer' else 0
        
        core_ratings = {'Inflight_wifi_service': 'Inflight wifi service', 'Online_boarding': 'Online boarding', 
                        'Seat_comfort': 'Seat comfort', 'Inflight_entertainment': 'Inflight entertainment'}
        for j_key, m_key in core_ratings.items(): df[m_key] = f_val(j_key)

        df['csat_score'] = df[list(core_ratings.values())].mean(axis=1)
        df['loyalty_shock_score'] = df['csat_score'] * df['is_disloyal_customer']
        df['service_friction_score'] = (5 - df['Inflight wifi service']) + (5 - df['Online boarding'])
        
        standard_mapping = {'Inflight_service': 'Inflight service', 'Food_and_drink': 'Food and drink', 
                            'Baggage_handling': 'Baggage handling', 'Checkin_service': 'Checkin service', 
                            'On_board_service': 'On-board service', 'Cleanliness': 'Cleanliness', 
                            'Gate_location': 'Gate location', 'Leg_room_service': 'Leg_room_service', 
                            'Ease_of_Online_booking': 'Ease of Online booking', 'Departure/Arrival_time_convenient': 'Departure/Arrival time convenient'}
        for j_key, m_key in standard_mapping.items(): df[m_key] = f_val(j_key)

    elif sector == 'ecommerce':
        # E-commerce Logic (Resolved 37/38 features)
        df['monthly_fee_log'] = np.log1p(float(f_val('Monthly_Charges', 30)))
        df['value_score_log'] = np.log1p(float(f_val('Total_Usage_GB', 100)) * float(f_val('Tenure', 1)))
        df['last_login_days_log'] = np.log1p(float(f_val('Last_Login_Days', 5)))
        df['feature_intensity_log'] = np.log1p(float(f_val('features_used', 2)))
        df['support_intensity_log'] = np.log1p(float(f_val('support_tickets', 0)))
        df['session_strength_log'] = np.log1p(float(f_val('total_monthly_time', 500)) / (float(f_val('monthly_logins', 1)) + 1))

        df['support_tickets_clipped'] = np.clip(float(f_val('support_tickets', 0)), 0, 10)
        df['escalations_clipped'] = np.clip(float(f_val('escalations', 0)), 0, 5)
        df['payment_failures_clipped'] = np.clip(float(f_val('payment_failures', 0)), 0, 3)
        df['referral_count_clipped'] = np.clip(float(f_val('referral_count', 0)), 0, 20)

        df['is_zombie_user'] = 1 if float(f_val('monthly_logins', 1)) < 2 else 0
        df['is_slow_ghost'] = 1 if float(f_val('total_monthly_time', 0)) < 100 and float(f_val('Tenure', 0)) > 6 else 0
        df['is_passive_promoter'] = 1 if float(f_val('nps', 7)) in [7, 8] else 0
        df['is_advocate'] = 1 if float(f_val('nps', 7)) > 8 else 0
        df['is_recency_danger'] = 1 if float(f_val('Last_Login_Days', 0)) > 15 else 0
        df['is_high_friction_payment'] = 1 if str(f_val('payment_method')) == 'Mailed Check' else 0
        df['is_bouncer'] = 1 if float(f_val('email_open_rate', 0)) < 0.05 else 0
        df['is_hidden_dissatisfaction'] = 1 if float(f_val('csat', 3)) < 3 and float(f_val('support_tickets', 0)) == 0 else 0

        df['csat_score'] = float(f_val('csat', 3))
        df['nps_normalized'] = float(f_val('nps', 7)) / 10.0
        df['loyalty_shock_score'] = (1 - df['nps_normalized']) * (1 / (float(f_val('Tenure', 1)) + 1))
        df['engagement_efficiency'] = float(f_val('total_monthly_time', 0)) / (float(f_val('features_used', 1)) + 1)
        df['usage_growth_rate'] = float(f_val('usage_delta', 0)) 
        df['loyalty_resilience'] = float(f_val('Tenure', 1)) * df['csat_score']
        df['email_open_rate_fixed'] = float(f_val('email_open_rate', 0.2))
        df['payment_structural_risk'] = 1 if float(f_val('payment_failures', 0)) > 1 else 0

        # Categoricals & Raw columns
        for col in ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city',
                    'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 'weekly_active_days', 'usage_density']:
            df[col] = f_val(col)
            
    # DYNAMIC ALIGNMENT: The "Nuclear Option"
    # Scikit-learn preprocessors know what they want. We ask them.
    try:
        required_cols = PREPROCESSORS[sector].feature_names_in_
        for col in required_cols:
            if col not in df.columns:
                df[col] = f_val(col, 0)
    except Exception as e:
        print(f"DEBUG: Schema alignment bypass - {str(e)}")
                
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
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid sector")
    
    df_raw = pd.DataFrame([payload.features])
    df_ready = apply_feature_engineering(df_raw, sector)
    
    # DEBUG HOOK: This prints exactly what the model sees to the logs
    print(f"DEBUG: Processing {sector}. Columns present: {df_ready.columns.tolist()}")
    
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