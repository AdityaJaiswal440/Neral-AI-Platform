import os
import joblib
import pandas as pd
import numpy as np
import shap
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# 1. SECURITY & GLOBAL REGISTRY
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

# 2. LIFESPAN MANAGEMENT (The Modern FastAPI Standard)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Models into 16GB Engine
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
    yield
    # Shutdown logic (if any) goes here
    MODELS.clear()

app = FastAPI(
    title="Neral AI: Unified Churn Core", 
    version="1.1", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. AUTHENTICATION GATE
def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET vs x-api-key")

# 4. LANDING PAGE
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>Neral AI Core</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px; background-color: #0e1117; color: white;">
            <h1 style="color: #ff4b4b;">Neral AI: Unified Core Online</h1>
            <p>Poison-Proofed Architecture Engaged | Timezone: UTC</p>
            <div style="padding: 20px; border: 1px solid #333; display: inline-block; border-radius: 10px;">
                <strong>Status:</strong> <span style="color: #00ff00;">Active</span><br>
                <strong>Endpoints:</strong> <code>/v1/predict</code> (POST)
            </div>
        </body>
    </html>
    """

# 5. POISON-PROOFED FEATURE ENGINEERING
def apply_feature_engineering(payload: dict, sector: str):
    """Senior AI Engineer Fix: Implements StringDtype to bypass NumPy isnan() check."""
    
    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload:
                val = payload[k]
                return val if val is not None and str(val).lower() != 'nan' else default
        return default

    cat_cols = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    row = {}
    if sector == 'aviation':
        dist = float(pd.to_numeric(get_raw('Flight_Distance'), errors='coerce') or 0)
        dep_d = float(pd.to_numeric(get_raw('Departure_Delay_Minutes'), errors='coerce') or 0)
        arr_d = float(pd.to_numeric(get_raw('Arrival_Delay_Minutes'), errors='coerce') or 0)
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
        row['service_friction_score'] = (10.0 - row['Inflight wifi service'] - row['Online boarding'])
        for s in ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service', 'On-board service', 'Cleanliness', 'Gate location', 'Leg room service', 'Ease of Online booking', 'Departure/Arrival time convenient']:
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
        for n in ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk', 'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 'usage_growth_rate', 'email_open_rate_fixed']:
            row[n] = float(get_raw(n.split('_')[0], 0.0))

    # FINAL ALIGNMENT & HARD-CASTING
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        if col in cat_cols:
            val = get_raw(col, "unknown")
            cleaned = str(val).strip()
            final_dict[col] = "unknown" if cleaned.lower() == "nan" else cleaned
        else:
            val = row.get(col, get_raw(col, 0.0))
            try:
                final_dict[col] = float(val)
            except (TypeError, ValueError):
                final_dict[col] = 0.0

    final_df = pd.DataFrame([final_dict])

    for col in required:
        if col in cat_cols:
            # Using StringDtype() to bypass scikit-learn's isnan() numeric check
            final_df[col] = final_df[col].astype(str).fillna("unknown").astype(pd.StringDtype())
        else:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0).astype(np.float64)
            
    return final_df

# 6. INFERENCE ENDPOINT
class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid Sector")
    
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
        print(f"TRANSFORM ERROR ON COLUMNS: {df_ready.dtypes}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)