import os
import joblib
import pandas as pd
import numpy as np
import shap
import uuid
import math
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# 1. LIFESPAN MANAGEMENT (Production Standard)
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

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
    MODELS.clear()

# 2. APP INITIALIZATION
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(
    title="Neral AI: Unified Churn Core", 
    version="1.3", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET")

# 3. ROOT LANDING (Silences 404 Logs)
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>Neral AI Core</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px; background-color: #0e1117; color: white;">
            <h1 style="color: #ff4b4b;">Neral AI: Unified Core Online</h1>
            <p>Surgical Dictionary Sanitization Engaged | Version 1.3</p>
            <div style="padding: 20px; border: 1px solid #333; display: inline-block; border-radius: 10px;">
                <strong>Status:</strong> <span style="color: #00ff00;">Active</span><br>
                <strong>Endpoints:</strong> <code>/v1/predict</code> (POST)
            </div>
        </body>
    </html>
    """

# 4. SURGICAL FEATURE ENGINEERING (Senior AI Engineer Logic)
def apply_feature_engineering(payload: dict, sector: str):
    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload:
                val = payload[k]
                return val if val is not None else default
        return default

    cat_cols = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    # Internal engineering dictionary
    row = {}
    if sector == 'ecommerce':
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

    # FINAL ALIGNMENT & HARD-CASTING (Senior AI Engineer's Master Block)
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        if col in cat_cols:
            val = get_raw(col, "unknown")
            str_val = str(val).strip()
            # Kill float NaN, None, and "nan" at dict-build time
            final_dict[col] = "unknown" if str_val.lower() in ("nan", "none", "") else str_val
        else:
            val = row.get(col, get_raw(col, 0.0))
            try:
                f = float(val)
                # math.isnan avoids NumPy's ufunc crash on strings
                final_dict[col] = 0.0 if math.isnan(f) or math.isinf(f) else f
            except (TypeError, ValueError):
                final_dict[col] = 0.0

    final_df = pd.DataFrame([final_dict])

    # CAST TO EXACT FITTED DTYPES
    for col in required:
        if col in cat_cols:
            final_df[col] = final_df[col].astype(object) # Revert to object to match .fit()
        else:
            final_df[col] = final_df[col].astype(np.float64)
            
    return final_df

# 5. INFERENCE ENDPOINT
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
        print(f"TRANSFORM ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)