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

# ==========================================================
# 1. THE NUCLEAR MONKEY PATCH (Execute First)
# ==========================================================
original_isnan = np.isnan
def patched_isnan(x):
    try:
        return original_isnan(x)
    except (TypeError, ValueError):
        return False # NumPy 1.x behavior: Strings are not NaNs
np.isnan = patched_isnan

# ==========================================================
# 2. LIFESPAN & MODEL LOADING
# ==========================================================
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    paths = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 
        'aviation': 'models/hcim_Aviation_v1.joblib'
    }
    for sector, path in paths.items():
        if os.path.exists(path):
            try:
                pipe = joblib.load(path)
                MODELS[sector] = pipe.named_steps['classifier']
                PREPROCESSORS[sector] = pipe.named_steps['preprocessor']
                EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
                print(f"SYSTEM: Loaded {sector} logic into memory.")
            except Exception as e:
                print(f"SYSTEM ERROR: Failed to load {sector} - {e}")
    print("SYSTEM: Unified Core Online. 16GB Memory Engine Engaged.")
    yield
    # Shutdown logic
    MODELS.clear()
    print("SYSTEM: Memory Cleared.")

# ==========================================================
# 3. GLOBAL APP DEFINITION (CRITICAL: Must be named 'app')
# ==========================================================
app = FastAPI(
    title="Neral AI: Unified Churn Core", 
    version="1.5", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 4. SECURITY & ROUTES
# ==========================================================
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET header.")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <body style="font-family: sans-serif; text-align: center; background: #0e1117; color: white; padding-top: 50px;">
            <h1 style="color: #ff4b4b;">Neral AI: Unified Core Online</h1>
            <p>Monkey-Patch 1.5 (NumPy 2.0 Resilience) | Status: <span style="color:#00ff00;">Active</span></p>
        </body>
    </html>
    """

def apply_feature_engineering(payload: dict, sector: str):
    def get_raw(key, default=None):
        # Checks for multiple variations of keys (logins vs monthly_logins)
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload: return payload[k]
        return default

    cat_cols = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    row = {}
    if sector == 'ecommerce':
        # Pure numeric sanitization
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
            'engagement_efficiency': time / (feats + 1.0),
            'loyalty_resilience': tenure * float(get_raw('csat', 3.0))
        })
        row['loyalty_shock_score'] = (1.0 - row['nps_normalized']) * (1.0 / (tenure + 1.0))
        for n in ['usage_density', 'discount_applied', 'monthly_logins', 'features_used', 'total_monthly_time', 'weekly_active_days', 'is_passive_promoter', 'is_advocate', 'is_recency_danger', 'is_high_friction_payment', 'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk', 'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped', 'referral_count_clipped', 'usage_growth_rate', 'email_open_rate_fixed']:
            row[n] = float(get_raw(n.split('_')[0], 0.0))

    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        if col in cat_cols:
            val = get_raw(col, "unknown")
            str_val = str(val).strip()
            final_dict[col] = "unknown" if str_val.lower() in ("nan", "none", "") else str_val
        else:
            val = row.get(col, get_raw(col, 0.0))
            try:
                f = float(val)
                final_dict[col] = 0.0 if math.isnan(f) or math.isinf(f) else f
            except: final_dict[col] = 0.0

    final_df = pd.DataFrame([final_dict])
    for col in required:
        if col in cat_cols: final_df[col] = final_df[col].astype(object)
        else: final_df[col] = final_df[col].astype(np.float64)
    return final_df

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    
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
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This block is only for local testing, HF uses its own internal call.
    uvicorn.run(app, host="0.0.0.0", port=8000)