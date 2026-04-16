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

# 1. GLOBAL REGISTRY
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

# 2. LIFESPAN & SURGICAL MODEL PATCHING
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
                preprocessor = pipe.named_steps['preprocessor']
                
                # SENIOR AI ENGINEER PATCH: Clean internal categories of baked-in NaNs
                # This fixes the model's internal memory of training data
                for _, transformer, _ in preprocessor.transformers_:
                    if hasattr(transformer, 'categories_'):
                        cleaned_categories = []
                        for cat_array in transformer.categories_:
                            # Convert any float('nan') in the model's memory to "unknown"
                            new_cats = np.array([
                                "unknown" if (isinstance(c, float) and math.isnan(c)) else c
                                for c in cat_array
                            ], dtype=object)
                            cleaned_categories.append(new_cats)
                        transformer.categories_ = cleaned_categories
                
                MODELS[sector] = pipe.named_steps['classifier']
                PREPROCESSORS[sector] = preprocessor
                EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
                print(f"SYSTEM: Loaded and Patched {sector} model.")
            except Exception as e:
                print(f"SYSTEM ERROR: Failed to patch {sector} - {e}")
    
    print("SYSTEM: Unified Core Online. 16GB Memory Engine Engaged.")
    yield
    MODELS.clear()

# 3. APP INITIALIZATION
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Neral AI Core", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET")

# 4. VERIFICATION LANDING PAGE
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <body style="font-family: sans-serif; text-align: center; background: #0e1117; color: white; padding-top: 50px;">
            <h1 style="color: #ff4b4b;">Neral AI: Unified Core Online</h1>
            <p><strong>Build Marker:</strong> <span style="color: #00ff00;">v2-numpy-fix</span></p>
            <p>Surgical Category Patching: <strong>ENABLED</strong></p>
        </body>
    </html>
    """

# 5. INPUT SANITIZATION
def apply_feature_engineering(payload: dict, sector: str):
    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload: return payload[k]
        return default

    cat_cols = {
        'aviation': ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    row = {}
    if sector == 'ecommerce':
        m_charges = float(get_raw('Monthly_Charges', 30.0))
        tenure = float(get_raw('Tenure', 1.0))
        row.update({
            'monthly_fee_log': np.log1p(m_charges),
            'value_score_log': np.log1p(float(get_raw('Total_Usage_GB', 100.0)) * tenure),
            'last_login_days_log': np.log1p(float(get_raw('Last_Login_Days', 5.0))),
            'feature_intensity_log': np.log1p(float(get_raw('features_used', 2.0))),
            'support_intensity_log': np.log1p(float(get_raw('support_tickets', 0.0))),
            'session_strength_log': np.log1p(float(get_raw('total_monthly_time', 500.0)) / (float(get_raw('monthly_logins', 1.0)) + 1.0)),
            'csat_score': float(get_raw('csat', 3.0)),
            'nps_normalized': (float(get_raw('nps', 7.0)) / 10.0),
            'is_zombie_user': 1.0 if float(get_raw('monthly_logins', 1.0)) < 2.0 else 0.0,
            'engagement_efficiency': float(get_raw('total_monthly_time', 500.0)) / (float(get_raw('features_used', 2.0)) + 1.0),
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
            # Intercept "nan" before it hits the DataFrame
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

@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    df_ready = apply_feature_engineering(payload.features, sector)
    try:
        processed = PREPROCESSORS[sector].transform(df_ready)
        prob = MODELS[sector].predict_proba(processed)[0, 1]
        all_feat = PREPROCESSORS[sector].get_feature_names_out()
        features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
        shap_vals = EXPLAINERS[sector](pd.DataFrame(processed, columns=features_clean))
        top_driver = features_clean[np.argmax(np.abs(shap_vals.values[0]))]
        return {"prediction_id": str(uuid.uuid4()), "probability": round(float(prob), 4), "trigger_diagnosis": top_driver, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")