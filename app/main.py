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

# 2. PYDANTIC MODEL — must be defined before predict()
class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

# 3. LIFESPAN & SURGICAL MODEL PATCHING
@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib',
        'aviation':  'models/hcim_Aviation_v1.joblib'
    }
    for sector, path in paths.items():
        if not os.path.exists(path):
            print(f"SYSTEM WARNING: {path} not found. Skipping.")
            continue
        try:
            pipe        = joblib.load(path)
            preprocessor = pipe.named_steps['preprocessor']

            # SURGICAL PATCH: Replace float NaN baked into fitted categories_
            for _, transformer, _ in preprocessor.transformers_:
                if hasattr(transformer, 'categories_'):
                    transformer.categories_ = [
                        np.array([
                            "unknown" if (isinstance(c, float) and math.isnan(c)) else c
                            for c in cat_array
                        ], dtype=object)
                        for cat_array in transformer.categories_
                    ]

            MODELS[sector]        = pipe.named_steps['classifier']
            PREPROCESSORS[sector] = preprocessor
            EXPLAINERS[sector]    = shap.Explainer(MODELS[sector])
            print(f"SYSTEM: Loaded and Patched [{sector}] model.")
        except Exception as e:
            print(f"SYSTEM ERROR: Failed to load {sector} — {e}")

    print("SYSTEM: Unified Core Online. Build v3-definitive.")
    yield
    MODELS.clear()
    EXPLAINERS.clear()
    PREPROCESSORS.clear()

# 4. APP INITIALIZATION
API_KEY        = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Neral AI Core", version="3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Check NERAL_SECRET vs x-api-key")

# 5. LANDING PAGE
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <body style="font-family:sans-serif;text-align:center;background:#0e1117;color:white;padding-top:50px;">
        <h1 style="color:#ff4b4b;">Neral AI: Unified Core Online</h1>
        <p><strong>Build:</strong> <span style="color:#00ff00;">v3-definitive</span></p>
        <p>Surgical Category Patching: <strong>ENABLED</strong></p>
        <p>Endpoints: <code>/v1/predict</code> (POST)</p>
      </body>
    </html>
    """

# 6. FEATURE ENGINEERING ENGINE
def apply_feature_engineering(payload: dict, sector: str):

    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.replace('_', ' '), key.lower()]:
            if k in payload:
                val = payload[k]
                return val if val is not None and str(val).lower() != 'nan' else default
        return default

    cat_cols = {
        'aviation':  ['Class', 'Customer Type', 'Type of Travel'],
        'ecommerce': ['customer_segment', 'tenure_group', 'contract_type',
                      'signup_channel', 'payment_method', 'complaint_type', 'city']
    }.get(sector, [])

    row = {}

    if sector == 'aviation':
        dist  = float(pd.to_numeric(get_raw('Flight_Distance'), errors='coerce') or 0)
        dep_d = float(pd.to_numeric(get_raw('Departure_Delay_Minutes'), errors='coerce') or 0)
        arr_d = float(pd.to_numeric(get_raw('Arrival_Delay_Minutes'), errors='coerce') or 0)
        row.update({
            'distance_log':           np.log1p(dist),
            'delay_intensity_log':    np.log1p(dep_d + arr_d),
            'is_business_travel':     1.0 if str(get_raw('Type of Travel')) == 'Business travel' else 0.0,
            'is_disloyal_customer':   1.0 if str(get_raw('Customer Type')) == 'disloyal Customer' else 0.0,
        })
        ratings = ['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment']
        for r in ratings:
            row[r] = float(get_raw(r, 3.0))
        row['csat_score']             = sum(row[r] for r in ratings) / 4.0
        row['loyalty_shock_score']    = row['csat_score'] * row['is_disloyal_customer']
        row['service_friction_score'] = 10.0 - row['Inflight wifi service'] - row['Online boarding']
        for s in ['Inflight service', 'Food and drink', 'Baggage handling', 'Checkin service',
                  'On-board service', 'Cleanliness', 'Gate location', 'Leg room service',
                  'Ease of Online booking', 'Departure/Arrival time convenient']:
            row[s] = float(get_raw(s, 3.0))

    elif sector == 'ecommerce':
        m_charges = float(get_raw('Monthly_Charges', 30.0))
        tenure    = float(get_raw('Tenure', 1.0))
        usage     = float(get_raw('Total_Usage_GB', 100.0))
        logins    = float(get_raw('monthly_logins', 1.0))
        time_val  = float(get_raw('total_monthly_time', 500.0))
        feats     = float(get_raw('features_used', 2.0))
        nps       = float(get_raw('nps', 7.0))
        csat      = float(get_raw('csat', 3.0))

        row.update({
            'monthly_fee_log':        np.log1p(m_charges),
            'value_score_log':        np.log1p(usage * tenure),
            'last_login_days_log':    np.log1p(float(get_raw('Last_Login_Days', 5.0))),
            'feature_intensity_log':  np.log1p(feats),
            'support_intensity_log':  np.log1p(float(get_raw('support_tickets', 0.0))),
            'session_strength_log':   np.log1p(time_val / (logins + 1.0)),
            'csat_score':             csat,
            'nps_normalized':         nps / 10.0,
            'is_zombie_user':         1.0 if logins < 2.0 else 0.0,
            'engagement_efficiency':  time_val / (feats + 1.0),
            'loyalty_resilience':     tenure * csat,
        })
        row['loyalty_shock_score'] = (1.0 - row['nps_normalized']) * (1.0 / (tenure + 1.0))

        # Use full key name — split('_')[0] was a bug that corrupted is_* features
        for n in ['usage_density', 'discount_applied', 'monthly_logins', 'features_used',
                  'total_monthly_time', 'weekly_active_days', 'is_passive_promoter',
                  'is_advocate', 'is_recency_danger', 'is_high_friction_payment',
                  'is_bouncer', 'is_hidden_dissatisfaction', 'payment_structural_risk',
                  'support_tickets_clipped', 'escalations_clipped', 'payment_failures_clipped',
                  'referral_count_clipped', 'usage_growth_rate', 'email_open_rate_fixed',
                  'is_slow_ghost']:
            row[n] = float(get_raw(n, 0.0))

    # FINAL ALIGNMENT: build DataFrame that exactly matches preprocessor's expected schema
    required   = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}

    for col in required:
        if col in cat_cols:
            val     = get_raw(col, "unknown")
            str_val = str(val).strip()
            final_dict[col] = "unknown" if str_val.lower() in ("nan", "none", "") else str_val
        else:
            val = row.get(col, get_raw(col, 0.0))
            try:
                f = float(val)
                final_dict[col] = 0.0 if (math.isnan(f) or math.isinf(f)) else f
            except (TypeError, ValueError):
                final_dict[col] = 0.0

    final_df = pd.DataFrame([final_dict])

    for col in required:
        if col in cat_cols:
            final_df[col] = final_df[col].astype(object)
        else:
            final_df[col] = final_df[col].astype(np.float64)

    return final_df

# 7. INFERENCE ENDPOINT
@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid sector '{sector}'. Valid: {list(MODELS.keys())}")

    df_ready = apply_feature_engineering(payload.features, sector)

    try:
        processed      = PREPROCESSORS[sector].transform(df_ready)
        prob           = MODELS[sector].predict_proba(processed)[0, 1]
        all_feat       = PREPROCESSORS[sector].get_feature_names_out()
        features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
        shap_vals      = EXPLAINERS[sector](pd.DataFrame(processed, columns=features_clean))
        top_driver     = features_clean[np.argmax(np.abs(shap_vals.values[0]))]

        return {
            "prediction_id":    str(uuid.uuid4()),
            "probability":      round(float(prob), 4),
            "trigger_diagnosis": top_driver,
            "status":           "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)