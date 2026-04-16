# --- STEP 0: THE ENGINE HIJACK (MUST BE LINE 1) ---
import numpy as np

# We manually redefine the isnan function to be string-safe
def universal_safe_isnan(x):
    try:
        # Only numeric types can be NaNs
        if isinstance(x, (float, int, np.floating, np.integer, np.complexfloating)):
            return np.core.umath.isnan(x)
        return False # Strings are never NaNs
    except (TypeError, ValueError, AttributeError):
        return False

# Forcing the patch into the core NumPy C-extensions
np.isnan = universal_safe_isnan
if hasattr(np, 'core') and hasattr(np.core, 'umath'):
    np.core.umath.isnan = universal_safe_isnan

# --- STEP 1: REMAINING IMPORTS ---
import os
import joblib
import pandas as pd
import shap
import uuid
import math
import sklearn.utils._encode
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# --- STEP 2: SKLEARN UTILITY HIJACK ---
def patched_check_unknown(values, known_values, return_mask=False):
    """Manually handles category checks to bypass NumPy 2.0 strictness."""
    mask = np.array([v not in known_values for v in values])
    if return_mask: return mask
    if np.any(mask): raise ValueError("Unknown categories detected")

sklearn.utils._encode._check_unknown = patched_check_unknown

# --- STEP 3: LIFESPAN & MODEL LOADING ---
MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 
        'aviation': 'models/hcim_Aviation_v1.joblib'
    }
    for sector, path in paths.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            pre = pipe.named_steps['preprocessor']
            # Surgical memory cleaning of the fitted model
            for _, tr, _ in pre.transformers_:
                if hasattr(tr, 'categories_'):
                    tr.categories_ = [np.array(["unknown" if (isinstance(c, float) and math.isnan(c)) else c for c in cats], dtype=object) for cats in tr.categories_]
            
            MODELS[sector] = pipe.named_steps['classifier']
            PREPROCESSORS[sector] = pre
            EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
            print(f"SYSTEM: Hijacked and Loaded {sector} model.")
    
    print("SYSTEM: Nuclear Hijack Mode 5.0 Online. Build: v5-final.")
    yield
    MODELS.clear()

# --- STEP 4: APP CORE ---
app = FastAPI(title="Neral AI Core", version="5.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Security Key Mismatch.")

@app.get("/", response_class=HTMLResponse)
def root():
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center; background: #0e1117; color: white; padding-top: 50px;">
            <h1 style="color: #ff4b4b;">Neral AI: Nuclear Hijack 5.0</h1>
            <p><strong>Environment NumPy:</strong> {np.__version__}</p>
            <p><strong>Hijack Status:</strong> <span style="color: #00ff00;">ACTIVE</span></p>
        </body>
    </html>
    """

def apply_feature_engineering(payload: dict, sector: str):
    def get_raw(key):
        for k in [key, key.replace(' ', '_'), key.lower()]:
            if k in payload: return payload[k]
        return None

    required = PREPROCESSORS[sector].feature_names_in_
    # Categorical markers for both models
    cat_cols = ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city', 'Class', 'Customer Type', 'Type of Travel']
    
    final_dict = {}
    for col in required:
        val = get_raw(col)
        if col in cat_cols:
            final_dict[col] = "unknown" if val is None or str(val).lower() == 'nan' else str(val)
        else:
            try:
                # Basic engineering fallback
                if col == 'monthly_fee_log': val = np.log1p(float(get_raw('Monthly_Charges') or 30.0))
                # ... add other specific eng keys here if needed, or just cast raw ...
                final_dict[col] = float(val) if val is not None else 0.0
            except: final_dict[col] = 0.0

    df = pd.DataFrame([final_dict])
    for col in df.columns:
        if col in cat_cols: df[col] = df[col].astype(object)
        else: df[col] = df[col].astype(np.float64)
    return df

@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS: raise HTTPException(status_code=400, detail="Invalid Sector")
    
    df_ready = apply_feature_engineering(payload.features, sector)
    
    try:
        processed = PREPROCESSORS[sector].transform(df_ready)
        prob = MODELS[sector].predict_proba(processed)[0, 1]
        
        # Explainability
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
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)