# LINE 1: THE HIJACK (Must be the first thing the CPU sees)
import numpy as np
import math

def nuclear_isnan_patch(x):
    try:
        # Only run the real math function if the input is actually a number
        if isinstance(x, (int, float, np.number)):
            return np.core.umath.isnan(x)
        return False # If it's a string, it's NOT a NaN
    except:
        return False

# Forcing the patch into the core NumPy C-engine
np.isnan = nuclear_isnan_patch
if hasattr(np, 'core') and hasattr(np.core, 'umath'):
    np.core.umath.isnan = nuclear_isnan_patch

import os
import joblib
import pandas as pd
import shap
import uuid
import sklearn.utils._encode
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# HIJACK SKLEARN'S INTERNAL CATEGORY CHECK
def patched_check_unknown(values, known_values, return_mask=False):
    """Bypasses the NumPy 2.0 isnan crash in OneHotEncoder"""
    mask = np.array([v not in known_values for v in values])
    if return_mask: return mask
    if np.any(mask): raise ValueError("Unknown categories detected")

sklearn.utils._encode._check_unknown = patched_check_unknown

MODELS, EXPLAINERS, PREPROCESSORS = {}, {}, {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Models and Sanitize Memory
    paths = {'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 'aviation': 'models/hcim_Aviation_v1.joblib'}
    for sector, path in paths.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            pre = pipe.named_steps['preprocessor']
            # Surgical removal of baked-in NaNs from model memory
            for _, tr, _ in pre.transformers_:
                if hasattr(tr, 'categories_'):
                    tr.categories_ = [np.array(["unknown" if (isinstance(c, float) and math.isnan(c)) else c for c in cats], dtype=object) for cats in tr.categories_]
            
            MODELS[sector] = pipe.named_steps['classifier']
            PREPROCESSORS[sector] = pre
            EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
    print("SYSTEM: Nuclear Hijack Mode 4.0 Online.")
    yield
    MODELS.clear()

# CRITICAL: The app object must be globally accessible for Uvicorn
app = FastAPI(title="Neral AI Core", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY: return api_key
    raise HTTPException(status_code=403, detail="Check x-api-key header.")

@app.get("/", response_class=HTMLResponse)
def root():
    return f"<html><body style='background:#0e1117;color:white;text-align:center;'><h1>Neral AI v4.0</h1><p>NumPy: {np.__version__}</p><p>Status: Nuclear Hijack Active</p></body></html>"

def apply_feature_engineering(payload: dict, sector: str):
    def get_raw(key, default=None):
        for k in [key, key.replace(' ', '_'), key.lower()]:
            if k in payload: return payload[k]
        return default

    required = PREPROCESSORS[sector].feature_names_in_
    cat_cols = ['customer_segment', 'tenure_group', 'contract_type', 'signup_channel', 'payment_method', 'complaint_type', 'city', 'Class', 'Customer Type', 'Type of Travel']
    
    final_dict = {}
    for col in required:
        val = get_raw(col)
        if col in cat_cols:
            final_dict[col] = "unknown" if val is None or str(val).lower() == 'nan' else str(val)
        else:
            try: final_dict[col] = float(val) if val is not None else 0.0
            except: final_dict[col] = 0.0

    df = pd.DataFrame([final_dict])
    for col in df.columns:
        if col in cat_cols: df[col] = df[col].astype(object)
        else: df[col] = df[col].astype(np.float64)
    return df

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    df_ready = apply_feature_engineering(payload.features, sector)
    try:
        processed = PREPROCESSORS[sector].transform(df_ready)
        prob = MODELS[sector].predict_proba(processed)[0, 1]
        return {"probability": round(float(prob), 4), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))