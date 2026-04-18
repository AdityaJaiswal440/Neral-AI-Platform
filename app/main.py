import os
import subprocess
import sys
import math
import uuid
import joblib
import pandas as pd
import numpy as np
import shap
import sklearn.utils._encode
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Set

# --- 1. DYNAMIC ENVIRONMENT GUARD (Windows Safe) ---
if os.environ.get("UPGRADED") != "TRUE":
    print("SYSTEM: Synchronizing environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.7.2"])
    os.environ["UPGRADED"] = "TRUE"
    # Restart using the module syntax to avoid path errors
    os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

# --- 2. SKLEARN PATCH ---
def patched_check_unknown(values, known_values, return_mask=False):
    known_set = set(known_values)
    mask = np.array([v not in known_set for v in values], dtype=bool)
    if return_mask: return mask
    if np.any(mask): raise ValueError("Unknown categories detected")
sklearn.utils._encode._check_unknown = patched_check_unknown

# --- 3. PYDANTIC MODELS ---
class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

# --- 4. GLOBAL STATE ---
MODELS = {}
PREPROCESSORS = {}
CAT_COLS_BY_SECTOR = {}

def _extract_cat_cols(preprocessor) -> Set[str]:
    cat_cols: Set[str] = set()
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder": continue
        if "encoder" in type(transformer).__name__.lower():
            if isinstance(cols, (list, np.ndarray)): cat_cols.update(cols)
            elif isinstance(cols, str): cat_cols.add(cols)
    return cat_cols

# --- 5. LIFESPAN (Model Loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Paths are relative to the ROOT directory where you run uvicorn
    paths = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib',
        'aviation':  'models/hcim_Aviation_v1.joblib',
    }
    for sector, path in paths.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            MODELS[sector] = pipe.named_steps['classifier']
            pre = pipe.named_steps['preprocessor']
            PREPROCESSORS[sector] = pre
            CAT_COLS_BY_SECTOR[sector] = _extract_cat_cols(pre)
            print(f"SYSTEM: Loaded {sector} model.")
        else:
            print(f"CRITICAL: Model not found at {path}")
    yield
    MODELS.clear()
    PREPROCESSORS.clear()

# --- 6. APP CORE ---
app = FastAPI(title="Neral AI Core", version="6.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY: return header_key
    raise HTTPException(status_code=403, detail="Security Key Mismatch.")

@app.get("/")
def root():
    return {"status": "ONLINE", "sectors": list(MODELS.keys()), "engine": "v6.1-Atomic"}

# --- 7. FEATURE ENGINEERING ---
def _to_float(val: Any) -> float:
    try:
        if val is None: return 0.0
        s = str(val).strip().lower()
        if s in ["yes", "true", "1"]: return 1.0
        if s in ["no", "false", "0"]: return 0.0
        return float(s)
    except: return 0.0

def apply_feature_engineering(payload: dict, sector: str) -> pd.DataFrame:
    cat_cols = CAT_COLS_BY_SECTOR[sector]
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    
    # Case-insensitive lookup
    p_low = {str(k).lower(): v for k, v in payload.items()}
    
    for col in required:
        val = p_low.get(col.lower())
        if col in cat_cols:
            final_dict[col] = "unknown" if val in [None, "nan", ""] else str(val)
        else:
            final_dict[col] = _to_float(val)
            
    return pd.DataFrame([final_dict])[list(required)]

# --- 8. PREDICT ENDPOINT ---
@app.post("/v1/predict")
async def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid Sector")

    # 1. Align Features
    df_final = apply_feature_engineering(payload.features, sector)
    
    # 2. Transform & Explain
    processed = PREPROCESSORS[sector].transform(df_final)
    if hasattr(processed, "toarray"): processed = processed.toarray()
    
    feature_names = [c.split('__')[-1] for c in PREPROCESSORS[sector].get_feature_names_out()]
    explainer = shap.TreeExplainer(MODELS[sector])
    shap_vals = explainer.shap_values(processed)
    
    # SHAP logic for binary classification
    contributions = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    
    # 3. Audit Log
    print(f"\n--- NERAL AI GROUND TRUTH [v6.1] ---")
    driver_map = []
    for i, (name, val) in enumerate(zip(feature_names, contributions)):
        driver_map.append({"name": name, "shap": val})
        if i < 5: print(f"IDX {i} | {name[:15]:<15} | SHAP: {val:.4f}")

    top_driver = max(driver_map, key=lambda x: x['shap'])['name']

    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": round(float(MODELS[sector].predict_proba(processed)[0, 1]), 4),
        "trigger_diagnosis": top_driver.upper().replace('_', ' ')
    }