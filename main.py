import numpy as np
import os, math, uuid, joblib, shap, pandas as pd
import sklearn.utils._encode
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Set

# --- HIJACK & SANITIZATION ---
_original_isnan = np.core.umath.isnan if hasattr(np, 'core') and hasattr(np.core.umath, 'isnan') else None
def universal_safe_isnan(x, *args, **kwargs):
    try:
        if _original_isnan is not None:
            return _original_isnan(x, *args, **kwargs)
        return False
    except TypeError:
        return np.zeros(x.shape, dtype=bool) if hasattr(x, 'shape') else False
np.isnan = universal_safe_isnan

def patched_check_unknown(values, known_values, return_mask=False):
    ks = set(known_values)
    m = np.array([v not in ks for v in values], dtype=bool)
    if return_mask: return m
    if np.any(m): pass # Silent handling

sklearn.utils._encode._check_unknown = patched_check_unknown

# --- ENGINE STATE ---
MODELS, PREPROCESSORS = {}, {}
CAT_COLS_BY_SECTOR: Dict[str, Set[str]] = {}

def _extract_cat_cols(preprocessor) -> Set[str]:
    cat_cols = set()
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder": continue
        if "encoder" in type(transformer).__name__.lower():
            if isinstance(cols, (list, np.ndarray)): cat_cols.update(cols)
            elif isinstance(cols, str): cat_cols.add(cols)
    return cat_cols

@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = {'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 'aviation': 'models/hcim_Aviation_v1.joblib'}
    for s, p in paths.items():
        if os.path.exists(p):
            pipe = joblib.load(p)
            pre = pipe.named_steps['preprocessor']
            
            # FORCE RE-INITIALIZATION OF CATEGORIES
            for _, tr, _ in pre.transformers_:
                if hasattr(tr, 'handle_unknown'):
                    tr.handle_unknown = 'ignore' # Ensure it doesn't crash, but we need the categories!
            
            # Scrub serialized NaNs
            for _, tr, _ in pre.transformers_:
                if hasattr(tr, 'categories_'):
                    tr.categories_ = [np.array(["unknown" if (isinstance(c, float) and math.isnan(c)) else c for c in cats], dtype=object) for cats in tr.categories_]
            MODELS[s], PREPROCESSORS[s] = pipe.named_steps['classifier'], pre
            CAT_COLS_BY_SECTOR[s] = _extract_cat_cols(pre)
            
            print(f"SYSTEM: {s.upper()} Core Online.")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
auth_header = APIKeyHeader(name="x-api-key", auto_error=False)

# --- FEATURE ENGINEERING ---
_BOOL_MAP: Dict[str, float] = {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "y": 1.0, "n": 0.0}

def _to_float(val: Any) -> float:
    if val is None: return 0.0
    if isinstance(val, (int, float, np.floating, np.integer)): return float(val)
    s = str(val).strip().lower()
    if s in _BOOL_MAP: return _BOOL_MAP[s]
    try: return float(s)
    except: return 0.0

def apply_feature_engineering(payload: dict, sector: str) -> pd.DataFrame:
    cat_cols = CAT_COLS_BY_SECTOR[sector]
    # Case-insensitive payload mapping to prevent capitalization key drops
    payload_lower = {str(k).lower(): v for k, v in payload.items()}

    def get_raw(key):
        target_variations = [
            str(key).lower(),
            str(key).lower().replace(' ', '_'),
            str(key).lower().replace('_', ' ')
        ]
        for k in target_variations:
            if k in payload_lower:
                return payload_lower[k]
        return None
        
    required = PREPROCESSORS[sector].feature_names_in_
    final_dict = {}
    for col in required:
        val = get_raw(col)
        if col in cat_cols:
            final_dict[col] = "unknown" if val is None or str(val).strip().lower() in ("nan", "none", "") else str(val)
        else:
            if col == "monthly_fee_log":
                raw_charge = get_raw("Monthly_Charges")
                final_dict[col] = float(np.log1p(float(raw_charge) if raw_charge is not None else 30.0))
            else:
                final_dict[col] = _to_float(val)
    df = pd.DataFrame([final_dict])
    for col in df.columns:
        df[col] = df[col].astype(object) if col in cat_cols else df[col].astype(np.float64)
    return df

# --- CORE INFERENCE LOGIC ---
@app.post("/v1/predict")
async def predict(payload: dict, api_key: str = Security(auth_header)):
    if api_key != API_KEY: raise HTTPException(status_code=403, detail="Security Key Mismatch")
    sector = payload.get('sector', '').lower()
    feat_raw = payload.get('features', {})
    
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid sector '{sector}'. Available: {list(MODELS.keys())}")
    
    # 1. Feature Engineering (Dynamically applied)
    df_ready = apply_feature_engineering(feat_raw, sector)

    try:
        # 1. GENERATE TRANSFORMED DATA
        processed_array = PREPROCESSORS[sector].transform(df_ready)
        
        # 2. DYNAMIC NAME EXTRACTION (Version-Resilient)
        # We pull names directly from the active preprocessor instance
        raw_cols = PREPROCESSORS[sector].get_feature_names_out()
        feature_names = [c.split('__')[-1] for c in raw_cols]
        
        # 3. CLASS-1 SHAP TARGETING
        explainer = shap.TreeExplainer(MODELS[sector])
        shap_values = explainer.shap_values(processed_array)
        
        # Handle SHAP output variety (Ensure we grab Class 1 - Churn)
        if isinstance(shap_values, list):
            # We target the probability of Class 1
            contributions = shap_values[1][0] 
        else:
            # If it's a single array, index it for the first row
            contributions = shap_values[0]

        # 4. SYSTEM AUDIT (Check your HF logs for this output!)
        # This dictionary maps EVERY feature to its raw SHAP value
        full_audit = dict(zip(feature_names, contributions))
        print(f"--- NERAL AI SYSTEM AUDIT [{sector.upper()}] ---")
        for k, v in sorted(full_audit.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"DRIVE SIGNAL: {k} -> {v:.4f}")

        # 5. THE TRUE DRIVER MASK (Ignored negative 'Anchor' values)
        drivers = {name: val for name, val in full_audit.items() if val > 0}
        
        if drivers:
            top_driver = max(drivers, key=drivers.get)
        else:
            top_driver = "STABLE BEHAVIORAL PROFILE"

        return {
            "prediction_id": str(uuid.uuid4()),
            "probability": round(float(MODELS[sector].predict_proba(processed_array)[0, 1]), 4),
            "trigger_diagnosis": top_driver.upper().replace('_', ' '),
            "status": "success"
        }
    except Exception as e:
        print(f"SYSTEM CRITICAL FAILURE: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
