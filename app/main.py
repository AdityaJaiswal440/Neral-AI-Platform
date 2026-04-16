# --- STEP 0: NUMPY IMPORT FIRST, NO PATCH ON np.core.umath ---
import numpy as np

# FIX #1: The original patch called np.core.umath.isnan(x) AFTER also replacing
# np.core.umath.isnan with itself → infinite recursion at scipy import time.
# Correct approach: save the original C-level function BEFORE any patching,
# then call ONLY that saved reference inside the wrapper. Never overwrite
# np.core.umath.isnan — scipy/sklearn call it directly and must hit the C impl.

_original_isnan = None
if hasattr(np, 'core') and hasattr(np.core, 'umath') and hasattr(np.core.umath, 'isnan'):
    _original_isnan = np.core.umath.isnan  # save the real C function

def universal_safe_isnan(x):
    """String-safe isnan that delegates to the real C-level implementation."""
    try:
        if isinstance(x, (float, int, np.floating, np.integer, np.complexfloating)):
            if _original_isnan is not None:
                return bool(_original_isnan(x))
            return bool(np.isnan(float(x)))  # fallback for numpy 2.x where np.core is absent
        return False  # strings, None, etc. are never NaN
    except (TypeError, ValueError, AttributeError):
        return False

# Patch ONLY the public-facing np.isnan — do NOT touch np.core.umath.isnan.
# scipy and sklearn import the C-level ufunc directly; overwriting it was the
# cause of the recursion.
np.isnan = universal_safe_isnan

# --- STEP 1: REMAINING IMPORTS (shap/sklearn now import cleanly) ---
import os
import math
import uuid
import joblib
import pandas as pd
import shap
import sklearn.utils._encode
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any

# --- STEP 2: SKLEARN ENCODE PATCH (handles unknown categories gracefully) ---
def patched_check_unknown(values, known_values, return_mask=False):
    """Bypass NumPy 2.0 strict-type issues in OrdinalEncoder._check_unknown."""
    known_set = set(known_values)
    mask = np.array([v not in known_set for v in values], dtype=bool)
    if return_mask:
        return mask
    if np.any(mask):
        raise ValueError("Unknown categories detected")

sklearn.utils._encode._check_unknown = patched_check_unknown

# --- STEP 3: PYDANTIC MODELS (must be defined BEFORE the route that uses them) ---
# FIX #2: PredictPayload was defined AFTER the /v1/predict route in the original
# file, causing a NameError at class-body evaluation time.

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

# --- STEP 4: LIFESPAN & MODEL LOADING ---
MODELS: Dict[str, Any] = {}
EXPLAINERS: Dict[str, Any] = {}
PREPROCESSORS: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib',
        'aviation':  'models/hcim_Aviation_v1.joblib',
    }
    for sector, path in paths.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            pre  = pipe.named_steps['preprocessor']

            # Clean NaN sentinel values left in fitted OrdinalEncoder categories
            for _, tr, _ in pre.transformers_:
                if hasattr(tr, 'categories_'):
                    tr.categories_ = [
                        np.array(
                            ["unknown" if (isinstance(c, float) and math.isnan(c)) else c
                             for c in cats],
                            dtype=object
                        )
                        for cats in tr.categories_
                    ]

            MODELS[sector]       = pipe.named_steps['classifier']
            PREPROCESSORS[sector] = pre
            EXPLAINERS[sector]   = shap.Explainer(MODELS[sector])
            print(f"SYSTEM: Loaded {sector} model.")
        else:
            print(f"SYSTEM: Model not found at {path}, skipping.")

    print("SYSTEM: Neral AI Core online.")
    yield
    MODELS.clear()
    EXPLAINERS.clear()
    PREPROCESSORS.clear()

# --- STEP 5: APP CORE ---
app = FastAPI(title="Neral AI Core", version="5.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY        = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Security Key Mismatch.")

@app.get("/", response_class=HTMLResponse)
def root():
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center;
                     background: #0e1117; color: white; padding-top: 50px;">
            <h1 style="color: #ff4b4b;">Neral AI Core v5.1</h1>
            <p><strong>NumPy:</strong> {np.__version__}</p>
            <p><strong>Models loaded:</strong> {list(MODELS.keys())}</p>
            <p><strong>Status:</strong> <span style="color:#00ff00;">ONLINE</span></p>
        </body>
    </html>
    """

# --- STEP 6: FEATURE ENGINEERING ---
CAT_COLS = {
    'customer_segment', 'tenure_group', 'contract_type', 'signup_channel',
    'payment_method', 'complaint_type', 'city',
    'Class', 'Customer Type', 'Type of Travel',
}

def apply_feature_engineering(payload: dict, sector: str) -> pd.DataFrame:
    def get_raw(key):
        for k in [key, key.replace(' ', '_'), key.lower()]:
            if k in payload:
                return payload[k]
        return None

    required  = PREPROCESSORS[sector].feature_names_in_
    final_dict: Dict[str, Any] = {}

    for col in required:
        val = get_raw(col)

        if col in CAT_COLS:
            final_dict[col] = (
                "unknown"
                if val is None or str(val).strip().lower() in ('nan', 'none', '')
                else str(val)
            )
        else:
            try:
                if col == 'monthly_fee_log':
                    raw_charge = get_raw('Monthly_Charges')
                    val = np.log1p(float(raw_charge) if raw_charge is not None else 30.0)
                final_dict[col] = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                final_dict[col] = 0.0

    df = pd.DataFrame([final_dict])
    for col in df.columns:
        if col in CAT_COLS:
            df[col] = df[col].astype(object)
        else:
            df[col] = df[col].astype(np.float64)
    return df

# --- STEP 7: PREDICT ENDPOINT ---
@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid sector '{sector}'. Available: {list(MODELS.keys())}")

    df_ready = apply_feature_engineering(payload.features, sector)

    try:
        processed     = PREPROCESSORS[sector].transform(df_ready)
        prob          = float(MODELS[sector].predict_proba(processed)[0, 1])

        all_feat      = PREPROCESSORS[sector].get_feature_names_out()
        features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]

        shap_vals  = EXPLAINERS[sector](pd.DataFrame(processed, columns=features_clean))
        top_driver = features_clean[int(np.argmax(np.abs(shap_vals.values[0])))]

        return {
            "prediction_id":    str(uuid.uuid4()),
            "probability":      round(prob, 4),
            "trigger_diagnosis": top_driver,
            "status":           "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {e}")

# --- STEP 8: LOCAL DEV ENTRYPOINT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)