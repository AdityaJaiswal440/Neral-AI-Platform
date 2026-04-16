# --- STEP 0: NUMPY IMPORT & SAFE ISNAN PATCH ---
import numpy as np

_original_isnan = None
if hasattr(np, 'core') and hasattr(np.core, 'umath') and hasattr(np.core.umath, 'isnan'):
    _original_isnan = np.core.umath.isnan  # Capture C-level ref before any patch

def universal_safe_isnan(x, *args, **kwargs):
    """String-safe isnan — delegates to the saved C-level function, never recurses."""
    try:
        if _original_isnan is not None:
            return _original_isnan(x, *args, **kwargs)
        return False
    except (TypeError, ValueError, AttributeError):
        return np.zeros(x.shape, dtype=bool) if hasattr(x, 'shape') else False

# Patch ONLY the public alias — never overwrite np.core.umath.isnan
np.isnan = universal_safe_isnan

# --- STEP 1: REMAINING IMPORTS ---
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
from typing import Dict, Any, Set

# --- STEP 2: SKLEARN ENCODE PATCH ---
def patched_check_unknown(values, known_values, return_mask=False):
    known_set = set(known_values)
    mask = np.array([v not in known_set for v in values], dtype=bool)
    if return_mask:
        return mask
    if np.any(mask):
        raise ValueError("Unknown categories detected")

sklearn.utils._encode._check_unknown = patched_check_unknown

# --- STEP 3: PYDANTIC MODELS (must be before routes) ---
class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

# --- STEP 4: GLOBAL STATE ---
MODELS: Dict[str, Any]        = {}
PREPROCESSORS: Dict[str, Any] = {}
# Built dynamically at load time from the actual preprocessor — never hardcoded
CAT_COLS_BY_SECTOR: Dict[str, Set[str]] = {}


def _extract_cat_cols(preprocessor) -> Set[str]:
    """
    Derive the exact set of categorical column names from a fitted
    ColumnTransformer by inspecting which sub-transformers are Encoders.
    This is the ground truth — no manual CAT_COLS list needed.
    """
    cat_cols: Set[str] = set()
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        t_name = type(transformer).__name__.lower()
        if "encoder" in t_name:
            if isinstance(cols, (list, np.ndarray)):
                cat_cols.update(cols)
            elif isinstance(cols, str):
                cat_cols.add(cols)
    return cat_cols


# --- STEP 5: LIFESPAN & MODEL LOADING ---
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

            # Clean NaN sentinels baked into fitted OrdinalEncoder categories
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

            MODELS[sector]             = pipe.named_steps['classifier']
            PREPROCESSORS[sector]      = pre
            CAT_COLS_BY_SECTOR[sector] = _extract_cat_cols(pre)
            print(f"SYSTEM: Loaded {sector} model.")
            print(f"  -> Categorical cols ({len(CAT_COLS_BY_SECTOR[sector])}): "
                  f"{sorted(CAT_COLS_BY_SECTOR[sector])}")
        else:
            print(f"SYSTEM: Model not found at {path}, skipping.")

    print("SYSTEM: Neral AI Core online.")
    yield
    MODELS.clear()
    PREPROCESSORS.clear()
    CAT_COLS_BY_SECTOR.clear()


# --- STEP 6: APP CORE ---
app = FastAPI(title="Neral AI Core", version="5.2", lifespan=lifespan)

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
    model_info = {s: sorted(CAT_COLS_BY_SECTOR.get(s, [])) for s in MODELS}
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center;
                     background: #0e1117; color: white; padding-top: 50px;">
            <h1 style="color: #ff4b4b;">Neral AI Core v5.2</h1>
            <p><strong>NumPy:</strong> {np.__version__}</p>
            <p><strong>Models loaded:</strong> {list(MODELS.keys())}</p>
            <p><strong>Cat cols per sector:</strong> {model_info}</p>
            <p><strong>Status:</strong> <span style="color:#00ff00;">ONLINE</span></p>
        </body>
    </html>
    """


# --- STEP 7: FEATURE ENGINEERING ---

# Coerce common boolean-string representations to numeric 0/1.
# Handles clients sending "Yes"/"No", "True"/"False" for binary flag columns
# (e.g. is_advocate, discount_applied) that the model expects as float64.
_BOOL_MAP: Dict[str, float] = {
    "yes": 1.0, "no": 0.0,
    "true": 1.0, "false": 0.0,
    "y": 1.0, "n": 0.0,
}

def _to_float(val: Any) -> float:
    """Safely coerce any value to float for a numeric model feature."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    s = str(val).strip().lower()
    if s in _BOOL_MAP:
        return _BOOL_MAP[s]         # "Yes" -> 1.0, "No" -> 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0                   # graceful default, never raises


def apply_feature_engineering(payload: dict, sector: str) -> pd.DataFrame:
    cat_cols = CAT_COLS_BY_SECTOR[sector]  # derived from model, not hardcoded

    def get_raw(key):
        # Try multiple key variants to be forgiving about naming conventions
        for k in [key, key.replace(' ', '_'), key.lower(), key.replace('_', ' ')]:
            if k in payload:
                return payload[k]
        return None

    required   = PREPROCESSORS[sector].feature_names_in_
    final_dict: Dict[str, Any] = {}

    for col in required:
        val = get_raw(col)

        if col in cat_cols:
            # Categorical branch: always produce a string; unknown for missing/nan
            final_dict[col] = (
                "unknown"
                if val is None or str(val).strip().lower() in ("nan", "none", "")
                else str(val)
            )
        else:
            # Numeric branch: derived features first, then safe bool-aware cast
            if col == "monthly_fee_log":
                raw_charge = get_raw("Monthly_Charges")
                val = np.log1p(float(raw_charge) if raw_charge is not None else 30.0)
                final_dict[col] = float(val)
            else:
                final_dict[col] = _to_float(val)   # handles "Yes"/"No" safely

    df = pd.DataFrame([final_dict])
    for col in df.columns:
        df[col] = df[col].astype(object) if col in cat_cols else df[col].astype(np.float64)
    return df


# --- STEP 8: PREDICT ENDPOINT ---
@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sector '{sector}'. Available: {list(MODELS.keys())}"
        )

    df_ready = apply_feature_engineering(payload.features, sector)

    try:
        # 1. TRANSFORM & ALIGN
        processed_array = PREPROCESSORS[sector].transform(df_ready)
        
        # 2. FORCE FEATURE NAME SYNC
        raw_names = PREPROCESSORS[sector].get_feature_names_out()
        clean_names = [n.split('__')[-1] for n in raw_names]
        X_test = pd.DataFrame(processed_array, columns=clean_names)
        
        # 3. EXECUTE INFERENCE
        prob = float(MODELS[sector].predict_proba(processed_array)[0, 1])
        
        # 4. DIAGNOSIS (THE DRIVER FIX)
        explainer = shap.TreeExplainer(MODELS[sector])
        shap_values = explainer.shap_values(X_test)
        
        contributions = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        drivers = {name: val for name, val in zip(clean_names, contributions) if val > 0}
        
        if drivers:
            top_driver = max(drivers, key=drivers.get)
        else:
            top_driver = "Stable behavioral profile"

        print(f"DEBUG: Top 3 Drivers for {sector}: {sorted(drivers.items(), key=lambda x: x[1], reverse=True)[:3]}")

        return {
            "prediction_id":     str(uuid.uuid4()),
            "probability":       round(prob, 4),
            "trigger_diagnosis": top_driver.upper(),
            "status":            "success",
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Failure: {e}")


# --- STEP 9: LOCAL DEV ENTRYPOINT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)