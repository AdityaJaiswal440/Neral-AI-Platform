"""
app/main.py  —  SPOTLIGHT SKELETON
Neral AI Platform v6.1 · Public Reference Release

REDACTED:
  - apply_feature_engineering() internals (Behavioral Foundry)
  - _extract_cat_cols() internals (categorical namespace derivation)
  - Model artifact paths (sector → .joblib mapping)
  - sklearn _check_unknown patch implementation

PUBLISHED:
  - FastAPI lifespan model-loading pattern
  - API key authentication architecture (Security dependency injection)
  - /v1/predict endpoint contract and SHAP Atomic Alignment interface
  - Distributed process startup guard (UPGRADED env flag)
"""

import os
import subprocess
import sys
import uuid
import joblib
import pandas as pd
import numpy as np
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Set


# ─────────────────────────────────────────────────────────────────────────────
# 1. DYNAMIC ENVIRONMENT GUARD
#    Forces scikit-learn to the exact version used during training.
#    Required because HF Spaces base image may carry a different sklearn build.
#    The UPGRADED flag prevents recursive re-execution after os.execv().
# ─────────────────────────────────────────────────────────────────────────────
if os.environ.get("UPGRADED") != "TRUE":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.7.2"])
    os.environ["UPGRADED"] = "TRUE"
    os.execv(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", "8000"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. SKLEARN RUNTIME PATCH
#    Patches sklearn.utils._encode._check_unknown to handle categories
#    not seen during training without raising a hard exception.
#    Implementation redacted. Interface: replaces the function in-place at
#    module import time before any Pipeline.transform() call is made.
# ─────────────────────────────────────────────────────────────────────────────
import sklearn.utils._encode

def _patched_check_unknown(values, known_values, return_mask=False):
    # [REDACTED] — proprietary unknown-category handling strategy
    raise NotImplementedError("Behavioral Foundry patch not included in Spotlight release.")

# In production: sklearn.utils._encode._check_unknown = _patched_check_unknown


# ─────────────────────────────────────────────────────────────────────────────
# 3. REQUEST SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
class PredictPayload(BaseModel):
    sector: str                  # "aviation" | "ecommerce"
    features: Dict[str, Any]    # Raw feature key-value map from client


# ─────────────────────────────────────────────────────────────────────────────
# 4. GLOBAL MODEL REGISTRY
#    Populated at lifespan startup. Keys are sector strings.
#    MODELS[sector]       → XGBClassifier (extracted from Pipeline)
#    PREPROCESSORS[sector]→ fitted ColumnTransformer
#    CAT_COLS_BY_SECTOR   → set of categorical column names per sector
# ─────────────────────────────────────────────────────────────────────────────
MODELS: Dict[str, Any] = {}
PREPROCESSORS: Dict[str, Any] = {}
CAT_COLS_BY_SECTOR: Dict[str, Set[str]] = {}


def _extract_cat_cols(preprocessor) -> Set[str]:
    """
    Derives the set of categorical column names from a fitted ColumnTransformer
    by inspecting transformer names for 'encoder' substrings.
    [REDACTED] — implementation withheld.
    """
    raise NotImplementedError("Categorical namespace derivation withheld in Spotlight release.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LIFESPAN — MODEL LOADING
#    FastAPI async context manager. Fires on uvicorn startup before the
#    event loop opens to requests. Loads both sector pipelines from disk,
#    extracts classifier and preprocessor from named Pipeline steps.
#    On shutdown, clears the registry to release memory.
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MODEL_PATHS = {
    # [REDACTED] — artifact paths withheld
    # "ecommerce": "models/hcim_E-comm_Stream_v1.joblib",
    # "aviation":  "models/hcim_Aviation_v1.joblib",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    for sector, path in SECTOR_MODEL_PATHS.items():
        if os.path.exists(path):
            pipe = joblib.load(path)
            MODELS[sector] = pipe.named_steps['classifier']
            PREPROCESSORS[sector] = pipe.named_steps['preprocessor']
            CAT_COLS_BY_SECTOR[sector] = _extract_cat_cols(PREPROCESSORS[sector])
            print(f"SYSTEM: Loaded [{sector}] model from {path}")
        else:
            print(f"CRITICAL: Artifact not found → {path}")
    yield
    MODELS.clear()
    PREPROCESSORS.clear()
    CAT_COLS_BY_SECTOR.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 6. FASTAPI APPLICATION CORE
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Neral AI Core", version="6.1-spotlight", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Constrained by API key at application layer
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 7. API KEY AUTHENTICATION
#    Secret injected via NERAL_SECRET environment variable (HF Space settings).
#    Never hardcoded in source. Validated via FastAPI Security dependency —
#    the dependency is declared at endpoint level, not middleware level, so
#    unauthenticated requests are rejected before entering the inference graph.
# ─────────────────────────────────────────────────────────────────────────────
_API_KEY_STORE = os.getenv("NERAL_SECRET")  # Must be set; no fallback in production
_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(header_key: str = Security(_api_key_header)) -> str:
    if header_key and header_key == _API_KEY_STORE:
        return header_key
    raise HTTPException(status_code=403, detail="Security Key Mismatch.")


@app.get("/")
def health_check():
    """Public health probe. Returns loaded sector list and engine version."""
    return {
        "status": "ONLINE",
        "engine": "v6.1-Atomic",
        "sectors": list(MODELS.keys()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. BEHAVIORAL FOUNDRY — FEATURE ALIGNMENT
#    Converts raw client payload to a DataFrame aligned with the preprocessor's
#    expected feature namespace. Case-insensitive key lookup, type coercion,
#    and categorical null handling are applied here.
#    [REDACTED] — full implementation withheld.
# ─────────────────────────────────────────────────────────────────────────────
def apply_feature_engineering(payload: dict, sector: str) -> pd.DataFrame:
    """
    Signature (public):
      payload : dict  — raw features from client JSON
      sector  : str   — "aviation" | "ecommerce"
      returns : pd.DataFrame with shape (1, N), columns = preprocessor.feature_names_in_

    [REDACTED] — Behavioral Foundry internals withheld in Spotlight release.
    """
    raise NotImplementedError("Behavioral Foundry withheld in Spotlight release.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. /v1/predict — INFERENCE ENDPOINT
#    Atomic SHAP Alignment pipeline:
#      1. apply_feature_engineering()  → aligned DataFrame
#      2. ColumnTransformer.transform() → processed numpy array
#      3. get_feature_names_out()      → Atomic Alignment namespace
#      4. shap.TreeExplainer()         → Shapley contribution vector
#      5. argmax(φ_j)                  → trigger_diagnosis
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/v1/predict")
async def predict(
    payload: PredictPayload,
    api_key: str = Security(require_api_key),
):
    sector = payload.sector.lower().strip()

    if sector not in MODELS:
        raise HTTPException(status_code=400, detail=f"Sector '{sector}' not in registry.")

    # Step 1: Behavioral Foundry alignment [REDACTED internals]
    df_aligned = apply_feature_engineering(payload.features, sector)

    # Step 2: Preprocessor transform
    processed = PREPROCESSORS[sector].transform(df_aligned)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()   # Sparse → dense for SHAP

    # Step 3: Atomic Alignment — strip OHE prefixes from feature names
    # This is the EAG closure step. Index displacement is eliminated here.
    feature_names = [
        col.split("__")[-1]
        for col in PREPROCESSORS[sector].get_feature_names_out()
    ]

    # Step 4: SHAP TreeExplainer
    # shap_values output format varies by XGBoost version:
    #   list  → binary classification, index [1] is churn class
    #   array → single matrix, used directly
    explainer = shap.TreeExplainer(MODELS[sector])
    shap_values = explainer.shap_values(processed)
    contributions = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    # Step 5: Build contribution map and resolve top driver
    driver_map = [
        {"name": name, "phi": float(phi)}
        for name, phi in zip(feature_names, contributions)
    ]

    top_driver = max(driver_map, key=lambda x: x["phi"])["name"]

    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": round(float(MODELS[sector].predict_proba(processed)[0, 1]), 4),
        "trigger_diagnosis": top_driver.upper().replace("_", " "),
    }
