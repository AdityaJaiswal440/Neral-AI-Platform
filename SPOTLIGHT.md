# NERAL AI · SPOTLIGHT ARCHITECTURE MANIFESTO
### Distributed MLOps Orchestration — Public Reference Release
#### Classification: SPOTLIGHT · Proprietary training logic, model weights, and Behavioral Foundry internals are withheld.

---

## 0. Redaction Scope

This document and accompanying code skeletons constitute the **Spotlight Release** of the Neral AI v6.1 platform.
The following are **explicitly withheld** from this release:

| Withheld Asset | Classification |
|---|---|
| `notebooks/Data-Preprocess.ipynb` | Proprietary — Behavioral Foundry feature construction |
| `freeze_model.py` (full) | Proprietary — XGBoost training hyperparameters and imbalance strategy |
| `models/*.joblib` | Proprietary — Serialized pipeline weights (tracked via Git LFS, not public) |
| `apply_feature_engineering()` internals | Proprietary — Input alignment and type coercion logic |
| `CAT_COLS_BY_SECTOR` construction | Proprietary — Categorical namespace derivation |
| `notebooks/HCIM_Current_State_Fixed.csv` | Proprietary — Cleaned training corpus |

What **is** published:
- The distributed process orchestration model (dual-process Docker container)
- The FastAPI secured inference gateway contract and authentication architecture
- The Atomic SHAP Alignment interface contract (input/output schema, not implementation)
- The Streamlit Command Bridge connection logic and SHAP force plot rendering pipeline

---

## 1. Distributed Process Architecture

The platform runs as a **dual-process monolith** inside a single Docker container, orchestrated by `start.sh`.
This is a deliberate latency trade-off: cold-start isolation is accepted in exchange for eliminating a separate
service registry and inter-container networking overhead on Hugging Face Spaces infrastructure.

```
┌─────────────────────── Docker Container (python:3.10-slim) ───────────────────────┐
│                                                                                   │
│  PID 1: start.sh (bash)                                                           │
│    │                                                                              │
│    ├──► [Process A]  uvicorn app.main:app  --port 8000  (background, &)           │
│    │         │                                                                    │
│    │         │   Lifespan hook fires on startup:                                  │
│    │         │   ├── joblib.load(models/hcim_Aviation_v1.joblib)                  │
│    │         │   ├── joblib.load(models/hcim_E-comm_Stream_v1.joblib)             │
│    │         │   └── _extract_cat_cols() → CAT_COLS_BY_SECTOR registry            │
│    │         │                                                                    │
│    │         └── /v1/predict  [POST, API-key gated]                               │
│    │               └── apply_feature_engineering() ← [REDACTED]                  │
│    │               └── ColumnTransformer.transform()                              │
│    │               └── shap.TreeExplainer.shap_values()                           │
│    │               └── Atomic Alignment → trigger_diagnosis                       │
│    │                                                                              │
│    ├──► sleep 15  (model warm-up guard — blocks Process B entry)                  │
│    │                                                                              │
│    └──► [Process B]  streamlit run frontend/app.py  --port 7860  (foreground)     │
│               │                                                                   │
│               └── HTTP POST → localhost:8000/v1/predict  (Secured REST Gateway)  │
│               └── Plotly Indicator gauge render                                   │
│               └── Diagnosis card HTML injection                                   │
│                                                                                   │
│  Hugging Face Spaces exposes ONLY port 7860 to the public internet.               │
│  Port 8000 is not routable externally — inference engine is container-internal.   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Warm-Up Guard

The `sleep 15` in `start.sh` is a **hard synchronization barrier**. Without it, Streamlit's first request
fires before `joblib.load()` completes — resulting in a `KeyError` on `MODELS[sector]`.
A production-grade alternative is a `/health` polling loop on Process B's side:

```bash
#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Health-poll instead of blind sleep
until curl -sf http://localhost:8000/ > /dev/null; do
    echo "SYSTEM: Waiting for inference engine..."
    sleep 2
done

streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
```

---

## 2. Secured REST Gateway Contract

The Streamlit Command Bridge communicates with the FastAPI inference engine via an
**authenticated internal REST gateway**. The contract is as follows:

```
POST /v1/predict
Host: <container-internal or HF Spaces public URL>
x-api-key: <NERAL_SECRET env var>
Content-Type: application/json

{
  "sector": "aviation | ecommerce",
  "features": { "<feature_name>": <value> }
}

→ 200 OK
{
  "prediction_id": "<uuid-v4>",
  "probability": <float, 4dp>,
  "trigger_diagnosis": "<UPPERCASE FEATURE NAME>"
}

→ 403  API key absent or mismatched
→ 400  Sector string not in registered set
→ 500  Preprocessor transform failure
```

**Authentication Architecture:**
- Key stored as `NERAL_SECRET` environment variable in the HF Space settings (not in source)
- Injected at request time via FastAPI `Security(APIKeyHeader)` dependency
- Failure on mismatch raises `HTTP 403` before the inference graph is entered
- No rate limiting is implemented at application layer (HF Spaces enforces upstream throttling)

---

## 3. Atomic SHAP Alignment — Interface Contract

The internal SHAP pipeline follows the **Atomic Alignment** protocol. The implementation is withheld.
The interface contract (input/output) is published for integration purposes:

**Input:** A `(1, N)` numpy array — the output of `ColumnTransformer.transform()` on a single record.

**Internal operation (redacted):** SHAP `TreeExplainer` computes Shapley values; the contribution vector
is re-anchored to a named feature namespace derived from `get_feature_names_out()` with OHE suffixes stripped.

**Output contract:**
```python
{
    "trigger_diagnosis": str,   # argmax(φ_j) over positive contributions only
    "probability": float,       # σ(z) from XGBClassifier.predict_proba()[:, 1]
    "prediction_id": str        # UUID-v4, used as audit trace key
}
```

The `trigger_diagnosis` field is guaranteed to reference a **pre-OHE** feature name.
It will never return an OHE-expanded suffix like `payment_method_credit_card`.

---

## 4. MLOps Freeze-and-Serve Pipeline

The training-to-deployment pipeline is a single-pass serialization chain:

```
[Notebook: Data-Preprocess.ipynb]   ← REDACTED
         │  Behavioral Foundry output: cleaned DataFrame
         ▼
[freeze_model.py]                   ← PARTIALLY REDACTED
         │  sklearn Pipeline: preprocessor → XGBClassifier
         │  joblib.dump() → models/*.joblib
         │  Integrity check: predict_proba() on held-out index
         ▼
[Git LFS push]
         │  .joblib artifacts tracked in .gitattributes
         │  Hydrated at container build time via git-lfs pull
         ▼
[Docker build → HF Spaces]
         │  COPY . . in Dockerfile includes models/
         │  joblib.load() fires at uvicorn lifespan startup
         ▼
[Live inference via /v1/predict]
```

No online learning. No model registry. No feature store.
This is a **frozen-weight, single-artifact** deployment pattern. Model updates require a full
freeze → serialize → push → rebuild cycle.

---

## 5. Dependency Lock Rationale

| Package | Version | Lock Rationale |
|---|---|---|
| `scikit-learn` | 1.7.2 | `ColumnTransformer.get_feature_names_out()` API surface changed between minor versions — must match training environment exactly |
| `xgboost` | 2.1.1 | `.joblib` artifact is not cross-version portable; XGBoost internal tree structure changes across releases |
| `shap` | 0.46.0 | `TreeExplainer` output format (`shap_values` list vs array) differs between versions — version-locked to match training-time explainer |
| `numpy` | 1.26.4 | Binary ABI compatibility with both `xgboost` and `shap` C extensions |

A dynamic environment synchronization guard (`UPGRADED` env flag + `subprocess.check_call`) is
implemented in `app/main.py` to enforce `scikit-learn==1.7.2` at container startup on HF Spaces,
where the base image may carry a different version.

---

*Neral AI · Spotlight Release · © 2026 · All proprietary internals withheld.*
