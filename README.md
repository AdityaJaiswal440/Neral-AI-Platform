---
title: Neral AI Backend
emoji: 🛰️
colorFrom: red
colorTo: black
sdk: docker
app_port: 7860
---

# Neral AI Platform · v6.1
### Hybrid Churn Intelligence Model (HCIM) — Engineering Reference

![Engine](https://img.shields.io/badge/Engine-HCIM_v6.1_Atomic-red?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-FastAPI_on_HF_Spaces-orange?style=flat-square)
![Runtime](https://img.shields.io/badge/Runtime-Docker_python3.10--slim-blue?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-Atomic_SHAP_Alignment_v6.1-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production_Locked-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Proprietary-lightgrey?style=flat-square)

---

## 1. Problem Statement: The Explainability–Action Gap (EAG)

Standard churn classifiers produce a scalar probability, $\hat{p} \in [0, 1]$. In high-stakes retention workflows, this scalar is operationally inert. The downstream action — whether to trigger a win-back offer, escalate to a senior CSM, or adjust contract terms — requires a *causally attributed, customer-specific* explanation that is **mathematically synchronized** with the prediction.

The core failure mode in legacy SHAP pipelines is **index displacement**: the preprocessor's `ColumnTransformer` reorders and expands features (OHE expansion, log transforms) such that `shap_values[i]` no longer corresponds to the raw feature at position `i`. When this displacement is ignored, the top churn driver returned may be a high-CSAT score — a logical inversion that produces prescriptive advice opposite to ground truth.

This repository implements the **Atomic SHAP Alignment** protocol to close the EAG entirely:

> After transformation, the processed feature matrix is re-anchored to a named `pd.DataFrame` constructed from `preprocessor.get_feature_names_out()`, with OHE suffixes stripped. All SHAP contributions are computed and ranked against this aligned namespace, guaranteeing that `contribution[i]` maps to `feature_name[i]` without residual index displacement.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NERAL AI PLATFORM v6.1                           │
│                                                                     │
│  ┌──────────────────────┐        ┌──────────────────────────────┐   │
│  │   Streamlit Command  │        │   FastAPI Inference Engine   │   │
│  │       Bridge         │        │   (Hugging Face Spaces)      │   │
│  │                      │        │                              │   │
│  │  frontend/app.py     │◄──────►│  app/main.py                 │   │
│  │  Neon Carbon UI      │  REST  │  /v1/predict  (POST)         │   │
│  │  Plotly Risk Gauge   │  +     │  x-api-key header enforced   │   │
│  │  Diagnosis Card      │  HMAC  │  HCIM-Ecomm + HCIM-Aviation  │   │
│  └──────────────────────┘  Key   └──────────────────────────────┘   │
│                                          │                          │
│                              ┌───────────▼──────────┐              │
│                              │  Behavioral Foundry   │              │
│                              │   (Preprocessing)     │              │
│                              │                       │              │
│                              │  ColumnTransformer    │              │
│                              │  OHE + Log Scaling    │              │
│                              │  Atomic Alignment     │              │
│                              └───────────┬──────────┘              │
│                                          │                          │
│                   ┌──────────────────────┴──────────────────┐       │
│                   │                                         │       │
│       ┌───────────▼──────────┐              ┌──────────────▼────┐  │
│       │  HCIM-Aviation-V1   │              │  HCIM-Ecomm-V1    │  │
│       │  hcim_Aviation_v1   │              │  hcim_E-comm_      │  │
│       │  .joblib            │              │  Stream_v1.joblib  │  │
│       │  XGBoost + Pipeline │              │  XGBoost + Pipeline│  │
│       └─────────────────────┘              └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Secured REST Gateway

The Streamlit Command Bridge communicates exclusively with the Hugging Face Spaces backend via a **Secured REST Gateway**:

- **Endpoint:** `POST https://adityajaiswal440-neral-ai-backend.hf.space/v1/predict`
- **Auth:** `x-api-key` header, validated server-side against the `NERAL_SECRET` environment variable via FastAPI `Security` dependency injection.
- **Failure Mode:** Any request with an absent or mismatched key receives `HTTP 403 — Security Key Mismatch` before the inference graph is entered.
- **Transport:** All inter-service calls traverse HTTPS. The Hugging Face runtime enforces TLS termination at the ingress.

The gateway is **not a proxy** — it is a stateless, key-authenticated inference contract. The backend holds all model state; the frontend holds zero model artifacts.

---

## 3. Objective Functions

### 3.1 Cost-Sensitive XGBoost (Both Sectors)

The HCIM uses XGBoost's built-in cost-sensitivity parameter `scale_pos_weight` to penalize False Negatives (missed churn) at a ratio calibrated to the training set imbalance:

$$\alpha_{\mathrm{spw}} = \frac{|\mathcal{N}_{\mathrm{neg}}|}{|\mathcal{N}_{\mathrm{pos}}|}$$

For both deployed models, this resolves to **10**, meaning the model incurs a 10× cost penalty for each missed churner relative to a false alarm. This asymmetric loss surface forces the decision boundary toward high recall at the cost of precision — the correct engineering trade-off for retention-critical deployments where the cost of inaction exceeds the cost of intervention.

### 3.2 XGBoost Gradient & Hessian (Log-Loss, Modified)

The underlying boosting objective is the regularised log-loss with L2 penalty on leaf weights:

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} \left[ w_i \cdot \left( y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right) \right] + \sum_{k=1}^{K} \Omega(f_k)$$

where:
- $w_i = \alpha_{\mathrm{spw}}$ if $y_i = 1$, else $w_i = 1$ &nbsp;&nbsp;*(where $\alpha_{\mathrm{spw}} = 10$, the class imbalance ratio)*
- $\Omega(f_k) = \gamma T_k + \frac{1}{2}\lambda \|w_k\|^2$ — tree complexity regularizer
- $T_k$ = number of leaves in tree $k$, $\lambda = 1$ (L2)

### 3.3 Probabilistic Churn Score

The final churn probability is the harmonically terminated sigmoid of the additive ensemble output $z$, the sum of contributions from all $K$ trees:

$$P(\text{churn} \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \sum_{k=1}^{K} f_k(\mathbf{x})$$

This scalar is **not** the diagnostic output — it is the risk signal that gates whether Atomic SHAP Alignment is invoked for causal attribution.

### 3.4 Atomic SHAP Attribution (EAG Closure Condition)

The Shapley value for feature $j$ is computed over the coalition of all features in the aligned namespace $\mathcal{F}$:

$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!\,(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ v(S \cup \{j\}) - v(S) \right]$$

The EAG is closed if and only if the index map $\pi: j \mapsto \mathtt{names}[j]$ is bijective and constructed **after** `ColumnTransformer.transform()`, where $\mathtt{names}$ is the output of `get_feature_names_out()`. This is enforced in `app/main.py:132`:

```python
feature_names = [c.split('__')[-1] for c in PREPROCESSORS[sector].get_feature_names_out()]
```

The top churn driver is then:

$$\hat{j}^{\,*} = \underset{j \in \mathcal{F}}{\arg\max} \; \phi_j \qquad \Rightarrow \qquad \mathtt{trigger} = \mathrm{name}(\hat{j}^{\,*})$$

---

## 4. The Behavioral Foundry (Feature Engineering)

The preprocessing pipeline — termed the **Behavioral Foundry** — operates as a `sklearn.pipeline.Pipeline` with two named steps: `preprocessor` (a `ColumnTransformer`) and `classifier` (an `XGBClassifier`). Each sector's Foundry is serialized as a single `.joblib` artifact, preserving the full transformation graph.

### 4.1 E-commerce / Streaming Sector · Feature Taxonomy

| Category | Features | Engineering Method |
|---|---|---|
| **Engagement Density** | `monthly_logins`, `weekly_active_days`, `usage_density`, `total_monthly_time` | Raw + log-transform |
| **Recency Signals** | `last_login_days_log`, `is_recency_danger`, `is_slow_ghost` | Log-transform + binary flag |
| **Payment Friction** | `payment_failures_clipped`, `is_high_friction_payment`, `payment_structural_risk` | Winsorized + binary |
| **Session Quality** | `session_strength_log`, `feature_intensity_log`, `engagement_efficiency` | Log-transform ratio |
| **Loyalty Architecture** | `loyalty_shock_score`, `loyalty_resilience`, `nps_normalized` | Composite |
| **Support Load** | `support_intensity_log`, `support_tickets_clipped`, `escalations_clipped` | Log + clip |
| **Advocacy State** | `referral_count_clipped`, `is_advocate`, `is_passive_promoter` | Binary + clip |
| **Satisfaction Signal** | `csat_score`, `is_hidden_dissatisfaction`, `email_open_rate_fixed` | Raw + derived |
| **Categorical** | `city`, `signup_channel`, `payment_method`, `tenure_group`, `complaint_type`, `customer_segment`, `contract_type` | OneHotEncoding |

Total feature dimensionality (post-OHE): **~50+ columns**. The Atomic Alignment strips OHE prefixes at inference time, collapsing back to the named behavioral signal for human-readable diagnostics.

### 4.2 Aviation Sector · Feature Taxonomy

| Validated Driver | SHAP Rank | Signal Type |
|---|---|---|
| `Inflight wifi service` | #1 | Service Quality (ordinal) |
| `Online boarding` | #2 | Digital Friction (ordinal) |
| `Inflight entertainment` | #3 | Satisfaction Proxy (ordinal) |
| `Departure Delay in Minutes` | #4 | Operational Friction (continuous) |
| `Class` / `Type of Travel` | #5–6 | Categorical Segment |

---

## 5. Validated Benchmarks

All metrics are derived from held-out test sets (stratified 80/20 split, seed-locked) and are **not** training-set metrics.

| Model ID | Sector | Recall | F1-Score | PR-AUC | `scale_pos_weight` |
|---|---|---|---|---|---|
| `HCIM-Aviation-V1` | Aviation | **99.99%** | **96.4%** | 0.92 | 10 |
| `HCIM-Ecomm-Streaming-V1` | E-commerce / Streaming | **84.3%** | **72.8%** | 0.89 | 10 |

> **Validated Benchmark Note — E-commerce:** The F1 of **72.8%** is architecturally correct under severe class imbalance (≈9:1 negative-to-positive). Optimizing for F1 at this imbalance ratio without `scale_pos_weight` correction produces a degenerate classifier with recall < 20%. The 72.8% figure represents the decision boundary after enforcing False Negative cost-weighting; it is not a ceiling.

> **Validated Benchmark Note — Aviation:** The **85% F1 (ensemble)** is the result of the Hybrid Aviation dataset integrating both structured delay/service features and behavioral loyalty signals. This benchmark is validated against `notebooks/Hybrid_Aviation_Churn_Integrated.csv` (11.6 MB, 100k+ records).

---

## 6. Deployment Stack

### 6.1 Container Runtime

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git
COPY requirements.txt . && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start.sh
EXPOSE 7860          # Hugging Face Spaces port binding
CMD ["./start.sh"]
```

The container is a single-layer Python 3.10 slim image. Git is installed to support LFS object hydration on cold start. The `start.sh` entrypoint invokes `uvicorn` with the `app.main:app` module path.

### 6.2 Dependency Versions (Production-Locked)

| Package | Pinned Version | Role |
|---|---|---|
| `fastapi` | 0.115.3 | ASGI inference server |
| `uvicorn` | 0.32.0 | ASGI runner |
| `xgboost` | 2.1.1 | Gradient boosting classifier |
| `shap` | 0.46.0 | TreeExplainer attribution |
| `scikit-learn` | 1.7.2 | Pipeline / ColumnTransformer |
| `pandas` | 2.2.3 | Atomic Alignment DataFrame host |
| `numpy` | 1.26.4 | SHAP value array substrate |
| `joblib` | 1.4.2 | Model artifact serialization |
| `streamlit` | 1.39.0 | Command Bridge UI runtime |
| `plotly` | 5.24.1 | Risk gauge rendering |

### 6.3 Model Artifacts (Git LFS)

```
models/
├── hcim_Aviation_v1.joblib       # 242 KB — full sklearn Pipeline
├── hcim_E-comm_Stream_v1.joblib  # 241 KB — full sklearn Pipeline
└── metadata.json                 # Sector configs, feature registries
```

Both `.joblib` files are tracked via **Git LFS** (see `.gitattributes`). Each artifact encapsulates the complete inference graph: `preprocessor → classifier`. There are no loose transformers or external scaler files.

---

## 7. API Contract

### `POST /v1/predict`

**Headers:**
```
x-api-key: <NERAL_SECRET>
Content-Type: application/json
```

**Request Body:**
```json
{
  "sector": "aviation | ecommerce",
  "features": {
    "<feature_name>": "<value>"
  }
}
```

**Response `200 OK`:**
```json
{
  "prediction_id": "uuid-v4",
  "probability": 0.8731,
  "trigger_diagnosis": "DEPARTURE DELAY IN MINUTES"
}
```

**Error Codes:**
| Code | Condition |
|---|---|
| `403` | API key absent or mismatched |
| `400` | Sector string not in `{aviation, ecommerce}` |
| `500` | Preprocessor transform failure (unknown category) |

The `trigger_diagnosis` field is the Atomically Aligned SHAP argmax — the single feature with the highest positive Shapley contribution toward the churn class, returned in uppercase with underscores replaced by spaces for direct ingestion into CRM action fields.

---

## 8. Repository Structure

```
Neral-AI-Platform/
├── app/
│   └── main.py                    # FastAPI core — inference + Atomic Alignment
├── frontend/
│   └── app.py                     # Streamlit Command Bridge UI
├── models/
│   ├── hcim_Aviation_v1.joblib    # Aviation sector pipeline (LFS)
│   ├── hcim_E-comm_Stream_v1.joblib # E-comm sector pipeline (LFS)
│   └── metadata.json              # Feature registries + benchmark records
├── notebooks/
│   ├── Data-Preprocess.ipynb      # Behavioral Foundry construction
│   ├── Modelling.ipynb            # XGBoost training + SHAP audit
│   ├── shap_audit.json            # Atomic Alignment audit log
│   ├── aviation_shap_summary.png  # Global SHAP beeswarm (Aviation)
│   ├── xgb_shap_summary.png       # Global SHAP beeswarm (E-comm)
│   └── aviation_waterfall.png     # Local SHAP waterfall (single record)
├── data/                          # Raw + processed datasets (gitignored)
├── scripts/                       # Utility scripts (freeze, validation)
├── Dockerfile                     # Production container definition
├── requirements.txt               # Dependency lock file
├── start.sh                       # Uvicorn entrypoint
├── freeze_model.py                # Pipeline serialization utility
└── test_api.py                    # Integration test suite
```

---

## 9. Known Engineering Constraints

| Constraint | Root Cause | Current Resolution |
|---|---|---|
| Cold-start latency (~8s) | `scikit-learn` version guard triggers `pip install` on first boot | `UPGRADED` env flag prevents re-execution on subsequent requests |
| OHE category mismatch on unseen values | `ColumnTransformer` rejects unknown strings | Patched `_check_unknown` in `sklearn.utils._encode` at boot |
| SHAP `TreeExplainer` memory peak | Full `shap_values` matrix allocation per request | Single-record inference; matrix is `(1, n_features)` — bounded at ~50 floats |
| CORS wildcard | `allow_origins=["*"]` — intentional for HF Spaces cross-origin | Constrained by API key enforcement at application layer |

---

## 10. Research Context

**Institution:** Gati Shakti Vishwavidyalaya (GSV), Vadodara — Artificial Intelligence & Data Science  
**Lead Architect:** Aditya Kumar  
**Sectors Targeted:** Aviation (loyalty fragility under service friction), E-commerce/Streaming (behavioral decay and payment structural risk)  
**Core Contribution:** Formal closure of the Explainability–Action Gap via Atomic SHAP Alignment in a production FastAPI inference engine, with multi-sector Behavioral Foundry pipelines and a Secured REST Gateway to a Streamlit Command Bridge.

---

## 11. Licensing

This software and all associated model artifacts, feature engineering methodologies, and architectural designs are the proprietary intellectual property of **Neral AI**. Unauthorized reproduction, distribution, or derivative use is prohibited.

For licensing, pilot program access, or research collaboration inquiries, contact the Lead Architect directly.

© 2026 Neral AI · All Rights Reserved.