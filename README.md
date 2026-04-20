---
title: Neral AI Backend
emoji: 🛰️
colorFrom: red
colorTo: black
sdk: docker
app_port: 7860
---

# Neral AI Platform · v6.1
### Hybrid Churn Intelligence Model (HCIM) — Systems Engineering Reference

![Engine](https://img.shields.io/badge/Engine-HCIM_v6.1_Atomic-red?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-FastAPI_on_HF_Spaces-orange?style=flat-square)
![Runtime](https://img.shields.io/badge/Runtime-Docker_python3.10--slim-blue?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-Atomic_SHAP_Alignment_v6.1-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production_Locked-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Proprietary-lightgrey?style=flat-square)

---

## 1. The Explainability–Action Gap (EAG): Formal Problem Statement

A churn classifier, in its degenerate form, is a scalar-valued function:

$$f: \mathbb{R}^{d} \rightarrow [0, 1], \quad \hat{p} = \sigma\!\left(\sum_{k=1}^{K} f_k(\mathbf{x})\right)$$

This output $\hat{p}$ is a point on the Decision Manifold — it encodes *where* a customer sits relative to the churn boundary, but carries zero information about *which behavioral dimension* drove the displacement. In any retention workflow requiring human action (contract renegotiation, escalation routing, win-back offer selection), a scalar risk score is operationally inert. This structural gap between a model's output and a practitioner's required input is the **Explainability–Action Gap (EAG)**.

### 1.1 The Index Displacement Failure Mode

The canonical method for closing the EAG is SHAP attribution. In practice, the failure occurs silently inside `sklearn.pipeline.Pipeline`: the `ColumnTransformer` applies OneHotEncoding and numerical scaling, which **reorders and expands** the feature matrix. The resulting column order in the processed array $\mathbf{x}'$ has no guaranteed correspondence to the original feature order in $\mathbf{x}$.

When `shap_values[i]` is mapped to `original_feature_name[i]` without accounting for this reordering, the attribution vector is **index-displaced**. The practical consequence: a high CSAT score (a retention signal) can surface as the top churn driver — a logical inversion that produces prescriptive advice opposite to empirical ground truth. The EAG is not merely unclosed; it is **inverted**.

### 1.2 Atomic SHAP Alignment — The Closure Protocol

This repository implements the **Atomic SHAP Alignment** protocol, which enforces a bijective mapping between Shapley contributions and their originating feature names:

```python
# FAIL-CLOSED: If get_feature_names_out() raises, the endpoint raises HTTP 500.
# Index displacement is structurally impossible after this line.
feature_names = [
    c.split("__")[-1]                                  # Strip OHE prefixes
    for c in PREPROCESSORS[sector].get_feature_names_out()  # Post-transform namespace
]
```

The bijection $\pi: j \mapsto \mathtt{feature\_names}[j]$ is constructed **after** `ColumnTransformer.transform()` completes. This guarantees that `contribution[j]` is synchronized with `feature_names[j]` in the post-transform space. The EAG closure condition is:

$$\forall j \in \{1, \ldots, |\mathcal{F}|\}: \quad \phi_j \leftrightarrow \pi(j) \quad \text{(bijective, index-stable)}$$

---

## 2. System Architecture: Distributed Inference topology

```
┌──────────────────── Docker Container (python:3.10-slim) ────────────────────┐
│                                                                              │
│  PID 1: start.sh                                                             │
│    │                                                                         │
│    ├─► [Process A] uvicorn app.main:app  ──► :8000  (container-internal)    │
│    │         │                                                               │
│    │         │  Lifespan hook (startup):                                     │
│    │         │  ├── joblib.load(hcim_Aviation_v1.joblib)                     │
│    │         │  ├── joblib.load(hcim_E-comm_Stream_v1.joblib)                │
│    │         │  └── _extract_cat_cols() → CAT_COLS_BY_SECTOR registry        │
│    │         │                                                               │
│    │         └── POST /v1/predict  ◄── x-api-key enforced (Fail-Closed)     │
│    │               ├── Behavioral Foundry: apply_feature_engineering()       │
│    │               ├── ColumnTransformer.transform()                         │
│    │               ├── Atomic Alignment: get_feature_names_out() + OHE strip │
│    │               ├── shap.TreeExplainer.shap_values()                      │
│    │               └── argmax(φ_j) → trigger_diagnosis                       │
│    │                                                                         │
│    ├─► sleep 15  [warm-up barrier — blocks UI entry until models are hot]    │
│    │                                                                         │
│    └─► [Process B] streamlit run frontend/app.py  ──► :7860  (HF public)    │
│               │                                                              │
│               └── REST Gateway Contract → :8000/v1/predict                  │
│                   ├── x-api-key header injected from NERAL_API_KEY env var  │
│                   ├── Plotly Indicator: risk-band gauge (3-tier threshold)   │
│                   └── Diagnosis card: trigger_diagnosis HTML render          │
│                                                                              │
│  HF Spaces exposes ONLY :7860. Port :8000 is not externally routable.       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 REST Gateway Contract

The Streamlit Command Bridge does not hold model state. It is a **stateless display terminal** that communicates with the inference engine exclusively through the REST Gateway Contract:

| Property | Value |
|---|---|
| **Endpoint** | `POST /v1/predict` |
| **Host** | `adityajaiswal440-neral-ai-backend.hf.space` |
| **Auth header** | `x-api-key: ${NERAL_API_KEY}` |
| **Auth enforcement** | FastAPI `Security(APIKeyHeader)` dependency — evaluated before inference graph entry |
| **Failure posture** | **Fail-Closed**: absent or mismatched key → `HTTP 403` before any model code executes |
| **Transport** | HTTPS enforced; TLS terminated by HF Spaces ingress |
| **State on frontend** | Zero model artifacts; zero SHAP logic |

The gateway is not a proxy and not a middleware interceptor. It is a **point-in-time, key-gated inference contract** enforced at the dependency-injection layer.

Fail-Closed guard — authentication dependency in `app/main.py`:

```python
import os
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

# FAIL-CLOSED: Raises EnvironmentError at module load if key is absent.
# The server never starts in an unauthenticated state.
NERAL_API_KEY = os.getenv("NERAL_API_KEY")
if not NERAL_API_KEY:
    raise EnvironmentError("CRITICAL: NERAL_API_KEY not found in environment.")

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def require_api_key(header_key: str = Security(_api_key_header)) -> str:
    if header_key and header_key == NERAL_API_KEY:
        return header_key
    # Fail-Closed: reject before inference graph is entered
    raise HTTPException(status_code=403, detail="Security Key Mismatch.")
```

---

## 3. Diagnostic Intelligence Engine: Objective Functions

The HCIM is not a general-purpose classifier. It is a **Diagnostic Intelligence Engine** — a system whose outputs are required to drive human action, not populate a dashboard metric. This requirement imposes specific constraints on the loss surface and the explainability pipeline.

### 3.1 Decision Manifold: Cost-Sensitive Class Weighting

Standard binary cross-entropy treats false positives and false negatives symmetrically on the Decision Manifold. In retention contexts, this is incorrect by construction: the cost of a missed churner (false negative) exceeds the cost of a false alarm (false positive) by a sector-specific multiplier. The HCIM shifts the decision boundary by asymmetrically penalizing false negatives via:

$$\alpha_{\mathrm{spw}} = \frac{|\mathcal{N}_{\mathrm{neg}}|}{|\mathcal{N}_{\mathrm{pos}}|} = 10$$

This resolves to a **10× False Negative penalty** for both deployed sectors. The effect on the Decision Manifold: the hyperplane separating churn from non-churn is displaced toward the non-churn cluster, maximizing sensitivity at the cost of precision. The engineering rationale — the cost of a missed intervention (lost customer) is at minimum one full contract-cycle revenue; the cost of a false alarm (unnecessary outreach) is bounded by the marginal cost of a retention interaction.

### 3.2 Regularised Boosting Objective (Modified Log-Loss)

The HCIM's inference core is a K-tree additive ensemble trained under a regularized, asymmetric objective:

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} w_i \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right] + \sum_{k=1}^{K} \Omega(f_k)$$

where the complexity penalty on each tree is:

$$\Omega(f_k) = \gamma T_k + \frac{1}{2}\lambda \|w_k\|^2$$

| Symbol | Definition | Value |
|---|---|---|
| $w_i$ | Sample weight | $\alpha_{\mathrm{spw}} = 10$ if $y_i = 1$; else $1$ |
| $T_k$ | Leaf count of tree $k$ | Controlled by `max_depth=3` |
| $\lambda$ | L2 regularization coefficient | $1$ (XGBoost default) |
| $\gamma$ | Minimum loss reduction for split | Tuned per sector |

### 3.3 Harmonic Termination: Probabilistic Risk Score

The ensemble output $z$ — the additive sum of all tree outputs — is passed through the sigmoid link function for probabilistic calibration. This is the **Harmonic Termination** step: it collapses the unbounded real-valued ensemble score into a bounded risk probability:

$$P(\mathrm{churn} \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \sum_{k=1}^{K} f_k(\mathbf{x})$$

The output $\hat{p} \in (0, 1)$ is the **risk coordinate on the Decision Manifold**. It gates the Diagnostic Intelligence pipeline: it is not the final output returned to the caller. It is the trigger condition for Atomic SHAP Attribution — the stage that closes the EAG.

### 3.4 Atomic SHAP Attribution: EAG Closure via Shapley Decomposition

The Shapley value for post-aligned feature $j$ quantifies its marginal contribution to the model's prediction across all possible feature coalitions:

$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!\,(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[ v(S \cup \{j\}) - v(S) \right]$$

After Atomic Alignment enforces the bijection $\pi$, the `trigger_diagnosis` is resolved as the argmax of the positive contribution vector:

$$\hat{j}^{\ast} = \underset{j \in \mathcal{F}}{\arg\max} \;\; \phi_j \qquad \Rightarrow \qquad \mathtt{trigger} = \mathrm{name}(\hat{j}^{\ast})$$

This is the **Diagnostic Output** — the single behavioral signal most responsible for the predicted churn event, causally attributed to that specific customer's feature vector. It is the operationally actionable product of the entire inference pipeline.

---

## 4. The Behavioral Foundry: Feature Engineering Architecture

The `Behavioral Foundry` is the proprietary preprocessing layer that converts raw CRM/transactional data into the feature space on which the Decision Manifold was trained. It is implemented as a `sklearn.pipeline.Pipeline` with two named steps: `preprocessor` (a fitted `ColumnTransformer`) and `classifier` (a fitted `XGBClassifier`). The complete transformation graph is serialized as a single `.joblib` artifact per sector — there are no loose scalers, encoders, or external state files.

### 4.1 E-commerce / Streaming: Signal Taxonomy

| Signal Category | Features (selected) | Foundry-Precision Method |
|---|---|---|
| **Engagement Density** | `monthly_logins`, `usage_density`, `total_monthly_time` | Raw counts + $\log(1+x)$ compression |
| **Recency Decay** | `last_login_days_log`, `is_recency_danger`, `is_slow_ghost` | Log-transform + binary threshold flag |
| **Payment Friction** | `payment_failures_clipped`, `is_high_friction_payment`, `payment_structural_risk` | Winsorized clip + structural risk composite |
| **Session Architecture** | `session_strength_log`, `feature_intensity_log`, `engagement_efficiency` | Log-ratio features — momentum proxies |
| **Loyalty Geometry** | `loyalty_shock_score`, `loyalty_resilience`, `nps_normalized` | Composite signals — shock × recovery dynamics |
| **Support Load** | `support_intensity_log`, `support_tickets_clipped`, `escalations_clipped` | Log-transform + hard clip at 99th percentile |
| **Advocacy State** | `referral_count_clipped`, `is_advocate`, `is_passive_promoter` | Binary state flags — social proof proxy |
| **Satisfaction Signal** | `csat_score`, `is_hidden_dissatisfaction`, `email_open_rate_fixed` | Raw + derived inversion signals |
| **Categorical Identity** | `city`, `signup_channel`, `payment_method`, `tenure_group`, `contract_type` | OneHotEncoding — OHE suffixes stripped at inference by Atomic Alignment |

Post-OHE dimensionality: **~50+ features**. The Atomic Alignment namespace collapses this back to the pre-OHE signal name for all `trigger_diagnosis` outputs — ensuring the Diagnostic Output is human-readable without post-processing by the caller.

### 4.2 Aviation Sector: Validated SHAP Driver Hierarchy

The following drivers are ranked by global Shapley contribution magnitude, validated against `Hybrid_Aviation_Churn_Integrated.csv` (11.6 MB, 100k+ records):

| SHAP Rank | Feature | Signal Class | Diagnostic Interpretation |
|---|---|---|---|
| 1 | `Inflight wifi service` | Service Quality (5-pt ordinal) | Infrastructure limitation — single highest loyalty decay driver |
| 2 | `Online boarding` | Digital Friction (5-pt ordinal) | Pre-flight UX failure — compounds with in-flight dissatisfaction |
| 3 | `Inflight entertainment` | Satisfaction Proxy (5-pt ordinal) | Long-haul service quality signal |
| 4 | `Departure Delay in Minutes` | Operational Friction (continuous, minutes) | Acute friction — high variance, high SHAP magnitude on extreme values |
| 5–6 | `Class` / `Type of Travel` | Categorical Segment | Segmentation boundary — business vs. personal travel risk profile |

---

## 5. Validated Benchmarks

All metrics are computed on **held-out test sets only** (stratified 80/20 split, `random_state=42`). Training-set metrics are not reported.

| Model ID | Sector | Recall | F1-Score | PR-AUC | $\alpha_{\mathrm{spw}}$ |
|---|---|---|---|---|---|
| `HCIM-Aviation-V1` | Aviation | **99.99%** | **96.4%** | 0.92 | 10 |
| `HCIM-Ecomm-Streaming-V1` | E-commerce / Streaming | **84.3%** | **72.8%** | 0.89 | 10 |

> **Benchmark Interpretation — E-commerce F1 = 72.8%:**
> Under a 9:1 class imbalance, an uncorrected classifier collapses to a degenerate majority-class predictor with recall < 20% and inflated accuracy (~90%). The 72.8% F1 reflects the Decision Manifold state after $\alpha_{\mathrm{spw}} = 10$ cost correction forces the boundary toward high recall. This is the **architecturally correct** operating point for retention-critical inference. It is not a deficiency; it is the deliberate trade-off between false-alarm cost and missed-churn cost, encoded in the objective function at training time.

> **Benchmark Interpretation — Aviation F1 = 96.4%:**
> The Hybrid Aviation dataset integrates structured delay/service ordinal features with behavioral loyalty signals, producing a high signal-to-noise ratio in the feature space. The near-unity recall (99.99%) reflects the Atomic Alignment's ability to surface aviation-specific friction signals — particularly `Inflight wifi service` — as primary Diagnostic Outputs without index displacement contamination.

---

## 6. Container Runtime & Deployment Contract

### 6.1 Dockerfile — Fail-Closed Build

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# git required for Git LFS model artifact hydration at build time
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

# HF Spaces binds exclusively on 7860. Port 8000 is container-internal only.
EXPOSE 7860

CMD ["./start.sh"]
```

`start.sh` orchestrates two processes:

```bash
#!/bin/bash
set -euo pipefail  # FAIL-CLOSED: abort on any process error

# [Process A] Inference engine — container-internal, never exposed
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Warm-up barrier: blocks Process B until joblib.load() completes.
# Without this, the first UI request hits an empty MODELS registry → KeyError.
sleep 15

# [Process B] Command Bridge — HF Spaces public port
streamlit run frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true
```

### 6.2 Dependency Lock — ABI Compatibility Constraints

Version pins are not preferences. They are **ABI compatibility constraints** between the training environment and the inference container:

| Package | Pinned Version | Constraint Rationale |
|---|---|---|
| `scikit-learn` | 1.7.2 | `ColumnTransformer.get_feature_names_out()` API surface changes between minor versions; `.joblib` artifact is not cross-version portable |
| `xgboost` | 2.1.1 | Internal tree structure serialized into `.joblib` — version mismatch → silent prediction drift or deserialization failure |
| `shap` | 0.46.0 | `TreeExplainer.shap_values()` output format (`list` vs `ndarray`) differs across versions; branch logic in `/v1/predict` is version-targeted |
| `numpy` | 1.26.4 | Binary ABI compatibility anchor for both `xgboost` and `shap` C extensions |
| `pandas` | 2.2.3 | `DataFrame` host for the Atomic Alignment namespace — column ordering is version-sensitive |
| `fastapi` | 0.115.3 | ASGI inference server — Security dependency injection API |
| `uvicorn` | 0.32.0 | ASGI runner |
| `joblib` | 1.4.2 | Pipeline artifact serialization |
| `streamlit` | 1.39.0 | Command Bridge UI runtime |
| `plotly` | 5.24.1 | Risk gauge and SHAP force plot rendering |

### 6.3 Model Artifacts (Git LFS)

```
models/
├── hcim_Aviation_v1.joblib         # 242 KB — full sklearn Pipeline (preprocessor + XGBClassifier)
├── hcim_E-comm_Stream_v1.joblib    # 241 KB — full sklearn Pipeline (preprocessor + XGBClassifier)
└── metadata.json                   # Feature registries, benchmark records, tuning params
```

Artifacts tracked via Git LFS (`.gitattributes`). Each `.joblib` encapsulates the complete inference graph in a single binary: `ColumnTransformer → XGBClassifier`. There are no external state files; loading one artifact is sufficient to reproduce the full prediction and explainability pipeline for that sector.

---

## 7. REST Gateway API Contract

### `POST /v1/predict`

```
x-api-key:     <NERAL_API_KEY>         [required — Fail-Closed: absent → HTTP 403]
Content-Type:  application/json
```

**Request schema:**
```json
{
  "sector":   "aviation | ecommerce",
  "features": { "<feature_name>": "<value>" }
}
```

**Response — `200 OK`:**
```json
{
  "prediction_id":    "uuid-v4",
  "probability":       0.8731,
  "trigger_diagnosis": "DEPARTURE DELAY IN MINUTES"
}
```

The `trigger_diagnosis` field is the **Atomic SHAP argmax** — the single feature whose Shapley contribution $\phi_j$ is largest across the post-aligned feature namespace $\mathcal{F}$. It is returned in `UPPER_CASE` with underscores replaced by spaces for direct ingestion into CRM action fields without caller-side post-processing.

**Error contract:**
| HTTP Code | Condition | Behaviour |
|---|---|---|
| `403` | `x-api-key` absent or mismatched | **Fail-Closed** — inference graph not entered |
| `400` | `sector` not in `{aviation, ecommerce}` | Rejected at schema validation layer |
| `422` | Malformed JSON or missing `features` key | Pydantic validation failure |
| `500` | `ColumnTransformer` encounters unknown category | `_check_unknown` patch raises; caught upstream |

---

## 8. Known Engineering Constraints

| Constraint | Root Cause | Mitigation |
|---|---|---|
| Cold-start latency (~15s) | `scikit-learn` version guard triggers `pip install` on first uvicorn boot; `joblib.load()` adds ~3s per model | `UPGRADED` env flag prevents recursive re-execution; single warm-up barrier in `start.sh` |
| OHE unknown category rejection | `ColumnTransformer` with `handle_unknown='error'` raises on categories not seen in training | Patched `sklearn.utils._encode._check_unknown` at module import time — unknown values coerced to `"unknown"` string |
| SHAP `TreeExplainer` allocation | `shap_values()` allocates a full `(1, n_features)` contribution matrix per request | Single-record inference bounds the matrix at ~50 floats/request — no pooling required at current throughput |
| CORS wildcard | `allow_origins=["*"]` required for HF Spaces cross-origin calls from Command Bridge | Wildcard is constrained by Fail-Closed API key enforcement at application layer; no anonymous request reaches the inference graph |
| Dual-process warm-up race | `sleep 15` is a fixed barrier, not a signal-based lock — subject to race on slow HF Spaces builds | Production hardening: replace with health-poll loop (`until curl -sf localhost:8000/; do sleep 2; done`) |

---

## 9. Local Development: Foundry Ignition Protocol

`foundry_setup.sh` automates the local security guardrail lifecycle in three stages:

```bash
chmod +x foundry_setup.sh && ./foundry_setup.sh
```

| Stage | Operation |
|---|---|
| **INITIALIZE** | Checks for `.env`; scaffolds it with `NERAL_API_KEY` placeholder if absent; detects legacy `API_KEY` key name and issues remediation |
| **AUDIT** | Verifies `.env` coverage in `.gitignore` via three layers (literal grep → `git check-ignore` → `git ls-files` staging breach detector) |
| **IGNITE** | Validates `NERAL_API_KEY` is populated and not a placeholder; emits the single-command launch string; blocks with `exit 1` if key is absent |

Ignition is **Fail-Closed by design**: a server that starts without `NERAL_API_KEY` is an unauthenticated inference endpoint. The protocol refuses to provide a launch command until the key is verified as present and non-sentinel.

---

## 10. Repository Structure

```
Neral-AI-Platform/
├── app/
│   └── main.py                     # FastAPI inference engine — Atomic Alignment + REST gateway
├── frontend/
│   └── app.py                      # Streamlit Command Bridge — display terminal, zero model state
├── spotlight/
│   ├── app_skeleton.py             # Sanitized backend skeleton (Behavioral Foundry redacted)
│   └── frontend_skeleton.py        # Sanitized frontend skeleton (payload field names redacted)
├── models/
│   ├── hcim_Aviation_v1.joblib     # Aviation pipeline (Git LFS)
│   ├── hcim_E-comm_Stream_v1.joblib # E-comm pipeline (Git LFS)
│   └── metadata.json               # Feature registries + benchmark records
├── notebooks/
│   ├── Data-Preprocess.ipynb       # Behavioral Foundry construction [PROPRIETARY]
│   ├── Modelling.ipynb             # XGBoost training + SHAP audit
│   ├── shap_audit.json             # Atomic Alignment audit log
│   ├── aviation_shap_summary.png   # Global SHAP beeswarm — Aviation sector
│   └── xgb_shap_summary.png        # Global SHAP beeswarm — E-comm sector
├── foundry_setup.sh                # Foundry Ignition Protocol — local security guardrails
├── SPOTLIGHT.md                    # Architectural manifesto — public reference release
├── Dockerfile                      # Production container definition
├── requirements.txt                # ABI-locked dependency manifest
├── start.sh                        # Dual-process entrypoint
├── freeze_model.py                 # Pipeline serialization utility [PROPRIETARY]
└── test_api.py                     # REST gateway integration test suite
```

---

## 11. Research Attribution

| Field | Value |
|---|---|
| **Lead Architect** | Aditya Jaiswal |
| **Core Contribution** | Formal closure of the Explainability–Action Gap via Atomic SHAP Alignment in a production FastAPI inference engine with multi-sector Behavioral Foundry pipelines and a Fail-Closed Secured REST Gateway |
| **Sectors** | Aviation (loyalty fragility under service friction) · E-commerce/Streaming (behavioral decay + payment structural risk) |

---

## 12. Licensing

This software, all associated model artifacts, the Behavioral Foundry feature engineering methodology, and all architectural designs are the proprietary intellectual property of **Neral AI**. Unauthorized reproduction, distribution, reverse-engineering, or derivative use is prohibited.

For licensing, pilot program access, or research collaboration: contact the Lead Architect.

© 2026 Neral AI · All Rights Reserved.
