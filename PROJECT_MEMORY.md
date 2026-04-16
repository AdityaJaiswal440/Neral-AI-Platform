Neral AI: Unified Churn Intelligence
[SOURCE OF TRUTH - PHASE 3: DEPLOYMENT & DASHBOARD]
1. Project Scope & Hardened Assets
Sectors: E-commerce (HCIM-E) and Aviation (HCIM-A).

Models: Cost-Sensitive XGBoost (SPW: 8.79 for E-comm) utilizing a Behavioral Foundry approach.

Infrastructure: Hugging Face Spaces (16GB RAM Engine).

Status: Backend Operational. The "Unified Core" is live and serving RESTful inference requests.

2. The Hardened Architecture (The "Nuclear" Shield)
To survive the transition from training to production, the system was hardened against three critical failure points:

Dependency Shield: Enforced strict versioning (numpy>=1.26.4,<2.0.0) to prevent binary-level isnan ufunc crashes.

Surgical Memory Patching: Implementation of a startup routine that "scrubs" serialized .joblib files to remove NaN sentinels baked in during the training phase.

Type-Safe Engineering: Deployment of a safe_float utility to reconcile categorical "Yes/No" inputs into the numeric format required by XGBoost.

3. Current Project Stage: [PHASE 3 - FRONTEND INTEGRATION]
✅ PHASE 1: DATA PREPARATION (COMPLETED)
Behavioral signals forged; demographics purged.

✅ PHASE 2: MODELING & BACKEND DEPLOYMENT (COMPLETED)
Model Fit: High-Recall training via aucpr optimization.

FastAPI Implementation: Lifespan-managed REST API with secure x-api-key headers.

Hugging Face Migration: Successfully moved from Render to a 16GB Dedicated Space.

Live Verification: Successful inference confirmed (e.g., Probability: 0.5743; Trigger: csat_score).

🏗️ PHASE 3: STREAMLIT DASHBOARD (ACTIVE)
Task 1: Initialize frontend/app.py on a secondary Hugging Face Space.

Task 2: Connect UI widgets (sliders/dropdowns) to the /v1/predict endpoint.

Task 3: Render real-time SHAP force plots for local explainability.

4. Active Workspace Handshake (Production Context)
Backend URL: https://adityajaiswal440-neral-ai-backend.hf.space/

API Endpoint: /v1/predict

Diagnostics: SHAP Engine Verified. The system successfully identifies the "Top Driver" for every prediction.

Security: NERAL_SECRET_2026 gate active.

5. HCIM Executive Instructions (Persona: Data Architect)
Decoupled Architecture: Maintain strict separation between the "Brain" (FastAPI Backend) and the "Face" (Streamlit Frontend).

Friction-First UI: The dashboard must emphasize behavioral inputs over static traits.

Explainability Mandate: Every prediction displayed on the dashboard must be accompanied by its SHAP diagnosis. A prediction without a "Why" is a failure.

Zero-Downtime Mentality: Any updates to the feature engineering must be tested against the "Nuclear Patch" to prevent regression of the NumPy 2.0 crash.