# Neral AI: Universal Hybrid Churn Intelligence (UHCI)
## Project Identity & Vision
- **Organization:** Neral AI
- **Product:** Universal Hybrid Churn Intelligence (UHCI) System.
- **Model Type:** Domain-Agnostic Model-as-a-Service (MaaS).
- **Core Philosophy:** "The Google Translate for Business Risk." Converting raw industrial logs into Explainable Retention Strategies.
- **Target Market:** The Indian Quintet (Aviation, Retail, Logistics, SaaS, Warehousing).

---

## Technical Architecture
### 1. The Agnostic Layer (Universal Schema)
Translates domain-specific data into normalized behavioral vectors:
- **Temporal Recency:** (Time since last value exchange).
- **Intensity Decay:** (Usage growth rate drops).
- **Friction Accumulation:** (Cumulative support/SLA failures).

### 2. The Hybrid Brain
- **Engine:** XGBoost (Extreme Gradient Boosting).
- **Base Learner:** CART (Decision Trees) with Second-Order Gradient (Hessian) optimization.
- **Imbalance Strategy:** Cost-Sensitive Learning using `scale_pos_weight` (~9.0) to handle 9:1 data imbalance.
- **Explainability:** SHAP (Global Insight) + LIME (Local Actionable Reason Codes).

### 3. The Behavioral Foundry (Feature Synthesis)
Critical Engineered Features (The "Neral" Edge):
- **Loyalty Shock Score:** `Tenure_Map * Price_Increase_Flag`
- **Support Intensity Log:** `ln(1 + Tickets_Clipped * Res_Time)`
- **Engagement Efficiency:** `Marketing_Clicks / (Email_Opens + 0.01)`
- **Value Score Log:** `ln(1 + Usage_Time / (Monthly_Fee + 1))`

---

## Current Project Stage: [PHASE 2 - MODELING]
### ✅ PHASE 1: THE FOUNDRY (COMPLETED)
- **Data Integrity:** Raw logs processed; demographic noise (age/gender) purged.
- **Multicollinearity Fix:** 15+ redundant features dropped to stabilize the Hessian.
- **Synthesis:** 32 High-Fidelity Numerical signals forged.

### 🏗️ PHASE 2: MODELING (IN PROGRESS)
- **Status:** XGBoost logic defined. Train-test split (stratified) ready.
- **Current Task:** Hyperparameter tuning and first training run.
- **Next Step:** SHAP value integration for the "Glass Box" effect.

### ⏳ PHASE 3: INTEGRATION (PENDING)
- **Front-End:** React.js Dashboard (Real-time risk alerts).
- **Back-End:** FastAPI (Model-as-a-Service endpoint).

---

## Metadata & Guidelines for Antigravity Model
- **Persona:** Full-Stack AI Architect / Founder Mindset.
- **Tone:** Technical, Direct, Revenue-Focused.
- **Constraints:** No low-fidelity demographics. Prioritize Recall over Accuracy.

---
## Active Workspace (IDE Handshake)
- **Primary Dataframe:** `df_clean` (Shape: [N, 40])
- **Target Variable:** `churn` (Binary: 0, 1)
- **Feature Split:** 32 Numerical (`num_features`), 7 Categorical (`cat_features`).
- **Engine Setup:** - `scale_pos_weight` calculated as `total_negative / total_positive`.
- **Library Stack:** `xgboost`, `shap`, `lime`, `sklearn`, `fastapi`, `react`.