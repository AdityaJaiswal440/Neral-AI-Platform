# Hybrid Churn Intelligence Model (HCIM)
## [SOURCE OF TRUTH - PHASE 2: MODELING IGNITION]

---

## 1. Project Scope & Datasets
- **Sectors:** E-commerce, Streaming, and Aviation.
- **Primary Data Assets:**
    1. `customer_churn_business_dataset.csv` (E-comm + Streaming)
    2. `Hybrid_Aviation_Churn_Integrated.csv` (Aviation - Authentic Survey Data)
- **Objective:** Cross-sector churn prediction using a unified Behavioral Foundry and Cost-Sensitive XGBoost logic.

---

## 2. The Behavioral Foundry (Core Signals)
Agnostic signals forged to bridge the "Friction-Loyalty Nexus":
- **Economic Intensity:** `monthly_fee_log`, `value_score_log`.
- **Friction Vectors:** `support_intensity_log` (E-comm), `service_friction_score` (Aviation).
- **Loyalty Anchors:** `loyalty_shock_score` (All), `loyalty_resilience` (Aviation).
- **Engagement Density:** `usage_density` (Streaming), `engagement_efficiency` (Aviation).

---

## 3. Current Project Stage: [PHASE 2 - MODELING]

### ✅ PHASE 1: DATA PREPARATION (COMPLETED)
- **Aviation:** Reconstructed using authentic logs; demographics (Age/Gender) purged.
- **E-comm/Streaming:** Synthesized behavioral features integrated; 9:1 imbalance confirmed.

### 🏗️ PHASE 2: MODELING (ACTIVE)
- **Task 1 (Aviation):** Ready for balanced baseline training (Standard weighting).
- **Task 2 (E-comm/Streaming):** - **Stratified 80/20 Split:** COMPLETED (Maintains ~10.2% churn ratio).
    - **SPW Calculation:** COMPLETED (Dynamic `scale_pos_weight` = 8.79).
- **Current Task:** Execute `model.fit()` and optimize for **Recall** using `eval_metric='aucpr'`.

---

## 4. Active Workspace Handshake (Antigravity Context)
- **Target:** `churn` (Binary 0/1).
- **Engine:** XGBoost (Extreme Gradient Boosting).
- **Model Parameters:** - `scale_pos_weight`: 8.79 (For E-comm) / 1.0 (For Aviation).
    - `eval_metric`: 'aucpr' (Focus on Precision-Recall Curve).
- **Validation:** SHAP Summary Plots are MANDATORY for transparency.

---

## 5. HCIM Executive Instructions (Persona: Data Architect)
- **Recall-First Mandate:** In an imbalanced 9:1 world, Accuracy is a "Liar." We prioritize catching the churner over global accuracy.
- **No Demographics:** Never suggest modeling based on Age, Gender, or Location. Focus exclusively on **Behavior and Friction**.
- **Mathematical Integrity:** Always use Stratified splits to prevent sampling bias.
- **Explainability:** No training is final until SHAP identifies the "Why" behind the "Who."