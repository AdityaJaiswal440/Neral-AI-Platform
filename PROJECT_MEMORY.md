# Hybrid Churn Intelligence Model (HCIM)
## [SOURCE OF TRUTH - THREE-SECTOR SHIELD]

---

## 1. Project Scope
- **Sectors:** E-commerce, Streaming, and Aviation.
- **Datasets:**
 1.`customer_churn_business_dataset.csv` (E-comm + Streaming)
    2. `Hybrid_Aviation_Churn_Integrated.csv` (Aviation)
- **Objective:** Cross-sector churn prediction using a unified Behavioral Foundry.

---

## 2. The Behavioral Foundry (Feature Set)
We use Agnostic Signals to bridge the gap between sectors:
- **Economic/Transaction:** `monthly_fee_log`, `value_score_log`.
- **Friction/Service:** `support_intensity_log` (E-comm), `service_friction_score` (Aviation).
- **Loyalty/Nexus:** `loyalty_shock_score` (All sectors).
- **Engagement:** `usage_density` (Streaming), `engagement_efficiency` (Aviation).

---

## 3. Current Project Stage: [PHASE 2 - COMPARATIVE MODELING]

### ✅ PHASE 1: DATA PREPARATION (COMPLETED)
- **E-comm/Streaming:** Original dataset processed, imbalanced logic applied.
- **Aviation:** Authentic survey-based data integrated, demographics purged.
- **Cleaning:** All datasets are now "Log-Normalized" and "Surgically Purged."

### 🏗️ PHASE 2: MODELING (ACTIVE)
- **Task 1:** Train the HCIM Engine on the **Aviation** dataset (High feature richness).
- **Task 2:** Train the HCIM Engine on the **E-comm/Streaming** dataset (High imbalance).
- **Goal:** Compare SHAP values across sectors to see if "Friction" is a universal predictor.

---

## 4. Active Workspace Handshake
- **Target Variable:** `churn` (Binary 0/1).
- **Engine:** XGBoost (Extreme Gradient Boosting).
- **Validation:** SHAP summary plots are mandatory for every sector run.

---

## 5. HCIM Coding Instructions
- **Persona:** Senior AI Researcher / Data Architect.
- **Standards:** - No Demographics (Age/Gender).
    - Use Stratified 80/20 splits.
    - For E-comm/Streaming: Use `scale_pos_weight`.
    - For Aviation: Use standard weighting (Balanced data).