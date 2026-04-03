# Neral AI: Universal Hybrid Churn Intelligence (UHCI)
## [SOURCE OF TRUTH - PHASE 2 IGNITION]

---

## 1. Project Vision & Identity
- **Organization:** Neral AI
- **Product:** Universal Hybrid Churn Intelligence (UHCI).
- **Motto:** "Google Translate for Business Risk."
- **Strategic Target:** The Indian Quintet (Aviation, Retail, Logistics, SaaS, Warehousing).
- **Mission:** Building a Domain-Agnostic "Agnostic Brain" that identifies universal mathematical triggers of customer friction and loss.

---

## 2. Technical Architecture: The Behavioral Foundry
### A. The Agnostic Signal Layer
We convert raw industrial data into high-fidelity behavioral vectors:
- **Temporal Recency:** Standardized time since last value-exchange.
- **Intensity Decay:** Gradient of usage/activity drop.
- **Friction Accumulation:** Aggregated service/SLA failure metrics.

### B. Engineered Features (The Neral Edge)
Phase 1 has forged 32 High-Fidelity numerical signals. Key formulas:
- **Loyalty Shock Score:** $$Tenure\_Map \times Price\_Increase\_Flag$$
- **Support Intensity Log:** $$\ln(1 + Tickets\_Clipped \times Res\_Time)$$
- **Value Score Log:** $$\ln(1 + Usage\_Time / (Monthly\_Fee + 1))$$
- **Loyalty Resilience:** $$(NPS\_Normalized \times CSAT\_Score) / 5$$
- **Engagement Efficiency:** $$Marketing\_Clicks / (Email\_Opens + 0.01)$$

---

## 3. Current Project Stage: [PHASE 2 - MODELING]

### ✅ PHASE 1: THE FOUNDRY (COMPLETED)
- **Data State:** `df_clean` is prepared (40 columns).
- **Surgical Purge:** 15+ redundant features and all demographics (Age, Gender, Location) have been removed to prevent "Averaging Bias."
- **Synthesis:** 32 Numerical + 7 Categorical features are locked.

### 🏗️ PHASE 2: MODELING (ACTIVE)
- **Status:** Architecture defined. Ready for first training run.
- **Immediate Task:** 1. Perform **Stratified 80/20 Train-Test Split** on `df_clean`.
    2. Calculate `scale_pos_weight` based on the 9:1 imbalance ratio.
    3. Initialize and fit `XGBClassifier` with `eval_metric='aucpr'`.
- **Primary KPI:** **Recall (Class 1)**. We prioritize identifying churners over global accuracy.

### ⏳ PHASE 3: INTEGRATION (PENDING)
- **Stack:** FastAPI (Backend) + React.js (Frontend) + SHAP (Explainability Layer).

---

## 4. Active Workspace Handshake
- **Target Variable:** `churn` (Binary 0/1, 9:1 Imbalance).
- **Feature Set:** - `num_features`: 32 synthesized signals.
    - `cat_features`: 7 domain-agnostic categories.
- **Library Stack:** `xgboost`, `shap`, `sklearn`, `pandas`, `numpy`.

---

## 5. Neral AI Executive Instructions (Persona: Founder/Architect)
**You must adhere to these constraints for every code block generated:**

1. **Revenue-Focused Optimization:** Never optimize for raw Accuracy. Always focus on **F1-Score** and **Recall**. 
2. **The "Glass Box" Requirement:** No model training script is complete without a SHAP summary plot block. If the model isn't explainable, it isn't Neral AI.
3. **No Demographic Contamination:** If the code suggests using Age, Gender, or Race, it is a hard failure. We only model **Behavior**.
4. **Stratification is Mandatory:** Every split must be stratified to preserve the 9:1 churn ratio across sets.
5. **Production Code Standards:** Use Pydantic-style thinking—clean, modular, and ready for an asynchronous API.