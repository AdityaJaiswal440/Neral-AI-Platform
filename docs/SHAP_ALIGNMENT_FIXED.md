# Technical Whitepaper: Solving Inference Alignment & Feature Drift in SHAP Explanations

## 1. Executive Summary
During the development of the **Neral AI Core**, a critical "Identity Crisis" was observed in the inference engine. The model (XGBoost) was predicting high churn probabilities correctly, but the diagnostic tool (SHAP) was attributing the risk to positive features (e.g., 5-star CSAT scores). This paper outlines the forensic analysis and the engineering of the **v6.1 Ground Truth Alignment** to resolve feature-index displacement.

## 2. The Problem Statement: "The CSAT Paradox"
While testing high-friction scenarios (Monthly Charges > $900), the system correctly identified high churn risk (0.605) but flagged `CSAT_SCORE` as the primary driver.
- **Symptom:** Logical inversion of feature impact.
- **Root Cause:** A combination of **Version Drift** (Scikit-learn 1.5.2 vs 1.7.2) and **Index Displacement** during the `ColumnTransformer` stage.



## 3. Root Cause Analysis (RCA)

### A. The Version Drift artifact
Scikit-learn 1.5.2 handled `OneHotEncoder` category ordering differently than the 1.7.2 training environment. This caused a silent "Column Shift" in the resulting NumPy array.
### B. Key-Casing Drift
Input JSON keys from the UI (e.g., `city`) did not strictly match the model’s internal feature names (e.g., `City`), causing the preprocessor to treat critical data as "Missing," leading to the "Dhaka Shadow" (defaulting to the first available category index).
### C. The Mapping Lens
SHAP was reading the raw NumPy array without an explicit feature-name handshake, causing it to assign the name tag of Index 0 (`CSAT_SCORE`) to the value belonging to Index 0's actual data (`Monthly_Charges`).

## 4. The Solution: Neral AI Ground Truth v6.1

We implemented a three-tier defensive architecture:

### Step 1: The Nuclear Restart (Version Locking)
We injected a runtime jailbreak in the startup script to force a local upgrade to `scikit-learn==1.7.2`, ensuring the inference environment perfectly mirrors the training environment.

### Step 2: The Case-Sensitive Handshake
We refactored the ingestion engine to perform a case-insensitive lookup against the model’s `feature_names_in_`.
```python
ordered_features = {key: payload.get(key.lower(), 0) for key in expected_keys}