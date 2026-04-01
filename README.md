# ◈ Neral AI - Universal Hybrid Churn Intelligence (UHCI)

**An Open-Source Product from GSV Vadodara**

![React](https://img.shields.io/badge/Frontend-React-blue)
![Tailwind](https://img.shields.io/badge/Styling-Tailwind_CSS-38B2AC)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Hybrid Churn Intelligence Platform designed to solve the "Signal Fidelity Problem" and the "Action Gap" in enterprise customer retention. Built as a PRML course project at Gati Shakti Vishwavidyalaya, this repository serves as an open-source alternative to proprietary corporate churn-prediction solutions.

## 📖 The Problem

Current baseline churn models treat every customer equally and output raw mathematical probabilities that lack business context. This leads to two critical failures:
1. **Class Imbalance Blindness:** Missing a rare, high-value churning enterprise client is far more costly than falsely flagging a stable user.
2. **The Action Gap:** A 75% churn probability for a free-tier user requires a different intervention than a 75% risk for a $50k/year enterprise account. Raw probabilities do not tell Customer Success Managers (CSMs) *what* to do.

## 🚀 Our Solution

The Churn Project introduces a unified high-fidelity engine leveraging:
* **Cost-Sensitive Boosting:** An explicitly modified XGBoost objective function that penalizes false negatives, treating the minority class (churners) with proportionally higher importance (~8.79x weight).
* **Strategic Value Weighting:** A composite risk score equation that normalizes Customer Lifetime Value (CLV) alongside the churn probability to mathematically prioritize human intervention.
* **CSM Intervention Dashboard:** A live React-based frontend that translates ML outputs into prioritized, plain-language retention actions.

## 🗄️ Dataset & Architecture

* **Data Provenance:** Kaggle Telco Churn Repository (Simulated 10k enterprise & SME records).
* **Feature Synthesis:** Includes custom metrics like Support Intensity Logs, Loyalty Shock Scores, and Ghost Signal Pruning.
* **Evaluation Metric:** Optimized strictly for **PR-AUC** rather than standard accuracy.

### Tech Stack
* **Machine Learning:** Python, XGBoost, Optuna (Hyperparameter Tuning), SHAP (Explainability)
* **Backend:** FastAPI, Pydantic
* **Frontend:** React, Tailwind CSS, Lucide Icons
* **Deployment:** Docker (Containerized for seamless local deployment)

## 💻 Getting Started (Local Development)

The project is split into two environments: the React frontend and the FastAPI backend.

### 1. Running the Frontend
```bash
cd frontend
npm install
npm run dev
