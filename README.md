# Neral-AI-Platform
Neral-AI: The Hybrid Churn Intelligence Platform [v6.1]

🏗️ Distributed Production Architecture

Neral-AI is engineered as a decoupled, cloud-distributed intelligence system. By separating the Inference Brain (FastAPI) from the Spotlight UI (Streamlit), we achieve industrial-grade scalability and low-latency diagnostic output.

Figure 1: End-to-end MLOps Pipeline showing the Secured REST Gateway and Distributed Inference Loop.

🎯 The "Why": Bridging the Explainability-Action Gap (EAG)

Most industry-standard churn models are "Black Boxes." They successfully predict what will happen (Global Accuracy) but fail to explain why it is happening for a specific user (Local Actionability). This creates the Explainability-Action Gap (EAG).

Neral-AI bridges this gap by transforming raw risk probabilities into Atomic Force Logs. We move beyond predicting loss; we provide the exact behavioral drivers required to execute retention maneuvers.

🧠 The Neral Core: Mathematical Objective

The system is built upon a mathematically harmonized Extreme Gradient Boosting (XGBoost) objective function. The core is tuned to treat churn as a harmonically terminated system, smoothing risk oscillations through L1/L2 regularization.

The objective function minimized during the Foundry training phase is defined as:

$$Obj(\theta) = \sum_{i} l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$

Where the complexity term $\Omega(f)$ is controlled by the Neral-AI hyperparameter forge:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

🛠️ Hyperparameter Calibration [v6.1]

Parameter

Value

Functional Role

Learning Rate ($\eta$)

0.01

Prevents overshooting in high-volatility behavior spikes.

Max Depth

6

Balances model complexity with EAG actionability.

Gamma ($\gamma$)

0.1

Prunes branches to "short-circuit" feature noise.

Scale_pos_weight

4.5

Chemically aligned to correct 9:1 class imbalance.

⚡ Atomic SHAP Alignment

To provide local explainability, Neral-AI utilizes an optimized KernelSHAP alignment. This converts model-agnostic risk into directional forces:

$$\phi_j = \sum_{S \subseteq \{x_1, \dots, x_p\} \setminus \{x_j\}} \frac{|S|!(p - |S| - 1)!}{p!} [f(S \cup \{x_j\}) - f(S)]$$

This allows the UI to display exactly which feature (e.g., Wifi Service or Usage Density) is acting as the primary churn driver.

📊 Validated Performance Metrics

Results verified through 5-Fold Stratified Cross-Validation ensuring sector-neutrality across diverse behavioral foundries.

Core Engine

Sector

F1-Macro

Focus Area

Nuclear Core

E-Commerce

72.8%

Transactional Volatility

VoxSnake Core

Aviation

85.0%

Service Satisfaction & High-Value Retention

🚀 Deployment & MLOps

Backend: FastAPI microservice hosted on Hugging Face Spaces.

Frontend: Streamlit UI consumed via a secured REST Gateway (x-api-key).

Pipeline: Automated behavioral data preprocessing and telemetry monitoring.

👤 Developer & Contact

Aditya Jaiswal Lead Architect & Founder, Neral-AI GSV AIML | Roll: 24AI007

LinkedIn | Live Demo | Research Paper

Disclaimer: This repository serves as a technical portfolio for the Neral-AI project. Proprietary weights and private datasets are omitted to protect IP.
