import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from xgboost import XGBClassifier
import shap
import json
import warnings

warnings.filterwarnings('ignore')

print("Loading and preparing data...")
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')
X = df.drop('churn', axis=1)
y = df['churn']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_exclude=np.number))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

all_features = preprocessor.get_feature_names_out()
all_features_clean = [f.split('__')[1] if '__' in f else f for f in all_features]

X_train_df = pd.DataFrame(X_train_transformed, columns=all_features_clean)
X_test_df = pd.DataFrame(X_test_transformed, columns=all_features_clean)

print("Training best XGBoost model...")
best_xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.01, max_depth=3, scale_pos_weight=10, subsample=0.7, eval_metric='aucpr', random_state=42
)
best_xgb.fit(X_train_df, y_train)

print("Initializing SHAP Explainer...")
# Using explainer directly to get Explanation object natively compatible with waterfall
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test_df)

print("Generating Global Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.savefig('notebooks/shap_global_summary.png', bbox_inches='tight', dpi=300)
plt.close()

print("Calculating Probabilities for Case Study...")
probs = best_xgb.predict_proba(X_test_df)[:, 1]
high_risk_idx = int(np.argmax(probs))
prob_val = probs[high_risk_idx]
print(f"Customer {high_risk_idx} found with probability {prob_val*100:.2f}%")

def explain_customer(index):
    # Sub-select explanation object
    exp = shap_values[index]
    
    # Generate waterfall
    plt.figure(figsize=(10, 8))
    # SHAP explainer from XGBClassifier natively provides 1D logic for binary
    shap.waterfall_plot(exp, show=False)
    plt.suptitle(f"Local Audit - Customer {index} (Churn Prob: {prob_val*100:.1f}%)", fontweight='bold')
    plt.savefig('notebooks/shap_local_waterfall.png', bbox_inches='tight', dpi=300)
    plt.close()

print("Generating Local Waterfall Plot...")
explain_customer(high_risk_idx)

# Extract values for markdown string
feat = X_test_df.iloc[high_risk_idx].to_dict()
sv = shap_values.values[high_risk_idx]
shaps = dict(zip(all_features_clean, sv))

audit_info = {
    'index': high_risk_idx,
    'prob': prob_val,
    'loyalty_shock_score_val': feat.get('loyalty_shock_score', 0.0),
    'loyalty_shock_score_shap': shaps.get('loyalty_shock_score', 0.0),
    'usage_density_val': feat.get('usage_density', 0.0),
    'usage_density_shap': shaps.get('usage_density', 0.0)
}

with open('notebooks/shap_audit.json', 'w') as f:
    json.dump(audit_info, f, indent=2)

print("SHAP script execution complete.")
