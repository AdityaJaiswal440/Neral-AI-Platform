import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Live Model and Dataset...")
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')
X = df.drop('churn', axis=1)
y = df['churn']

# Ensure exact same split logic as originally established
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_test_reset = X_test.reset_index()

pipeline = joblib.load('models/hcim_E-comm_Stream_v1.joblib')
preprocessor = pipeline.named_steps['preprocessor']
xgb_model = pipeline.named_steps['classifier']

print("2. Mapping Probabilities for X_test...")
probs = pipeline.predict_proba(X_test_reset.drop(columns=['index']))[:, 1]
top_100_idx = np.argsort(probs)[::-1][:100]

print("3. Connecting SHAP Glass Box Engine...")
X_test_transformed = preprocessor.transform(X_test_reset.drop(columns=['index']))
all_features = preprocessor.get_feature_names_out()
all_features_clean = [f.split('__')[1] if '__' in f else f for f in all_features]
X_test_df_clean = pd.DataFrame(X_test_transformed, columns=all_features_clean)

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_df_clean)

def get_risk_bin(prob):
    if prob < 0.3: return "Low Risk"
    elif prob <= 0.7: return "Medium Risk"
    else: return "High Risk"

def get_business_action(index):
    prob = probs[index]
    
    # Original Customer ID
    customer_id = X_test_reset.iloc[index]['index']
    
    # Assess SHAP weightings
    person_shaps = shap_values.values[index]
    abs_shaps = np.abs(person_shaps)
    top_2_indices = np.argsort(abs_shaps)[::-1][:2]
    top_features = [all_features_clean[i] for i in top_2_indices]
    
    action = f"Standard Review (Driver: {top_features[0]})"
    trigger_driver = top_features[0]
    
    # Assess over top 2 contributors to allow fallback mappings
    for driver in top_features:
        if driver == 'loyalty_shock_score':
            action = "Retention Discount (Price Lock)"
            trigger_driver = driver
            break
        elif driver == 'usage_density':
            action = "Engagement Nudge (Onboarding)"
            trigger_driver = driver
            break
        elif driver == 'csat_score':
            action = "Service Recovery (VIP Ticket)"
            trigger_driver = driver
            break

    return {
        'customer_id': customer_id,
        'churn_prob': round(prob, 4),
        'risk_bin': get_risk_bin(prob),
        'primary_driver': trigger_driver,
        'secondary_driver': top_features[1] if trigger_driver == top_features[0] else top_features[0],
        'recommended_action': action
    }

print("4. Executing Batch Analysis for Top 100 Highest Risk users...")
actions = []
for idx in top_100_idx:
    actions.append(get_business_action(idx))

report_df = pd.DataFrame(actions)
report_df.to_csv('Prescriptive_Action_Report.csv', index=False)

print("\n--- Prescriptive Action Report Successfully Generated ---")
print(report_df.head(10).to_string(index=False))
