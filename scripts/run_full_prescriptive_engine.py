import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Loading Data and Model...")
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_test_reset = X_test.reset_index()

pipeline = joblib.load('models/hcim_E-comm_Stream_v1.joblib')
preprocessor = pipeline.named_steps['preprocessor']
xgb_model = pipeline.named_steps['classifier']

print("Extracting probabilities...")
probs = pipeline.predict_proba(X_test_reset.drop(columns=['index']))[:, 1]

print("Initializing SHAP Prescriptive Engine for FULL Dataframe...")
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

actions = []
total_discount_cost = 0.0

for idx in range(len(X_test_reset)):
    prob = probs[idx]
    customer_id = X_test_reset.iloc[idx]['index']
    
    person_shaps = shap_values.values[idx]
    abs_shaps = np.abs(person_shaps)
    top_2_indices = np.argsort(abs_shaps)[::-1][:2]
    top_features = [all_features_clean[i] for i in top_2_indices]
    
    action = f"Standard Review (Driver: {top_features[0]})"
    trigger_driver = top_features[0]
    
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

    # Extract actual monthly fee by executing Inverse Log
    monthly_fee = np.exp(X_test_reset.iloc[idx]['monthly_fee_log'])
    discount_amount = 0.0
    
    if action == "Retention Discount (Price Lock)":
        discount_amount = 0.20 * monthly_fee
        total_discount_cost += discount_amount

    actions.append({
        'customer_id': customer_id,
        'churn_prob': round(prob, 4),
        'risk_bin': get_risk_bin(prob),
        'primary_driver': trigger_driver,
        'secondary_driver': top_features[1] if trigger_driver == top_features[0] else top_features[0],
        'recommended_action': action,
        'current_monthly_fee_usd': round(monthly_fee, 2),
        'discount_cost_usd': round(discount_amount, 2)
    })

report_df = pd.DataFrame(actions)
report_df.to_csv('Full_Retention_Strategy_Report.csv', index=False)

print("\n--- Aggregate Prescriptive Analysis ---")
print(f"Total Users Processed: {len(X_test_reset)}")
num_price_lock = len(report_df[report_df['recommended_action'] == 'Retention Discount (Price Lock)'])
print(f"Total 'Price Lock' Targets: {num_price_lock}")
print(f"Total Immediate Cost of 20% Retention Discounts: ${total_discount_cost:,.2f} per month")
print("\nReport successfully saved to Full_Retention_Strategy_Report.csv")
