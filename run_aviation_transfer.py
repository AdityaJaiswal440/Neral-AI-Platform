import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
import shap
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Aviation Hybrid Dataset...")
df = pd.read_csv('notebooks/Hybrid_Aviation_Churn_Integrated.csv')

print("2. Atomic Foundry: Engineering Behavioral Signals...")
# Construct csat_score from all survey features
service_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 
    'Inflight entertainment', 'On-board service', 'Leg room service', 
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]
if all(col in df.columns for col in service_cols):
    df['csat_score'] = df[service_cols].mean(axis=1)

# Aviation-specific loyalty shock score 
# If dataset lacks raw 'tenure', we trigger 'loyalty_shock' using the 
# compounding stress of delay_intensity intersecting with service_friction_score
if 'delay_intensity_log' in df.columns and 'service_friction_score' in df.columns:
    df['loyalty_shock_score'] = df['delay_intensity_log'] * df['service_friction_score']

X = df.drop('churn', axis=1)
y = df['churn']

print("3. Preprocessing Split & Transformation...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_exclude=np.number))
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
all_feat = preprocessor.get_feature_names_out()
features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]

X_train_df = pd.DataFrame(X_train_transformed, columns=features_clean)
X_test_df = pd.DataFrame(X_test_transformed, columns=features_clean)

print("4. Base Arena Sweep (Logistic Regression vs Random Forest)...")
lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr.fit(X_train_df, y_train)
lr_pred = lr.predict(X_test_df)

rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf.fit(X_train_df, y_train)
rf_pred = rf.predict(X_test_df)

print("5. Tuned XGBoost Transfer Learning (Domain: Aviation)...")
# E-commerce transfer weights applied to grid search
param_grid = {
    'scale_pos_weight': [5, 10, 15],  # testing limits for imbalance
    'max_depth': [3, 5, 7],           # 3 was champion in E-comm
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0]
}

xgb_base = XGBClassifier(eval_metric='aucpr', random_state=42)
random_search = RandomizedSearchCV(
    estimator=xgb_base, param_distributions=param_grid,
    n_iter=10, scoring='recall', cv=3, random_state=42, n_jobs=-1
)
random_search.fit(X_train_df, y_train)

best_xgb = random_search.best_estimator_
xgb_pred = best_xgb.predict(X_test_df)

print("Building Summary Comparison Matrix...")
def evaluate(y_true, y_pred, name):
    return {
        'Model Strategy': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall (Class 1)': recall_score(y_true, y_pred),
        'F1-Score (Class 1)': f1_score(y_true, y_pred)
    }

summary = [
    evaluate(y_test, lr_pred, 'Aviation Baseline: Logistic Regression'),
    evaluate(y_test, rf_pred, 'Aviation Baseline: Random Forest'),
    evaluate(y_test, xgb_pred, f"Aviation Tuned XGBoost")
]

summary_df = pd.DataFrame(summary)
summary_df.to_csv('Aviation_Sector_Summary.csv', index=False)
print("\n=== AVIATION SECTOR MODEL METRICS ===")
print(summary_df.to_string(index=False))

print(f"\nDiscovered optimal Aviation Tuning: {random_search.best_params_}")

print("\n6. SHAP Interpretability - Evaluating Dominant Vectors...")
# Calculate SHAP using sub-sample for speed if dataset is huge, otherwise exact tree explainer
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test_df)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("Aviation Sector: XGBoost SHAP Audit", fontsize=14, fontweight='bold', pad=20)
plt.savefig('notebooks/aviation_shap_summary.png', bbox_inches='tight', dpi=300)
plt.close()

print("Aviation Pipeline Evaluation Complete! SHAP output saved.")
