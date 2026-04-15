import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier
import shap
import json
import warnings

warnings.filterwarnings('ignore')

# 1. Setup Data
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

# Precompute transformation for faster CV
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

all_features = preprocessor.get_feature_names_out()
all_features_clean = [f.split('__')[1] if '__' in f else f for f in all_features]

X_train_df = pd.DataFrame(X_train_transformed, columns=all_features_clean)
X_test_df = pd.DataFrame(X_test_transformed, columns=all_features_clean)

# 2. Hyperparameter Tuning on XGBoost
print("Starting RandomizedSearchCV for XGBoost...")
param_grid = {
    'scale_pos_weight': range(5, 13),
    'max_depth': [2, 3, 5, 7, 9],
    'learning_rate': [0.01, 0.03, 0.04, 0.05, 0.07, 0.1],
    'subsample': [0.7, 0.8, 0.9]
}

xgb_base = XGBClassifier(n_estimators=200, eval_metric='aucpr', random_state=42)

# Fast random search prioritizing F1/Recall
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=15, 
    scoring='recall',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_df, y_train)
best_xgb = random_search.best_estimator_
print(f"Best XGB Params: {random_search.best_params_}")

# 3. Threshold Optimization
print("Optimizing Threshold for >= 75% Recall...")
y_pred_proba_xgb = best_xgb.predict_proba(X_test_df)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_xgb)

# Find optimal threshold where recall is at least 0.75
optimal_idx = np.where(recalls >= 0.75)[0][-1]
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
print(f"Optimal Threshold for >=75% Recall: {optimal_threshold}")

# Plot Precision-Recall vs Threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
plt.axvline(x=optimal_threshold, color='r', linestyle=':', label=f'Optimal Thresh: {optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.title('Precision & Recall vs Threshold (Tuned XGBoost)')
plt.grid(True)
plt.savefig('notebooks/pr_threshold_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# Generate predictions using optimal threshold
y_pred_xgb_opt = (y_pred_proba_xgb >= optimal_threshold).astype(int)
y_pred_xgb_default = best_xgb.predict(X_test_df)

# 4. Hybrid Ensemble (Stacking)
print("Training Stacking Ensemble...")
lr_balanced = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)

estimators = [
    ('xgb', best_xgb),
    ('lr', lr_balanced)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

stack_model.fit(X_train_df, y_train)
y_pred_stack = stack_model.predict(X_test_df)

# Evaluation Summary Table
summary = []

def get_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True)
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Recall (Class 1)': rep['1']['recall'],
        'F1-Score (Class 1)': rep['1']['f1-score']
    }

summary.append(get_metrics(y_test, y_pred_xgb_default, "Tuned XGBoost (Thresh 0.5)"))
summary.append(get_metrics(y_test, y_pred_xgb_opt, f"Tuned XGBoost (Thresh {optimal_threshold:.2f})"))
summary.append(get_metrics(y_test, y_pred_stack, "Stacking Ensemble (XGB + LR)"))

summary_df = pd.DataFrame(summary)
print("\n=== FINAL EVALUATION SUMMARY ===")
print(summary_df.to_string(index=False))
summary_df.to_csv('notebooks/ensemble_summary.csv', index=False)

# 5. SHAP Re-Audit on Ensemble
print("Running SHAP KernelExplainer on Stacking Ensemble...")
# Use predicting probabilities of class 1
def ensemble_predict(X):
    # KernelExplainer expects numpy arrays usually, ensure wrapping safely
    return stack_model.predict_proba(pd.DataFrame(X, columns=all_features_clean))[:, 1]

# Downsample heavily for KernelExplainer speed
X_train_summary = shap.kmeans(X_train_df, 20)
X_test_sample = X_test_df.sample(n=100, random_state=42)

explainer = shap.KernelExplainer(ensemble_predict, X_train_summary)
shap_values = explainer.shap_values(X_test_sample, nsamples=100)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("SHAP Summary Plot: Stacking Ensemble Decisions", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('notebooks/ensemble_shap_summary.png', bbox_inches='tight', dpi=300)
plt.close()

print("Execution fully complete.")
