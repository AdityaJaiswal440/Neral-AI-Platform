import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import shap
import json
import warnings

warnings.filterwarnings('ignore')

# 1. Setup Data
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')
X = df.drop('churn', axis=1)
y = df['churn']

# 2. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_exclude=np.number))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Transform data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Features Extraction
all_features = preprocessor.get_feature_names_out()
all_features_clean = [f.split('__')[1] if '__' in f else f for f in all_features]

X_train_df = pd.DataFrame(X_train_transformed, columns=all_features_clean)
X_test_df = pd.DataFrame(X_test_transformed, columns=all_features_clean)

# 3. Model
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=8.79,
    eval_metric='aucpr',
    random_state=42
)

# 4. Training
print("Training XGBoost Engine...")
xgb.fit(X_train_df, y_train)

print("Evaluating...")
y_pred = xgb.predict(X_test_df)
report = classification_report(y_test, y_pred)
out_dict = classification_report(y_test, y_pred, output_dict=True)

# Save string report
result_data = {
    "report": report,
    "recall": out_dict['1']['recall'],
    "f1": out_dict['1']['f1-score'],
    "accuracy": accuracy_score(y_test, y_pred)
}

with open('notebooks/xgb_results.json', 'w') as f:
    json.dump(result_data, f, indent=2)
    
print("Generating SHAP Plot...")
# 5. SHAP Explainer
explainer = shap.TreeExplainer(xgb)
try:
    # SHAP returns different things depending on xgboost version, often just a single array for binary
    shap_values = explainer(X_test_df)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
except Exception as e:
    print(f"SHAP Error with object api: {str(e)}.. falling back to legacy")
    shap_values = explainer.shap_values(X_test_df)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)

plt.title("SHAP Summary Plot: XGBoost Decisions", fontsize=14, fontweight='bold', pad=20)
plt.savefig('notebooks/xgb_shap_summary.png', bbox_inches='tight', dpi=300)
print("Saved SHAP plot to notebooks/xgb_shap_summary.png")
