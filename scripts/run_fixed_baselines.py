import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json

# Load Data
df = pd.read_csv('notebooks/HCIM_Current_State_Fixed.csv')

# Explicitly ensure target 'churn' is not in our feature selectors
X = df.drop('churn', axis=1)
y = df['churn']

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing Pipeline (Scale Numerical, One-Hot Encode Categorical)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_exclude=np.number))
    ]
)

# Model Dictionary
models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "RandomForestClassifier": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100, random_state=42)
}

# Execution Loop
summary = []
pipelines = {}
reports = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    pipelines[name] = pipeline
    
    y_test_pred = pipeline.predict(X_test)
    
    reports[name] = classification_report(y_test, y_test_pred)
    
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    summary.append({
        'Model Name': name, 
        'Test Accuracy': accuracy_score(y_test, y_test_pred), 
        'Recall (Class 1)': report_dict['1']['recall'], 
        'F1-Score (Class 1)': report_dict['1']['f1-score']
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('notebooks/baseline_summary.csv', index=False)

# Feature Importance
rf_model = pipelines["RandomForestClassifier"].named_steps['classifier']
# Get feature names back dynamically
num_cols = preprocessor.transformers_[0][2] # the array of numeric column names is not here because we used selector
# actually for make_column_selector we can just fit_transform and get columns
# the easiest way in scikit_learn >= 1.0 is get_feature_names_out()
all_features = preprocessor.get_feature_names_out()
# The names look like num__loyalty_shock_score or cat__city_London
all_features_clean = [f.split('__')[1] for f in all_features]

fi_df = pd.DataFrame({'Feature': all_features_clean, 'Importance': rf_model.feature_importances_})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# Plot
colors = ['red' if feat == 'loyalty_shock_score' else 'steelblue' for feat in fi_df.head(20)['Feature']]

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), palette=colors)
plt.title('Top 20 Feature Importances (Random Forest)\nHighlighting loyalty_shock_score', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('notebooks/rf_feature_importance.png')

# Save outputs to JSON
output_data = {
    "summary": summary,
    "reports": reports
}
with open('notebooks/baseline_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Execution complete.")
