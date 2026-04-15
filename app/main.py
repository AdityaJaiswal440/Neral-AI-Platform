import os
import joblib
import pandas as pd
import numpy as np
import shap
import uuid
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# 1. Configuration & Security
API_KEY = os.getenv("NERAL_SECRET", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Neral AI: Hybrid Churn Intelligence", version="1.0")

# 2. Security Layer
def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Invalid or Missing API Key")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Global Registry
MODELS = {}
EXPLAINERS = {}
PREPROCESSORS = {}

def apply_feature_engineering(df: pd.DataFrame, sector: str):
    """Replicates training logic with fallback for both underscores and spaces."""
    if sector == 'aviation':
        # Helper to get values regardless of naming convention
        def f(key): return df.get(key, df.get(key.replace('_', ' '), 3))

        # 1. Log Transformations
        df['distance_log'] = np.log1p(float(f('Flight_Distance')))
        df['delay_intensity_log'] = np.log1p(float(f('Departure_Delay_Minutes')) + float(f('Arrival_Delay_Minutes')))

        # 2. Categorical Flags
        df['is_business_travel'] = 1 if str(f('Type_of_Travel')) == 'Business travel' else 0
        df['is_disloyal_customer'] = 1 if str(f('Customer_Type')) == 'disloyal Customer' else 0

        # 3. Score Logic
        rating_cols = ['Inflight_wifi_service', 'Online_boarding', 'Seat_comfort', 'Inflight_entertainment']
        ratings = [float(f(c)) for c in rating_cols]
        df['csat_score'] = sum(ratings) / len(ratings)
        df['loyalty_shock_score'] = df['csat_score'] * df['is_disloyal_customer']
        df['service_friction_score'] = (5 - float(f('Inflight_wifi_service'))) + (5 - float(f('Online_boarding')))
        
        # 4. Fill missing raw columns that the model expects
        standard_mapping = {
            'Inflight_service': 'Inflight service',
            'Food_and_drink': 'Food and drink',
            'Baggage_handling': 'Baggage handling',
            'Checkin_service': 'Checkin service',
            'On_board_service': 'On-board service', # Note the hyphen!
            'Cleanliness': 'Cleanliness',
            'Gate_location': 'Gate location',
            'Leg_room_service': 'Leg room service',
            'Ease_of_Online_booking': 'Ease of Online booking',
            'Departure/Arrival_time_convenient': 'Departure/Arrival time convenient'
        }
        for json_key, model_key in standard_mapping.items():
            if model_key not in df.columns:
                df[model_key] = f(json_key)
                
    return df

@app.on_event("startup")
def load_models():
    sectors = {'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib', 'aviation': 'models/hcim_Aviation_v1.joblib'}
    for sector, path in sectors.items():
        if os.path.exists(path):
            pipeline = joblib.load(path)
            MODELS[sector] = pipeline.named_steps['classifier']
            PREPROCESSORS[sector] = pipeline.named_steps['preprocessor']
            EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
            print(f"Loaded {sector} logic...")
    print("Unified Core Online. 16GB Memory Engine Engaged.")

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

def extract_top_driver(sector: str, df_final: pd.DataFrame) -> str:
    transformed = PREPROCESSORS[sector].transform(df_final)
    all_feat = PREPROCESSORS[sector].get_feature_names_out()
    features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
    df_shap = pd.DataFrame(transformed, columns=features_clean)
    shap_vals = EXPLAINERS[sector](df_shap)
    top_idx = np.argmax(np.abs(shap_vals.values[0]))
    return features_clean[top_idx]

@app.post("/v1/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    df_raw = pd.DataFrame([payload.features])
    
    # 1. Engineering (Creates columns with underscores like loyalty_shock_score)
    df_engineered = apply_feature_engineering(df_raw, sector)
    
    # 2. SURGICAL RENAME: Only rename the raw features, NOT the engineered ones
    rename_map = {
        'Inflight_wifi_service': 'Inflight wifi service',
        'Online_boarding': 'Online boarding',
        'Inflight_entertainment': 'Inflight entertainment',
        'Seat_comfort': 'Seat comfort'
    }
    df_engineered.rename(columns=rename_map, inplace=True)
    
    # 3. Inference
    processed_data = PREPROCESSORS[sector].transform(df_engineered)
    prob = MODELS[sector].predict_proba(processed_data)[0, 1]
    
    # 4. Diagnosis
    top_driver = extract_top_driver(sector, df_engineered)
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": round(float(prob), 4),
        "trigger_diagnosis": top_driver,
        "prescriptive_rescue": "Executive Concierge" if prob > 0.5 else "Standard Review",
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)