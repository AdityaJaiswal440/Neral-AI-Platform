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
    """Replicates training logic with explicit existence guarantees."""
    if sector == 'aviation':
        # Helper to get scalar values safely and fix FutureWarnings
        def f_val(key, default=3):
            val = df.get(key, df.get(key.replace('_', ' '), default))
            if isinstance(val, pd.Series):
                return val.iloc[0] if not val.empty else default
            return val if val is not None else default

        # 1. Log Transformations
        df['distance_log'] = np.log1p(float(f_val('Flight_Distance', 0)))
        df['delay_intensity_log'] = np.log1p(float(f_val('Departure_Delay_Minutes', 0)) + float(f_val('Arrival_Delay_Minutes', 0)))

        # 2. Categorical Flags
        df['is_business_travel'] = 1 if str(f_val('Type_of_Travel')) == 'Business travel' else 0
        df['is_disloyal_customer'] = 1 if str(f_val('Customer_Type')) == 'disloyal Customer' else 0

        # 3. Score Logic & Column Existence Guarantee
        # These are the "Raw" features the model expects (with spaces eventually)
        core_ratings = {
            'Inflight_wifi_service': 'Inflight wifi service',
            'Online_boarding': 'Online boarding',
            'Seat_comfort': 'Seat comfort',
            'Inflight_entertainment': 'Inflight entertainment'
        }
        
        # Ensure they exist in df as the model expects them
        for json_key, model_key in core_ratings.items():
            df[model_key] = f_val(json_key)

        # Calculate CSAT using the now-guaranteed columns
        df['csat_score'] = df[list(core_ratings.values())].mean(axis=1)
        df['loyalty_shock_score'] = df['csat_score'] * df['is_disloyal_customer']
        df['service_friction_score'] = (5 - df['Inflight wifi service']) + (5 - df['Online boarding'])
        
        # 4. Standard features mapping (ensuring hyphen and space consistency)
        standard_mapping = {
            'Inflight_service': 'Inflight service',
            'Food_and_drink': 'Food and drink',
            'Baggage_handling': 'Baggage handling',
            'Checkin_service': 'Checkin service',
            'On_board_service': 'On-board service',
            'Cleanliness': 'Cleanliness',
            'Gate_location': 'Gate location',
            'Leg_room_service': 'Leg room service',
            'Ease_of_Online_booking': 'Ease of Online booking',
            'Departure/Arrival_time_convenient': 'Departure/Arrival time convenient'
        }
        for json_key, model_key in standard_mapping.items():
            df[model_key] = f_val(json_key)
                
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
@app.post("/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    df_raw = pd.DataFrame([payload.features])
    
    # 1. Engineering (This now guarantees all 18+ columns exist with correct names)
    df_ready = apply_feature_engineering(df_raw, sector)
    
    # 2. Inference
    processed_data = PREPROCESSORS[sector].transform(df_ready)
    prob = MODELS[sector].predict_proba(processed_data)[0, 1]
    
    # 3. Diagnosis
    top_driver = extract_top_driver(sector, df_ready)
    
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