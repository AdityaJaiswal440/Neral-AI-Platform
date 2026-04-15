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

def apply_feature_engineering(df_clean: pd.DataFrame, sector: str):
    """Replicates training logic using keys from the incoming JSON (underscores)."""
    if sector == 'aviation':
        # 1. Log Transformations
        df_clean['distance_log'] = np.log1p(df_clean.get('Flight_Distance', 0))
        dep_delay = df_clean.get('Departure_Delay_Minutes', 0)
        arr_delay = df_clean.get('Arrival_Delay_Minutes', 0)
        df_clean['delay_intensity_log'] = np.log1p(dep_delay + arr_delay)

        # 2. Categorical Encoding (Fixed the Boolean .astype error)
        # We use .eq() to ensure we compare the Series, not a scalar None
        df_clean['is_business_travel'] = df_clean.get('Type_of_Travel', pd.Series([""])).eq('Business travel').astype(int)
        df_clean['is_disloyal_customer'] = df_clean.get('Customer_Type', pd.Series([""])).eq('disloyal Customer').astype(int)

        # 3. Score Logic (Using underscores to match JSON)
        rating_cols = ['Inflight_wifi_service', 'Online_boarding', 'Seat_comfort', 'Inflight_entertainment']
        for col in rating_cols:
            if col not in df_clean.columns: df_clean[col] = 3
            
        df_clean['csat_score'] = df_clean[rating_cols].mean(axis=1)
        df_clean['loyalty_shock_score'] = df_clean['csat_score'] * df_clean['is_disloyal_customer']
        
        wifi = df_clean.get('Inflight_wifi_service', 3)
        boarding = df_clean.get('Online_boarding', 3)
        df_clean['service_friction_score'] = (5 - wifi) + (5 - boarding)
        
        # 4. Final consistency check for standard columns
        standard_cols = [
            'Inflight_service', 'Food_and_drink', 'Baggage_handling', 
            'Checkin_service', 'On_board_service', 'Cleanliness', 
            'Gate_location', 'Leg_room_service', 'Ease_of_Online_booking', 
            'Departure/Arrival_time_convenient'
        ]
        for col in standard_cols:
            if col not in df_clean.columns: df_clean[col] = 3
                
    return df_clean

@app.on_event("startup")
def load_models():
    sectors = {
        'ecommerce': 'models/hcim_E-comm_Stream_v1.joblib',
        'aviation': 'models/hcim_Aviation_v1.joblib'
    }
    
    for sector, path in sectors.items():
        if os.path.exists(path):
            pipeline = joblib.load(path)
            MODELS[sector] = pipeline.named_steps['classifier']
            PREPROCESSORS[sector] = pipeline.named_steps['preprocessor']
            EXPLAINERS[sector] = shap.Explainer(MODELS[sector])
            print(f"Loaded {sector} logic...")
        else:
            print(f"WARNING: Model not found at {path}")
    print("Unified Core Online. 16GB Memory Engine Engaged.")

class PredictPayload(BaseModel):
    sector: str
    features: Dict[str, Any]

def extract_top_driver(sector: str, df_engineered: pd.DataFrame) -> str:
    """Uses the final renamed data to generate SHAP explanations."""
    transformed = PREPROCESSORS[sector].transform(df_engineered)
    all_feat = PREPROCESSORS[sector].get_feature_names_out()
    features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
    df_final = pd.DataFrame(transformed, columns=features_clean)
    
    shap_vals = EXPLAINERS[sector](df_final)
    abs_shaps = np.abs(shap_vals.values[0])
    top_idx = np.argmax(abs_shaps)
    return features_clean[top_idx]

@app.post("/v1/predict")
@app.post("/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail="Valid sectors: 'ecommerce', 'aviation'")
    
    # 1. Create DataFrame from raw features (keeps underscores)
    df_raw = pd.DataFrame([payload.features])
    
    # 2. STEP 1: Feature Engineering (Finds underscores)
    df_engineered = apply_feature_engineering(df_raw, sector)
    
    # 3. STEP 2: Standardize names for the Model/Transformer (Underscore -> Space)
    # This ensures "Inflight_wifi_service" becomes "Inflight wifi service"
    df_engineered.columns = [c.replace('_', ' ') for c in df_engineered.columns]
    
    # 4. Inference
    prob = MODELS[sector].predict_proba(PREPROCESSORS[sector].transform(df_engineered))[0, 1]
    
    # 5. Interpretability
    top_driver = extract_top_driver(sector, df_engineered)
    
    # 6. Prescriptive Logic
    rescue_action = "Standard Review"
    driver_low = top_driver.lower()
    
    if sector == 'ecommerce':
        if 'loyalty' in driver_low: rescue_action = "Retention Discount (Price Lock) - 20%"
        elif 'usage' in driver_low: rescue_action = "Engagement Nudge (Onboarding Specialist)"
        elif 'csat' in driver_low: rescue_action = "Service Recovery (VIP Ticket)"
    elif sector == 'aviation':
        if 'wifi' in driver_low or 'boarding' in driver_low:
            rescue_action = "Executive Concierge (Digital Friction Override)"
        elif 'delay' in driver_low:
            rescue_action = "Flight Comp (Lounge Access / Upgrade)"
        elif 'seat' in driver_low or 'food' in driver_low:
            rescue_action = "Amenity Upgrade (Premium Comp)"
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": round(float(prob), 4),
        "trigger_diagnosis": top_driver,
        "prescriptive_rescue": rescue_action,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)