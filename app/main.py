import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
import shap
import uuid
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="HCIM Unified Inference API", version="1.0")

# Security Layer Initialization
API_KEY = os.getenv("API_KEY", "NERAL_SECRET_2026")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Forbidden: Invalid or Missing API Key")

# CORS Middleware Layer
# To restrict payload to our secure Front-end URL later, change allow_origins=["*"] to allow_origins=["https://neral-ai.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Models Dictionary
MODELS = {}
EXPLAINERS = {}
PREPROCESSORS = {}

@app.on_event("startup")
def load_models():
    # Environment-agnostic model paths for Docker portability
    ecomm_path = os.getenv("MODEL_PATH_ECOMM", "models/hcim_E-comm_Stream_v1.joblib")
    aviation_path = os.getenv("MODEL_PATH_AVIATION", "models/hcim_Aviation_v1.joblib")
    
    ecomm_pipeline = joblib.load(ecomm_path)
    MODELS['ecommerce'] = ecomm_pipeline.named_steps['classifier']
    PREPROCESSORS['ecommerce'] = ecomm_pipeline.named_steps['preprocessor']
    EXPLAINERS['ecommerce'] = shap.Explainer(MODELS['ecommerce'])
    
    aviation_pipeline = joblib.load(aviation_path)
    MODELS['aviation'] = aviation_pipeline.named_steps['classifier']
    PREPROCESSORS['aviation'] = aviation_pipeline.named_steps['preprocessor']
    EXPLAINERS['aviation'] = shap.Explainer(MODELS['aviation'])
    
    print("Unified Core Online. Models locked in memory.")

class PredictPayload(BaseModel):
    sector: str
    data: Dict[str, Any]

def extract_top_driver(sector: str, df_raw: pd.DataFrame) -> str:
    transformed = PREPROCESSORS[sector].transform(df_raw)
    all_feat = PREPROCESSORS[sector].get_feature_names_out()
    features_clean = [f.split('__')[1] if '__' in f else f for f in all_feat]
    df_clean = pd.DataFrame(transformed, columns=features_clean)
    
    shap_vals = EXPLAINERS[sector](df_clean)
    abs_shaps = np.abs(shap_vals.values[0])
    top_idx = np.argmax(abs_shaps)
    return features_clean[top_idx]

# Endpoint secured behind get_api_key validation layer
@app.post("/predict")
def predict(payload: PredictPayload, api_key: str = Security(get_api_key)):
    sector = payload.sector.lower()
    if sector not in MODELS:
        raise HTTPException(status_code=400, detail="Sector must be 'ecommerce' or 'aviation'")
    
    raw_dict = payload.data
    df_raw = pd.DataFrame([raw_dict])
    
    if sector == 'aviation':
        df_raw.columns = [c.replace('_', ' ') if c in ['Inflight_wifi_service', 'Online_boarding', 'Inflight_entertainment', 'Seat_comfort', 'Food_and_drink', 'On-board_service', 'Leg_room_service', 'Baggage_handling', 'Checkin_service', 'Inflight_service', 'Gate_location', 'Departure/Arrival_time_convenient', 'Ease_of_Online_booking'] else c for c in df_raw.columns]
    
    pipeline = Pipeline(steps=[('preprocessor', PREPROCESSORS[sector]), ('classifier', MODELS[sector])])
    prob = pipeline.predict_proba(df_raw)[0, 1]
    
    top_driver = extract_top_driver(sector, df_raw)
    rescue_action = "Standard Review"
    
    if sector == 'ecommerce':
        if 'loyalty_shock' in top_driver.lower():
            rescue_action = "Retention Discount (Price Lock) - 20%"
        elif 'usage_density' in top_driver.lower():
            rescue_action = "Engagement Nudge (Onboarding Specialist)"
        elif 'csat' in top_driver.lower():
            rescue_action = "Service Recovery (VIP Ticket)"
            
    elif sector == 'aviation':
        if 'wifi' in top_driver.lower() or 'boarding' in top_driver.lower():
            rescue_action = "Executive Concierge (Digital Friction Override)"
        elif 'delay' in top_driver.lower():
            rescue_action = "Flight Comp (Lounge Access / Upgrade)"
        elif 'seat' in top_driver.lower() or 'food' in top_driver.lower():
            rescue_action = "Amenity Upgrade (Premium Comp)"
    
    return {
        "prediction_id": str(uuid.uuid4()),
        "probability": float(prob),
        "trigger_diagnosis": top_driver,
        "prescriptive_rescue": rescue_action
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
