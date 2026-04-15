import pandas as pd
import numpy as np
import json
from fastapi.testclient import TestClient
from app.main import app
import warnings
warnings.filterwarnings('ignore')

print("Extracting verified 'WiFi-Frustrated Business Traveler' payload from dataset...")
df = pd.read_csv('notebooks/Hybrid_Aviation_Churn_Integrated.csv')
service_cols = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 
                'Inflight entertainment', 'On-board service', 'Leg room service', 
                'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']

df['csat_score'] = df[service_cols].mean(axis=1)
df['loyalty_shock_score'] = df['delay_intensity_log'] * df['service_friction_score']

mask = (df['Class'] == 'Business') & (df['Inflight wifi service'] <= 1) & (df['delay_intensity_log'] < 2.0)
target_df = df[mask].drop(columns=['churn'])

wifi_frustrated_customer = target_df.iloc[0].to_dict()

payload_dict = {k.replace(' ', '_'): v for k, v in wifi_frustrated_customer.items()}
payload = {
    "sector": "aviation",
    "data": payload_dict
}

print("\n--- Sending Unified Inference Payload ---")
with TestClient(app) as client:
    response = client.post("/predict", json=payload, headers={"x-api-key": "NERAL_SECRET_2026"})
    
    print(f"\nAPI Response Code: {response.status_code}")
    if response.status_code == 200:
        res_json = response.json()
        print("--- API JSON RESPONSE ---")
        print(json.dumps(res_json, indent=4))
        
        assert "Executive Concierge" in res_json.get("prescriptive_rescue", ""), "FATAL: Rescue Action Mapping Failed!"
        assert res_json.get("probability", 0) > 0.80, "FATAL: Target Churn Probability is too low!"
        print("\n✅ Verification Complete! Unified FastAPI Server successfully diagnosed digital friction and routed the Executive Concierge.")
    else:
        print(f"Server Error: {response.text}")
