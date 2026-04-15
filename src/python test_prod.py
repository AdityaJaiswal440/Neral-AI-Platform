import requests
import json

# Your LIVE URL from Render
URL = "https://neral-ai-backend.onrender.com/v1/predict"
HEADERS = {
    "x-api-key": "NERAL_SECRET_2026",
    "Content-Type": "application/json"
}

# The "WiFi Paradox" Test Case
payload = {
    "sector": "aviation",
    "features": {
        "Inflight_wifi_service": 1,
        "Online_boarding": 1,
        "Inflight_entertainment": 4,
        "Class": "Business",
        "Customer_Type": "Loyal Customer",
        "Type_of_Travel": "Business travel",
        "Flight_Distance": 1200,
        "Departure_Delay_Minutes": 10
    }
}

print(f"--- Sending Request to {URL} ---")
try:
    # Note: It might take 30-60 seconds to "wake up" Render the first time
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=120)
    
    print(f"Status Code: {response.status_code}")
    print("Response Data:")
    print(json.dumps(response.json(), indent=4))

except Exception as e:
    print(f"An error occurred: {e}")