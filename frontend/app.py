import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go

# --- THE FOUNDRY AESTHETIC (Neon Carbon) ---
st.set_page_config(page_title="NERAL AI | FOUNDRY TERMINAL", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #05070a !important;
        font-family: 'JetBrains Mono', monospace;
        color: #00ff41;
    }
    
    .stTextArea textarea {
        background-color: #0a0e14 !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #8b0000);
        color: white; border: none; font-weight: bold;
        text-transform: uppercase; letter-spacing: 2px;
        width: 100%; height: 3em;
    }

    .diagnosis-card {
        background-color: #0a0e14;
        border: 1px solid #1f2937;
        padding: 24px; border-radius: 4px;
        box-shadow: inset 0 0 10px #000;
    }
    
    h1, h2, h3 { color: #ffffff !important; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

# --- APP LAYOUT ---
st.title("🛰️ HYBRID CHURN INTELLIGENCE MODEL v2.3")
st.caption("ENGINE: v6.1 GROUND TRUTH | STATUS: ACTIVE")

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("⌨️ Input Logic")
    sector = st.selectbox("SUBSYSTEM", ["Ecommerce", "Aviation"])
    
    # Pre-configured startup payloads
    default_data = {
        "ecommerce": {
            "sector": "ecommerce",
            "features": {
                "Monthly_Charges": 950.0, "Tenure": 1, "City": "Vadodara",
                "csat_score": 5, "monthly_logins": 15, "total_monthly_time": 120,
                "Contract_Type": "Month-to-Month"
            }
        },
        "aviation": {
            "sector": "aviation",
            "features": {
                "Type of Travel": "Business travel", "Class": "Business",
                "Departure Delay in Minutes": 180, "Inflight entertainment": 2,
                "Online boarding": 3
            }
        }
    }

    input_json = st.text_area("PAYLOAD FOUNDRY:", 
                              value=json.dumps(default_data[sector.lower()], indent=2), 
                              height=400)
    execute = st.button("🚀 EXECUTE INFERENCE")

with col2:
    st.subheader("⚛️ Reactor Feedback")
    
    if execute:
        try:
            # Pointing to the new modular backend
            url = "http://127.0.0.1:8000/v1/predict"
            headers = {"x-api-key": "NERAL_SECRET_2026"}
            
            payload = json.loads(input_json)
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                res = response.json()
                prob = res['probability']
                risk_pct = round(prob * 100, 1) # Percentage logic
                driver = res['trigger_diagnosis']
                
                # 1. THE ANIMATED REACTOR GAUGE
                # Dynamic coloring: 0-30: Green, 30-70: Yellow, 70-100: Red
                color_map = "#00ff41" if risk_pct < 30 else "#ffcc00" if risk_pct < 70 else "#ff4b4b"
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_pct,
                    number = {'suffix': "%", 'font': {'size': 90, 'color': "white"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "#00ff41", 'tickwidth': 2},
                        'bar': {'color': color_map},
                        'bgcolor': "#0a0e14",
                        'borderwidth': 1, 'bordercolor': "#30363d",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(0, 255, 65, 0.05)'},
                            {'range': [30, 70], 'color': 'rgba(255, 204, 0, 0.05)'},
                            {'range': [70, 100], 'color': 'rgba(255, 75, 75, 0.05)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=380, margin=dict(t=60, b=0, l=25, r=25),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. FOUNDRY DIAGNOSIS BOX
                st.markdown(f"""
                <div class="diagnosis-card" style="border-top: 4px solid {color_map};">
                    <p style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px;">Verified Diagnosis</p>
                    <h2 style="color: {color_map}; margin-top: 0px; font-size: 1.5rem;">{driver}</h2>
                    <p style="color: #374151; font-size: 0.65rem; font-family: monospace;">TRACE_ID: {res['prediction_id']}</p>
                </div>
                """, unsafe_allow_html=True)

                # 3. LOGIC VECTORS (SHAP MOCK)
                st.divider()
                st.subheader("📊 Signal Telemetry")
                # Using columns for a cleaner "Metric Grid" look
                m1, m2, m3 = st.columns(3)
                m1.metric("CORE_TEMP", "98.6%", "STABLE")
                m2.metric("SIGNAL_STRENGTH", "0.88", "HIGH")
                m3.metric("LATENCY", "42ms", "-4ms")
                
            else:
                st.error(f"REACTOR CRITICAL FAILURE: {response.text}")
        except Exception as e:
            st.error(f"TERMINAL OFFLINE: Ensure uvicorn is running on port 8000.")
    else:
        st.write("### AWAITING PAYLOAD...")

# --- FOOTER ---
st.divider()
st.markdown("<p style='text-align: center; color: #1f2937; font-size: 0.7rem;'>NERAL AI STARTUP CORE v2.3 | GSV VADODARA | PROPRIETARY IP</p>", unsafe_allow_html=True)