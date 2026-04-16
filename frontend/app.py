import streamlit as st
import json
import requests
import plotly.graph_objects as go
from datetime import datetime
import uuid

# --- 1. SYSTEM CONFIG & SECRETS ---
st.set_page_config(page_title="Neral AI | Behavioral Churn Intelligence", layout="wide", initial_sidebar_state="collapsed")

# For local testing, ensure you have .streamlit/secrets.toml or these fallbacks
BACKEND_URL = st.secrets.get("BACKEND_URL", "https://adityajaiswal440-neral-ai-backend.hf.space/v1/predict")
NERAL_SECRET = st.secrets.get("NERAL_SECRET", "NERAL_SECRET_2026")

# --- 2. INITIALIZE SESSION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- 3. THE "DEEP SPACE" CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .hero-text { text-align: center; padding: 80px 20px 20px 20px; }
    .hero-title { font-size: 4.5rem; font-weight: 800; color: #ff4b4b; margin-bottom: 0px; letter-spacing: -2px; }
    .hero-subtitle { font-size: 1.4rem; color: #888; margin-bottom: 40px; }
    
    /* Main CTA Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8080 100%);
        color: white; border: none; padding: 12px 40px;
        font-size: 1.2rem; border-radius: 50px; font-weight: bold;
        margin: 0 auto; display: block;
    }
    
    /* The Terminal Input */
    .stTextArea textarea { 
        background-color: #10141b !important; 
        color: #00ff00 !important; 
        font-family: 'Courier New', monospace !important; 
        border: 1px solid #333 !important; 
    }
    
    /* Diagnosis Box */
    .diagnosis-box { 
        border: 1px solid #ff4b4b; 
        padding: 20px; 
        background: rgba(255, 75, 75, 0.05); 
        border-radius: 10px; 
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. PAGE: WELCOME LAUNCHPAD ---
def show_welcome():
    st.markdown('<div class="hero-text">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">NERAL AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Unified Behavioral Churn Intelligence</p>', unsafe_allow_html=True)
    
    if st.button("INITIALIZE SYSTEM CORE"):
        st.session_state.authenticated = True
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("<div style='text-align:center;'><h4>E-COMMERCE</h4><p style='color:#666;'>Behavioral signal analysis.</p></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div style='text-align:center;'><h4>AVIATION</h4><p style='color:#666;'>Friction point stabilization.</p></div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div style='text-align:center;'><h4>SHAP ENGINE</h4><p style='color:#666;'>Verifiable AI diagnosis.</p></div>", unsafe_allow_html=True)

# --- 5. PAGE: THE DATA TERMINAL ---
def show_terminal():
    st.markdown("### 🌌 NERAL AI: DATA TERMINAL")
    st.markdown(f"Status: `CONNECTED` | Endpoint: `{BACKEND_URL[:30]}...`")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⌨️ Input Payload")
        default_json = {
            "sector": "ecommerce",
            "features": {
                "Monthly_Charges": 99.99, "Tenure": 12, "Total_Usage_GB": 500,
                "monthly_logins": 25, "nps": 8, "csat": 4,
                "city": "Vadodara", "customer_segment": "Premium"
            }
        }
        json_input = st.text_area("JSON FOUNDRY:", value=json.dumps(default_json, indent=4), height=400)
        execute_btn = st.button("🚀 EXECUTE INFERENCE")

    with col2:
        st.subheader("⚛️ Reactor Feedback")
        
        if execute_btn:
            try:
                # Local Validation
                payload = json.loads(json_input)
                
                # API Call
                headers = {"x-api-key": NERAL_SECRET, "Content-Type": "application/json"}
                response = requests.post(BACKEND_URL, headers=headers, json=payload, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    prob = data['probability']
                    diag = data.get('trigger_diagnosis', 'UNKNOWN')

                    # 1. Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickcolor': "white"},
                            'bar': {'color': "#ff4b4b" if prob > 0.5 else "#00ff00"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 1,
                            'bordercolor': "gray",
                        }
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Courier New"})
                    st.plotly_chart(fig, use_container_width=True)

                    # 2. Diagnosis Card
                    st.markdown(f"""
                        <div class="diagnosis-box">
                            <h4 style="color: #ff4b4b; margin:0;">VERIFIED DIAGNOSIS</h4>
                            <hr style="border-color:#333;">
                            <p>PRIMARY CHURN DRIVER: <strong style="color:#00ff00;">{diag.upper()}</strong></p>
                            <p style="font-size:0.8rem; color:#666;">ID: {data.get('prediction_id', uuid.uuid4())}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    st.code(response.text)
            
            except json.JSONDecodeError:
                st.error("Syntax Error: Invalid JSON format.")
            except Exception as e:
                st.error(f"Connection Failed: {e}")
        else:
            st.info("System Standby. Awaiting JSON Payload.")

    if st.sidebar.button("LOGOUT / REBOOT"):
        st.session_state.authenticated = False
        st.rerun()

# --- 6. ROUTING ENGINE ---
if not st.session_state.authenticated:
    show_welcome()
else:
    show_terminal()