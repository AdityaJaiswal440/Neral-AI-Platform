"""
frontend/app.py  —  SPOTLIGHT SKELETON
Neral AI Command Bridge · Streamlit UI · Public Reference Release

REDACTED:
  - Default payload feature values (sector-specific field names and values)
  - Actual API key value

PUBLISHED:
  - Full Secured REST Gateway connection logic
  - Plotly Indicator risk gauge rendering pipeline
  - Diagnosis card HTML injection pattern
  - SHAP force plot rendering architecture (Plotly horizontal bar)
  - Dynamic risk-band color mapping logic
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & NEON CARBON AESTHETIC
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NERAL AI | Churn Intelligence",
    layout="wide",
    page_icon="🛰️",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
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

    .stButton > button {
        background: linear-gradient(90deg, #ff4b4b, #8b0000);
        color: white;
        border: none;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        height: 3em;
    }

    .diagnosis-card {
        background-color: #0a0e14;
        border: 1px solid #1f2937;
        padding: 24px;
        border-radius: 4px;
        box-shadow: inset 0 0 10px #000;
    }

    h1, h2, h3 { color: #ffffff !important; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SECURED REST GATEWAY CONFIG
#    Backend runs on port 8000 (container-internal) and is exposed publicly
#    via HF Spaces' routed URL. The x-api-key header is the sole auth layer.
#    In production: GATEWAY_URL and GATEWAY_KEY are set as environment vars.
# ─────────────────────────────────────────────────────────────────────────────
GATEWAY_URL = "https://adityajaiswal440-neral-ai-backend.hf.space/v1/predict"
GATEWAY_KEY = "[REDACTED — set NERAL_SECRET in HF Space environment]"

GATEWAY_HEADERS = {
    "x-api-key": GATEWAY_KEY,
    "Content-Type": "application/json",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. RISK-BAND COLOR MAP
#    Three discrete bands matching the gauge step definitions.
#    Threshold logic is deterministic — no model involvement.
# ─────────────────────────────────────────────────────────────────────────────
def resolve_risk_color(risk_pct: float) -> str:
    if risk_pct < 30:
        return "#00ff41"    # Green  — low churn probability
    elif risk_pct < 70:
        return "#ffcc00"    # Yellow — moderate risk
    else:
        return "#ff4b4b"    # Red    — critical churn signal


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTLY RISK GAUGE — REACTOR INDICATOR
#    Renders a tri-band gauge with a hard threshold line at 90%.
#    The gauge is themed to the Neon Carbon design system.
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_gauge(risk_pct: float) -> go.Figure:
    color = resolve_risk_color(risk_pct)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 90, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#00ff41", "tickwidth": 2},
            "bar": {"color": color},
            "bgcolor": "#0a0e14",
            "borderwidth": 1,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 30],  "color": "rgba(0, 255, 65, 0.05)"},
                {"range": [30, 70], "color": "rgba(255, 204, 0, 0.05)"},
                {"range": [70, 100],"color": "rgba(255, 75, 75, 0.05)"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,    # Hard escalation threshold
            },
        },
    ))

    fig.update_layout(
        height=380,
        margin=dict(t=60, b=0, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP FORCE PLOT — HORIZONTAL BAR RENDERER
#    Renders a SHAP-style force plot using a Plotly horizontal bar chart.
#    Input: driver_list — list of {"name": str, "phi": float} dicts, sorted by |phi|.
#    Positive phi → pushes toward churn (red). Negative phi → retention signal (green).
#    NOTE: The API currently returns only the top driver. A full driver_list
#    would require the backend to expose the complete contribution vector.
#    This renderer is architected to receive that full vector when available.
# ─────────────────────────────────────────────────────────────────────────────
def render_shap_force_plot(driver_list: list) -> go.Figure:
    # Sort by absolute contribution magnitude
    driver_list = sorted(driver_list, key=lambda x: abs(x["phi"]))

    names = [d["name"] for d in driver_list]
    values = [d["phi"] for d in driver_list]
    colors = ["#ff4b4b" if v > 0 else "#00ff41" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1f2937", width=0.5)),
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#9ca3af", size=10, family="JetBrains Mono"),
    ))

    fig.update_layout(
        title=dict(
            text="SHAP Contribution Vector — Atomic Alignment",
            font=dict(color="#ffffff", size=13, family="JetBrains Mono"),
        ),
        xaxis=dict(
            title="φ_j  (Shapley Value)",
            color="#6b7280",
            gridcolor="#1f2937",
            zerolinecolor="#374151",
        ),
        yaxis=dict(color="#00ff41", gridcolor="#1f2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#05070a",
        height=max(250, len(driver_list) * 28),
        margin=dict(l=10, r=60, t=40, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. DIAGNOSIS CARD — HTML INJECTION
#    Renders the trigger_diagnosis field with risk-band color and trace ID.
# ─────────────────────────────────────────────────────────────────────────────
def render_diagnosis_card(driver: str, color: str, trace_id: str):
    st.markdown(f"""
    <div class="diagnosis-card" style="border-top: 4px solid {color};">
        <p style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 4px;">
            Atomic SHAP · Verified Trigger Diagnosis
        </p>
        <h2 style="color: {color}; margin-top: 0; font-size: 1.5rem;">{driver}</h2>
        <p style="color: #374151; font-size: 0.65rem; font-family: monospace;">
            TRACE_ID: {trace_id}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. APPLICATION LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛰️ HYBRID CHURN INTELLIGENCE MODEL v6.1")
st.caption("ENGINE: Atomic SHAP Alignment · GATEWAY: Secured REST · STATUS: ACTIVE")

col_input, col_output = st.columns([1.1, 1])

with col_input:
    st.subheader("⌨️ Payload Input")
    sector = st.selectbox("SUBSYSTEM", ["Ecommerce", "Aviation"])

    # [REDACTED] — default_data payloads contain proprietary field names
    # In production this is pre-populated from the Behavioral Foundry feature registry.
    default_payload = {
        "sector": sector.lower(),
        "features": {},  # [REDACTED]
    }

    raw_json = st.text_area(
        "PAYLOAD INPUT:",
        value=json.dumps(default_payload, indent=2),
        height=400,
    )
    execute = st.button("🚀 EXECUTE INFERENCE")


with col_output:
    st.subheader("⚛️ Reactor Feedback")

    if execute:
        try:
            # ── SECURED REST GATEWAY CALL ──────────────────────────────────
            payload = json.loads(raw_json)
            response = requests.post(
                GATEWAY_URL,
                json=payload,
                headers=GATEWAY_HEADERS,
                timeout=30,
            )

            if response.status_code == 200:
                res = response.json()
                prob       = res["probability"]
                risk_pct   = round(prob * 100, 1)
                driver     = res["trigger_diagnosis"]
                trace_id   = res["prediction_id"]
                color      = resolve_risk_color(risk_pct)

                # ── RISK GAUGE ─────────────────────────────────────────────
                st.plotly_chart(render_risk_gauge(risk_pct), use_container_width=True)

                # ── DIAGNOSIS CARD ─────────────────────────────────────────
                render_diagnosis_card(driver, color, trace_id)

                # ── SHAP FORCE PLOT ────────────────────────────────────────
                # Current API returns only top driver. Full driver_list would
                # require backend to expose complete φ_j vector.
                # Stub: single-entry force plot for architecture demonstration.
                st.divider()
                st.subheader("📊 SHAP Force Plot — Signal Attribution")
                stub_drivers = [{"name": driver, "phi": prob}]
                st.plotly_chart(render_shap_force_plot(stub_drivers), use_container_width=True)
                st.caption(
                    "⚠ Full contribution vector not exposed in current API contract. "
                    "Stub renders top driver only."
                )

            elif response.status_code == 403:
                st.error("GATEWAY: 403 — API key rejected. Check NERAL_SECRET configuration.")
            elif response.status_code == 400:
                st.error(f"GATEWAY: 400 — Invalid sector or malformed payload. → {response.text}")
            else:
                st.error(f"GATEWAY: {response.status_code} — Unexpected failure. → {response.text}")

        except requests.exceptions.Timeout:
            st.error("GATEWAY TIMEOUT — Backend did not respond within 30s. Model may still be loading.")
        except json.JSONDecodeError:
            st.error("PAYLOAD ERROR — Input is not valid JSON.")
        except Exception as exc:
            st.error(f"TERMINAL FAULT — {type(exc).__name__}: {exc}")

    else:
        st.markdown("""
        <div style="color: #1f2937; font-family: JetBrains Mono; padding: 40px 0; text-align: center;">
            <p>◈ AWAITING PAYLOAD ◈</p>
            <p style="font-size: 0.7rem;">Construct JSON in the Payload Input panel and execute inference.</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align: center; color: #1f2937; font-size: 0.7rem;'>"
    "NERAL AI · SPOTLIGHT RELEASE · GSV VADODARA · PROPRIETARY INTERNALS WITHHELD"
    "</p>",
    unsafe_allow_html=True,
)
