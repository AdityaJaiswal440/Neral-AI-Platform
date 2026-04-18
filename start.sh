#!/bin/bash
# 1. Start the Backend (The Brain) on port 8000 in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Wait 5 seconds for the brain to wake up
sleep 5

# 3. Start the Frontend (The UI) on port 7860
# This makes the UI the ONLY thing the viewer sees
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0