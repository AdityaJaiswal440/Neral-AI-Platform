#!/bin/bash

# 1. Start the Brain (FastAPI) in the background on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Critical: Wait for models to load before the UI connects
sleep 15

# 3. Start the Terminal (Streamlit) on port 7860
# Hugging Face ONLY displays what is on 7860
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true