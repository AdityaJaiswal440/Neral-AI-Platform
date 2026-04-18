#!/bin/bash
# 1. Start the Backend (The Brain) on port 8000 in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Start the Frontend (The UI) on port 7860
# Hugging Face will now show the Streamlit UI to every visitor
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0