#!/bin/bash
# Start the Backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start the Frontend
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0