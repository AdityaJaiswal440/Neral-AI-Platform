#!/bin/bash

# 1. Kill any existing processes (Safety check)
fuser -k 8000/tcp
fuser -k 7860/tcp

# 2. Start the Backend on Port 8000 (The Brain)
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 3. Wait for the engine to warm up
sleep 10

# 4. Start the Frontend on Port 7860 (The Spotlight)
# We add the "headless" flag for cloud stability
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true