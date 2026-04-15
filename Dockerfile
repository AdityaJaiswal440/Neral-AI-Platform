FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies FIRST for caching optimization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the environment file, the models, and the code
COPY .env .
COPY models/ ./models/
COPY app/ ./app/

# Expose API traffic port
EXPOSE 8000

# Fire up uvicorn server mapping to the app logic
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
