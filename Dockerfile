FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Grant execution rights to the start script
RUN chmod +x start.sh

# Hugging Face uses port 7860 by default
EXPOSE 7860

CMD ["./start.sh"]