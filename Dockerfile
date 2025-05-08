# Use Python 3.10 as base image
FROM python:3.10-slim

# Install system dependencies including curl for Ollama installation
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a script to handle the startup sequence
RUN echo '#!/bin/bash\n\
ollama serve & \
sleep 5 && \
ollama pull llama3.2 && \
ollama pull mxbai-embed-large && \
python llm-compare.py "$1" "$2"' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
