FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy lerobot source (assuming it's in parent directory)
COPY ../lerobot /app/lerobot

# Install lerobot with required extras
WORKDIR /app/lerobot
RUN pip3 install -e ".[smolvla]"

# Copy server files
WORKDIR /app/server
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY inference_server.py .
COPY server_config.yaml .
COPY client_example.py .
COPY test_server.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_ID=NLTuan/smolvla_red_block_in_tape
ENV PORT=8000
ENV HOST=0.0.0.0
ENV DEVICE=cuda

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run server
CMD ["python3", "inference_server.py"]
