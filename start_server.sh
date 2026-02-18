#!/bin/bash
# Startup script for LeRobot Inference Server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  LeRobot Inference Server Startup${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if lerobot is installed
echo -e "\n${YELLOW}Checking dependencies...${NC}"
if ! python -c "import lerobot" 2>/dev/null; then
    echo -e "${RED}Error: lerobot not installed${NC}"
    echo -e "Please install lerobot first:"
    echo -e "  cd /workspace/lerobot"
    echo -e "  pip install -e .[smolvla]"
    exit 1
fi
echo -e "${GREEN}✓ lerobot is installed${NC}"

# Check if FastAPI is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing server dependencies...${NC}"
    pip install -r requirements.txt
fi
echo -e "${GREEN}✓ Server dependencies installed${NC}"

# Load environment variables if .env exists
if [ -f .env ]; then
    echo -e "\n${YELLOW}Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
MODEL_ID=${MODEL_ID:-"NLTuan/smolvla_red_block_in_tape"}
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}
DEVICE=${DEVICE:-"cuda"}

echo -e "\n${YELLOW}Configuration:${NC}"
echo -e "  Model ID: ${MODEL_ID}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo -e "  Device: ${DEVICE}"

# Check GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo -e "${GREEN}✓ GPU available${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected, will use CPU${NC}"
    DEVICE="cpu"
fi

# Start server
echo -e "\n${GREEN}Starting inference server...${NC}"
echo -e "${YELLOW}Server will be available at: http://${HOST}:${PORT}${NC}"
echo -e "${YELLOW}API docs at: http://localhost:${PORT}/docs${NC}"
echo -e "\n${GREEN}================================================${NC}\n"

python3 inference_server.py \
    --model_id="${MODEL_ID}" \
    --host="${HOST}" \
    --port="${PORT}" \
    --device="${DEVICE}"
