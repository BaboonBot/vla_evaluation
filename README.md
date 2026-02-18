# LeRobot Inference Server

A REST API server for serving LeRobot policy models. Provides a simple HTTP interface for inference with SmolVLA and other LeRobot policies.

## Features

- 🚀 Simple REST API with FastAPI  
- 🔌 Multiple policy support (SmolVLA, ACT, Diffusion, PI0)
- 🖼️ Image support via base64 encoding
- ⚡ GPU acceleration (automatic CUDA detection)
- 🌐 VSCode tunnel support for restricted networks

## Installation

```bash
# Install dependencies
cd /workspace/vla_evaluation
pip install -r requirements.txt

# Install LeRobot (if not already installed)
cd /workspace/lerobot
pip install -e .[smolvla]
```

## Quick Start

### 1. Start the Server

```bash
cd /workspace/vla_evaluation
python3 inference_server.py --port=8000
```

Server loads in ~20-30 seconds. You'll see:
```
Loading model from NLTuan/smolvla_red_block_in_tape...
Model loaded successfully
Uvicorn running on http://0.0.0.0:8000
```

### 2. Access the Server

**Local access:**
```bash
curl http://localhost:8000/health
```

**Remote access via VSCode tunnel:**
1. Open VSCode **PORTS** tab (bottom panel)
2. Click **Forward a Port** → enter `8000`
3. Right-click port 8000 → **Port Visibility** → **Public**
4. Copy the tunnel URL (e.g., `https://abc-8000.use.devtunnels.ms`)
5. Use this URL to access your server remotely

### 3. Make Predictions

**Using the Python client:**
```bash
# Local
python3 client_tunnel.py --local

# Remote (tunnel)
python3 client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms

# Interactive mode (will prompt for URL)
python3 client_tunnel.py
```

## API Endpoints

Visit http://localhost:8000/docs for interactive API documentation.

### `GET /health`
Check server health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "NLTuan/smolvla_red_block_in_tape",
  "device": "cuda"
}
```

### `GET /model/info`  
Get detailed model configuration.

**Response:**
```json
{
  "model_id": "NLTuan/smolvla_red_block_in_tape",
  "policy_type": "smolvla",
  "action_dim": 7,
  "chunk_size": 20,
  "camera_inputs": ["top"]
}
```

### `POST /predict`
Run inference on an observation.

**Request:**
```json
{
  "observation": {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block",
    "observation.images.top": "base64_encoded_image_string"
  },
  "action_steps": 10
}
```

**Response:**
```json
{
  "actions": [[0.1, 0.2, ...], ...],
  "action_dim": 7,
  "num_steps": 10
}
```

## Usage Examples

### Python Client

```python
from client_tunnel import LeRobotInferenceClient
import numpy as np

# Connect to server (local or tunnel)
client = LeRobotInferenceClient("http://localhost:8000")
# Or: client = LeRobotInferenceClient("https://abc-8000.use.devtunnels.ms")

# Get model info
info = client.get_model_info()
print(f"Model: {info['model_id']}, Action dim: {info['action_dim']}")

# Make prediction with image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block",
    "observation.images.top": client.encode_image(image)
}

result = client.predict(observation, action_steps=20)
print(f"Predicted {result['num_steps']} actions")
print(f"First action: {result['actions'][0]}")
```

### Command Line (curl)

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Prediction (requires base64 encoded image for SmolVLA)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": {"observation.state": [0.1,0.2,0.3,0.4,0.5,0.6], "task": "pick red block"}}'
```

## Configuration

### Command Line Arguments

```bash
python3 inference_server.py \
  --model_id=NLTuan/smolvla_red_block_in_tape \
  --device=cuda \
  --port=8000 \
  --host=0.0.0.0
```

### Environment Variables

```bash
export MODEL_ID=NLTuan/smolvla_red_block_in_tape
export PORT=8000
export DEVICE=cuda
python3 inference_server.py
```

### Config File

Edit `server_config.yaml`:

```yaml
model:
  model_id: "NLTuan/smolvla_red_block_in_tape"
  policy_type: "auto"
  device: null  # Auto-detect (cuda/cpu)

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

inference:
  max_batch_size: 1
  timeout: 30
```

## Troubleshooting

### Port Already in Use
```bash
# Kill existing server
pkill -f "inference_server.py"

# Or use a different port
python3 inference_server.py --port=8001
```

### CUDA Out of Memory
```bash
# Use CPU instead
python3 inference_server.py --device=cpu
```

### Tunnel Connection Issues
- Ensure port visibility is set to **Public** in VSCode
- Check firewall settings
- Verify tunnel URL is correct (should end with `.use.devtunnels.ms` or similar)

## Files

- **inference_server.py** - Main FastAPI server
- **client_tunnel.py** - Python client (supports local and tunnel)
- **requirements.txt** - Python dependencies
- **server_config.yaml** - Configuration file
- **.env.example** - Environment variable template

## Supported Models

The server supports all LeRobot policy types:

- **SmolVLA**: Vision-Language-Action models (e.g., `NLTuan/smolvla_red_block_in_tape`)
- **ACT**: Action Chunking Transformer (e.g., `lerobot/act_pusht_image`)
- **Diffusion**: Diffusion Policy (e.g., `lerobot/diffusion_pusht`)
- **PI0**: Physical Intelligence models (e.g., `physical-intelligence/pi0-pusht`)
- **TDMPC**, **VQ-BeT**, and other LeRobot policies

## License

See [LICENSE](LICENSE) file.
