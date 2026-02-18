# Quick Usage Guide

## Starting the Server

```bash
cd /workspace/vla_evaluation
python3 inference_server.py --port=8000
```

Wait ~20-30 seconds for "Model loaded successfully" message.

## Testing Locally

```bash
# Test connection
curl http://localhost:8000/health

# Run examples
python3 client_tunnel.py --local
```

## Remote Access (VSCode Tunnels)

### Setup Port Forwarding

1. Open VSCode bottom panel → **PORTS** tab
2. Click **"Forward a Port"** → Enter `8000`
3. Right-click port 8000 → **Port Visibility** → **Public**
4. Copy the tunnel URL (e.g., `https://abc-8000.use.devtunnels.ms`)

### Use the Tunnel

```bash
# Test
python3 client_tunnel.py --tunnel_url=https://YOUR-TUNNEL-URL.use.devtunnels.ms

# Or interactive mode
python3 client_tunnel.py
```

## Python Code Example

```python
from client_tunnel import LeRobotInferenceClient
import numpy as np

# Connect to server
client = LeRobotInferenceClient("http://localhost:8000")
# Or: client = LeRobotInferenceClient("https://your-tunnel-url.use.devtunnels.ms")

# Check server
health = client.health_check()
print(f"Server status: {health['status']}")
print(f"Device: {health['device']}")

# Get model info
info = client.get_model_info()
print(f"Model: {info['model_id']}")
print(f"Action dim: {info['action_dim']}")
print(f"Required cameras: {info['camera_inputs']}")

# Make prediction (MUST include an image for SmolVLA)
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block",
    "observation.images.camera1": client.encode_image(image)
}

result = client.predict(observation, action_steps=10)
print(f"Predicted {result['num_steps']} action steps")
print(f"First action: {result['actions'][0]}")
```

## Important Notes

### SmolVLA Requirements

**All examples now include camera images** because SmolVLA requires them. Valid camera keys:
- `observation.images.camera1`
- `observation.images.camera2`
- `observation.images.camera3`
- `observation.images.empty_camera_0`

**Every request must include at least one camera image.**

### Image Format

Images must be:
- NumPy arrays with shape `(height, width, 3)`
- dtype: `uint8` (0-255 range)
- RGB format
- Recommended size: 480x640 (will be resized by model)

### State Format

State observations should be a list of floats:
```python
"observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
```

### Task Description

Include a natural language task description:
```python
"task": "pick up the red block"
```

## Common Issues

### Server Won't Start (Port in Use)
```bash
pkill -f "inference_server.py"
python3 inference_server.py --port=8000
```

### CUDA Out of Memory
```bash
python3 inference_server.py --device=cpu
```

### Timeout Errors on Consecutive Requests

**Problem:** Step 2 or 3 times out after Step 1 succeeds.

**Cause:** Consecutive inference requests can take longer as GPU memory accumulates. The default 60-second timeout may be insufficient for slower systems or complex models.

**Solutions:**

**Option 1: Increase timeout (recommended)**
```bash
# Increase to 180 seconds
python3 client_tunnel.py --local --timeout=180

# Or in your code
client = LeRobotInferenceClient("http://localhost:8000", timeout=180)
```

**Option 2: Predict fewer action steps**
```python
# Predict fewer steps (faster inference)
result = client.predict(observation, action_steps=5)  # Instead of 32
```

**Option 3: Restart server between long sessions**
```bash
pkill -f "inference_server.py"
python3 inference_server.py --port=8000
```

### Tunnel Connection Failed
- Check port visibility is **Public** in VSCode
- Verify tunnel URL is correct
- Test locally first: `curl http://localhost:8000/health`

### 500 Error: "All image features are missing"
**This should not happen with the provided examples** (they all include images now).

If you see this in your own code, you forgot to include a camera image:

```python
# ❌ WRONG - Missing image
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block"
}

# ✓ CORRECT - Includes image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block",
    "observation.images.camera1": client.encode_image(image)
}
```

## File Structure

```
vla_evaluation/
├── inference_server.py      # Main FastAPI server
├── client_tunnel.py          # Python client library
├── requirements.txt          # Dependencies
├── server_config.yaml        # Configuration
├── README.md                 # Full documentation
├── USAGE.md                  # This file (quick reference)
└── smolvla.ipynb            # Original notebook
```

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Server health check
- `GET /model/info` - Model configuration
- `POST /predict` - Run inference
- `GET /docs` - Interactive API documentation

## Model Information

Current model: `NLTuan/smolvla_red_block_in_tape`
- Action dimension: 6
- Chunk size: 32 (predicts 32 future actions by default)
- Requires: state + task + camera image(s)
