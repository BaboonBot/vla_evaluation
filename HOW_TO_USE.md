# How to Use Your LeRobot Inference Server

## Step 1: Start the Server

In your terminal:
```bash
cd /workspace/vla_evaluation
python3 inference_server.py --port=8000
```

You'll see:
```
Loading model from NLTuan/smolvla_red_block_in_tape...
Model loaded successfully
...
Uvicorn running on http://0.0.0.0:8000
```

⏱️ **Takes ~20-30 seconds to load the model**

## Step 2: Forward Port in VSCode

1. Look at the bottom of VSCode → find the **PORTS** tab
2. Click the **"Forward a Port"** button (or **+** icon)
3. Type: `8000`
4. **IMPORTANT**: Right-click the new port → **Port Visibility** → **Public**
5. Copy the **Forwarded Address** (looks like: `https://abc123-8000.use.devtunnels.ms`)

## Step 3: Test Your Tunnel

Open a NEW terminal and run:

```bash
# Replace with YOUR actual tunnel URL from step 2
export TUNNEL_URL="https://abc123-8000.use.devtunnels.ms"

# Test it (should return health status)
python3 -c "import requests; print(requests.get('$TUNNEL_URL/health').json())"
```

**Expected output:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "NLTuan/smolvla_red_block_in_tape",
  "device": "cuda"
}
```

## Step 4: Make Your First Prediction

### Quick Test (Simple):
```python
import requests

tunnel_url = "https://YOUR-TUNNEL-URL.use.devtunnels.ms"

# For SmolVLA, you need: state + task + camera image
response = requests.post(
    f"{tunnel_url}/predict",
    json={
        "observation": {
            "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "task": "pick up the red block"
            # Note: This will fail because SmolVLA needs images!
        }
    }
)
print(response.json())
```

### Full Example (With Camera Image):
```python
import requests
import base64
import io
import numpy as np
from PIL import Image

tunnel_url = "https://YOUR-TUNNEL-URL.use.devtunnels.ms"

# Create a test image (in reality, get this from your camera)
camera_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Convert to base64
img = Image.fromarray(camera_image)
buffer = io.BytesIO()
img.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode()

# Make prediction
response = requests.post(
    f"{tunnel_url}/predict",
    json={
        "observation": {
            "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "task": "pick up the red block",
            "observation.images.camera1": image_b64  # Base64 image
        }
    },
    timeout=60
)

result = response.json()
print(f"Predicted {result['num_steps']} action steps")
print(f"First action: {result['actions'][0]}")
```

## Step 5: Use the Pre-Made Client

We've created helper scripts for you:

### Option A: Interactive Client
```bash
python3 client_tunnel.py
# It will ask for your tunnel URL
```

### Option B: Direct Client
```bash
python3 client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms
```

### Option C: Test with Real Dataset
```bash
python3 test_with_dataset.py
# Uses actual dataset images to test the server
```

### Option D: Quick Tunnel Test Script
```bash
./test_tunnel.sh https://YOUR-TUNNEL.use.devtunnels.ms
# Runs all tests automatically
```

## Common Workflows

### For Testing Locally (Same Machine)
```bash
# Terminal 1: Start server
python3 inference_server.py --port=8000

# Terminal 2: Test locally
curl http://localhost:8000/health
python3 client_tunnel.py --local
```

### For Remote Access (Through Tunnel)
```bash
# Terminal 1: Start server
python3 inference_server.py --port=8000

# VSCode: Forward port 8000 (set to Public!)

# Terminal 2: Test remotely
python3 client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms
```

### For Integration with Your Code
```python
from client_example import LeRobotInferenceClient

# Create client (works with both local and tunnel URLs)
client = LeRobotInferenceClient("https://YOUR-TUNNEL.use.devtunnels.ms")

# Get model info
info = client.get_model_info()
print(f"Action dimension: {info['action_dim']}")
print(f"Cameras needed: {info['camera_inputs']}")

# Make prediction
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block",
    "observation.images.camera1": client.encode_image(your_image_array)
}
result = client.predict(observation)
print(f"Actions: {result['actions']}")
```

## Checking Server Status

```bash
# See if server is running
curl http://localhost:8000/health

# View server logs (if running in background)
tail -f /tmp/inference_server.log

# Check what's using port 8000
lsof -i :8000

# Stop the server
pkill -f "inference_server.py"
```

## API Documentation

Once the server is running, visit in your browser:
- **Interactive docs**: `https://YOUR-TUNNEL.use.devtunnels.ms/docs`
- **Alternative docs**: `https://YOUR-TUNNEL.use.devtunnels.ms/redoc`

## Troubleshooting

### Server won't start
```bash
# Check if port is already in use
lsof -i :8000

# View error logs
cat /tmp/inference_server.log
```

### Can't connect through tunnel
- Make sure port visibility is **Public** (not Private)
- Check PORTS tab in VSCode for the correct URL
- Test locally first: `curl http://localhost:8000/health`

### Slow predictions
- Normal with tunneling (adds network latency)
- For faster testing, use `--local` flag
- Consider running on same machine as robot

### Need a different model
```bash
python3 inference_server.py \
  --model_id=your/model-id \
  --policy_type=act \
  --port=8000
```

## Quick Reference

```bash
# Start server
python3 inference_server.py --port=8000

# Test health
curl http://localhost:8000/health

# Test with dataset
python3 test_with_dataset.py

# Use client
python3 client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms

# Stop server
pkill -f "inference_server.py"
```

---

**That's it!** Your server is now serving LeRobot model predictions via REST API! 🎉
