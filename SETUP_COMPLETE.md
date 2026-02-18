# 🎉 LeRobot Inference Server - READY!

## ✅ What's Set Up

Your inference server is ready in `/workspace/vla_evaluation` with full support for **VSCode port forwarding** (DevTunnels)!

### Files Created:
- **`inference_server.py`** - FastAPI REST API server  
- **`client_example.py`** - Python client  
- **`client_tunnel.py`** - Tunnel-aware client (for your setup!)
- **`test_with_dataset.py`** - Test with real SmolVLA dataset
- **`requirements.txt`** - All dependencies ✅ INSTALLED
- **`QUICKSTART_TUNNELS.md`** - Your quick start guide
- **`test_tunnel.sh`** - One-command tunnel testing

## 🚀 Quick Start (use `python3` not `python`)

### 1. Start Server
```bash
cd /workspace/vla_evaluation
python3 inference_server.py --port=8000
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### 2. Forward Port in VSCode
1. Open **PORTS** tab (bottom panel)
2. Click **Forward a Port** → Enter `8000`
3. **Right-click → Port Visibility → Public** ⚠️ IMPORTANT!
4. Copy your tunnel URL (e.g., `https://xyz-8000.use.devtunnels.ms`)

### 3. Test (from anywhere!)
```bash
# Test health (replace with YOUR tunnel URL)
python3 -c "import requests; print(requests.get('https://YOUR-TUNNEL.use.devtunnels.ms/health').json())"

# Expected: {"status": "healthy", "model_loaded": true, ...}
```

### 4. Use the Client
```bash
# Option A: Interactive (will prompt for URL)
python3 client_tunnel.py

# Option B: Direct
python3 client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms

# Option C: Test script
./test_tunnel.sh https://YOUR-TUNNEL.use.devtunnels.ms
```

## 📝 Important Notes for SmolVLA

SmolVLA is a **Vision-Language-Action** model and requires:

1. **Camera images** (at least one)
2. **Task instruction** (text)
3. **Robot state** (joint positions)

### Example Request Format:
```python
import requests
import base64
import io
import numpy as np
from PIL import Image

# Encode image to base64
def encode_image(img_array):
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Your camera image (480x640x3 or any size)
camera_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Prepare observation
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 6 joint positions
    "task": "pick up the red block",  # Task description
    "observation.images.camera1": encode_image(camera_image)  # Base64 image
}

# Send request
response = requests.post(
    "https://YOUR-TUNNEL.use.devtunnels.ms/predict",
    json={"observation": observation},
    timeout=60
)

result = response.json()
print(f"Predicted {result['num_steps']} actions!")
print(f"Actions: {result['actions']}")
```

## 🔧 Server Status Commands

```bash
# Start server
python3 inference_server.py --port=8000

# Check if running
curl http://localhost:8000/health

# View logs (if running in background)
tail -f /tmp/server.log

# Stop server
pkill -f "inference_server.py"
```

## 📚 API Endpoints

All work through your tunnel URL!

- `GET /health` - Server health check
- `GET /model/info` - Model configuration
- `POST /predict` - Run inference

Full API docs at: `https://YOUR-TUNNEL.use.devtunnels.ms/docs`

## 💡 Tips for Your Setup

1. **Tunnel URL Changes**: Each VSCode restart gives a new URL
2. **Save Your URL**: `export TUNNEL_URL="https://..."`  
3. **Both Servers**: Run Flask (8080) and Inference (8000) together
4. **Test Locally First**: Use `localhost:8000` before tunneling
5. **HTTPS is Free**: VSCode tunnels automatic HTTPS

## 🐛 Troubleshooting

**"python: command not found"**  
→ Use `python3` instead of `python`

**Connection refused after forwarding port**
→ Make sure visibility is set to **Public** in PORTS tab

**"All image features are missing"**  
→ SmolVLA needs at least one camera image in base64 format

**Slow predictions through tunnel**  
→ Normal! Tunnel adds latency. For local testing use `--local` flag

## 📖 Further Reading

- `README.md` - Complete documentation
- `QUICKSTART_TUNNELS.md` - Detailed tunnel setup
- `VSCODE_TUNNEL_SETUP.md` - Advanced tunnel configuration
- `test_with_dataset.py` - See how to load real dataset images

## 🎯 Your Exact Workflow

Since you already have Flask on port 8080:

```bash
# Terminal 1: Your Flask server  
python3 main.py  # Port 8080

# Terminal 2: Inference server
cd vla_evaluation  
python3 inference_server.py --port=8000

# VSCode: Forward BOTH ports (8080 and 8000)
# Then test both:
python3 -c "import requests; print(requests.get('https://xxx-8080.use.devtunnels.ms/').json())"  # Flask
python3 -c "import requests; print(requests.get('https://xxx-8000.use.devtunnels.ms/health').json())"  # LeRobot
```

---

**You're all set! 🚀** The server is configured to work exactly like your Flask server but serves LeRobot model predictions!
