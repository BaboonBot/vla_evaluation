# 🚀 Quick Start for Your Setup (VSCode Tunnels)

Since you're using VSCode port forwarding through DevTunnels, here's your streamlined workflow:

## 1️⃣ Start the Inference Server (One Time)

```bash
cd /workspace/vla_evaluation
python3 inference_server.py --port=8000
```

Wait for: `PolicyServer started on 0.0.0.0:8000`

## 2️⃣ Forward the Port in VSCode

1. Open **PORTS** tab (bottom panel in VSCode)
2. Click **➕ Forward a Port**  
3. Enter `8000`
4. **Right-click → Port Visibility → Public** (important!)
5. Copy the forwarded URL (e.g., `https://c36pk8pn-8000.use.devtunnels.ms`)

## 3️⃣ Test Your Tunnel

```bash
# Replace with YOUR actual tunnel URL
export TUNNEL_URL="https://c36pk8pn-8000.use.devtunnels.ms"

# Quick test (just like your Flask example)
python3 -c "import requests; print(requests.get('$TUNNEL_URL/health').json())"
```

Expected output:
```json
{"status": "healthy", "model_loaded": true, "model_id": "...", "device": "cuda"}
```

## 4️⃣ Use the Client

### Easy mode (interactive):
```bash
python client_tunnel.py
# It will ask for your tunnel URL
```

### Direct mode:
```bash
python client_tunnel.py --tunnel_url=$TUNNEL_URL
```

### In your own code:
```python
from client_example import LeRobotInferenceClient

# Use your tunnel URL (changes each VSCode restart)
client = LeRobotInferenceClient("https://c36pk8pn-8000.use.devtunnels.ms")

# Make predictions
observation = {"observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
result = client.predict(observation)
print(f"Got {result['num_steps']} actions:", result['actions'])
```

## 🔥 Pro Tips for Your Setup

### Save your tunnel URL each session:
```bash
# In your terminal once you forward the port
echo "export TUNNEL_URL='https://xyz-8000.use.devtunnels.ms'" >> ~/.bashrc_session
source ~/.bashrc_session

# Then use $TUNNEL_URL everywhere
python client_tunnel.py --tunnel_url=$TUNNEL_URL
```

### Run both servers (Flask + Inference):
```bash
# Terminal 1: Your Flask server
python main.py  # Port 8080

# Terminal 2: Inference server  
cd vla_evaluation && python inference_server.py --port=8000

# Forward both ports in VSCode:
# - 8080 → Flask health check
# - 8000 → LeRobot inference
```

### Quick test script:
```bash
# Test everything at once
./test_tunnel.sh https://YOUR-TUNNEL-URL.use.devtunnels.ms
```

## 📝 Common Commands

```bash
# Start server
python3 inference_server.py --port=8000

# Test from anywhere (replace URL)
python3 -c "import requests; print(requests.get('https://YOUR-TUNNEL.use.devtunnels.ms/health').json())"

# Run examples  
python client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms --example=simple

# Test only (no examples)
python client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms --test-only
```

## ⚠️ Troubleshooting

**"Connection refused"**
→ Make sure port visibility is **Public** (not Private) in VSCode

**"404 Not Found"**  
→ Double-check the tunnel URL in the PORTS tab

**Tunnel URL keeps changing**
→ Normal behavior. The URL changes each VSCode restart. You can:
   - Copy it each time from PORTS tab
   - Use the interactive client: `python client_tunnel.py`
   - Set up persistent tunnels (see VSCODE_TUNNEL_SETUP.md)

**Slow predictions**
→ Expected with tunneling. For testing, you can use `--local` flag:
   ```bash
   python client_tunnel.py --local  # Uses localhost instead
   ```

## 🎯 Your Exact Use Case

Based on your Flask example, here's the equivalent:

**Your Flask test:**
```bash
python3 -c "import requests; print(requests.get('https://c36pk8pn-8080.use.devtunnels.ms/').json())"
# Output: {"status": "online", "message": "Vast.ai Brain is Ready"}
```

**LeRobot inference test:**
```bash
python3 -c "import requests; print(requests.get('https://c36pk8pn-8000.use.devtunnels.ms/health').json())"
# Output: {"status": "healthy", "model_loaded": true, ...}
```

**LeRobot prediction:**
```python
import requests
result = requests.post(
    'https://c36pk8pn-8000.use.devtunnels.ms/predict',
    json={'observation': {'observation.state': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}}
).json()
print(f"Predicted actions: {result['actions']}")
```

That's it! The server works exactly like your Flask server, just expose port 8000 instead of 8080. 🎉
