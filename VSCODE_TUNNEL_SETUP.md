# VSCode Port Forwarding Setup Guide

## Your Situation
Since your school network blocks direct connections, you're using VSCode's DevTunnels port forwarding. This works perfectly with the inference server!

## Quick Start with VSCode Tunnels

### 1. Start the Inference Server
```bash
cd /workspace/vla_evaluation
python inference_server.py --port=8000
```

The server will start on `0.0.0.0:8000` (already configured for tunneling).

### 2. Forward the Port in VSCode
1. Open the **PORTS** tab in VSCode (bottom panel)
2. Click **Forward a Port**
3. Enter `8000`
4. Set visibility to **Public** (important!)
5. Copy the forwarded URL (e.g., `https://xyz123-8000.use.devtunnels.ms`)

### 3. Test the Connection
```bash
# Test health endpoint
python3 -c "import requests; print(requests.get('https://YOUR-TUNNEL-URL.use.devtunnels.ms/health').json())"

# Example with your actual tunnel:
python3 -c "import requests; print(requests.get('https://c36pk8pn-8000.use.devtunnels.ms/health').json())"
```

## Using the Client with Tunnels

### Option 1: Use the tunnel-aware client
```bash
python client_tunnel.py --tunnel_url=https://c36pk8pn-8000.use.devtunnels.ms
```

### Option 2: Use the regular client with tunnel URL
```bash
python client_example.py --server_url=https://c36pk8pn-8000.use.devtunnels.ms
```

### Option 3: Python code with tunnel
```python
from client_example import LeRobotInferenceClient

# Use your forwarded tunnel URL
client = LeRobotInferenceClient("https://c36pk8pn-8000.use.devtunnels.ms")

# Everything else works the same!
observation = {"observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
result = client.predict(observation)
print(result['actions'])
```

## Common Issues & Solutions

### Issue: Connection refused
**Solution**: Make sure the port visibility is set to **Public** in VSCode

### Issue: 404 Not Found
**Solution**: Verify the tunnel URL is correct (check the PORTS tab)

### Issue: Slow inference
**Solution**: This is expected with tunneling. The latency includes:
- Your machine → VSCode tunnel servers → Your school network → Back
- For local testing, you can still use `localhost:8000` from VSCode terminal

### Issue: Tunnel URL changes
**Solution**: The tunnel URL changes each time you restart VSCode. Options:
1. Use a fixed URL by setting up a persistent tunnel
2. Store the URL in an environment variable each session

## Performance Tips for Tunneling

1. **Batch requests**: Group multiple predictions to reduce round-trip overhead
2. **Use smaller models**: Faster inference = less time waiting through tunnel
3. **Cache results**: Store predictions if you'll need them again
4. **Local testing first**: Test with `localhost:8000` before using tunnel

## Example: Full Workflow

```bash
# Terminal 1: Start server
cd /workspace/vla_evaluation
python inference_server.py --port=8000

# Wait for "Server started on 0.0.0.0:8000"

# Terminal 2: Forward port in VSCode PORTS tab
# Get your tunnel URL (e.g., https://xyz-8000.use.devtunnels.ms)

# Terminal 3: Test it
export TUNNEL_URL="https://xyz-8000.use.devtunnels.ms"
python3 -c "import requests; print(requests.get('$TUNNEL_URL/health').json())"

# Terminal 3: Run client
python client_example.py --server_url=$TUNNEL_URL
```

## Integration with Your Existing Setup

Since your `main.py` Flask server works on port 8080, you can run both:

```bash
# Terminal 1: Your Flask server
python main.py  # Runs on port 8080

# Terminal 2: Inference server
cd vla_evaluation
python inference_server.py --port=8000

# Forward both ports:
# - 8080 → Flask health check
# - 8000 → LeRobot inference
```

## HTTPS with Tunnels

Good news! VSCode tunnels automatically provide HTTPS, so your connections are encrypted even through your school network.

## Persistent Tunnels (Advanced)

To keep the same URL across sessions:

```bash
# Install VS Code CLI
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz

# Create a persistent tunnel
./code tunnel --name my-lerobot-server --accept-server-license-terms
```

Then access via: `https://my-lerobot-server.devtunnels.ms:8000`

## Security Note

Since you're exposing the server publicly through tunnels:
1. Don't commit API keys or tokens
2. Consider adding authentication if running for extended periods
3. Monitor the logs for unexpected requests
4. Stop the server when not in use

## Quick Reference Commands

```bash
# Start server
python inference_server.py --port=8000

# Test from anywhere (replace with your tunnel URL)
curl https://YOUR-TUNNEL.use.devtunnels.ms/health

# Python one-liner test
python3 -c "import requests; r=requests.get('https://YOUR-TUNNEL.use.devtunnels.ms/health'); print(r.json())"

# Run examples through tunnel
python client_example.py --server_url=https://YOUR-TUNNEL.use.devtunnels.ms
```
