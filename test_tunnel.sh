#!/bin/bash
# Quick test script for tunnel connection
# Usage: ./test_tunnel.sh https://xyz-8000.use.devtunnels.ms

TUNNEL_URL=${1:-""}

if [ -z "$TUNNEL_URL" ]; then
    echo "Usage: ./test_tunnel.sh <tunnel_url>"
    echo "Example: ./test_tunnel.sh https://c36pk8pn-8000.use.devtunnels.ms"
    exit 1
fi

echo "Testing tunnel connection to: $TUNNEL_URL"
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
python3 -c "import requests; import json; r=requests.get('$TUNNEL_URL/health'); print(json.dumps(r.json(), indent=2))"

if [ $? -eq 0 ]; then
    echo "✓ Health check passed!"
else
    echo "✗ Health check failed!"
    exit 1
fi

echo ""
echo "2. Testing /model/info endpoint..."
python3 -c "import requests; import json; r=requests.get('$TUNNEL_URL/model/info'); print(json.dumps(r.json(), indent=2))"

if [ $? -eq 0 ]; then
    echo "✓ Model info retrieved!"
else
    echo "✗ Model info failed!"
    exit 1
fi

echo ""
echo "3. Testing inference endpoint..."
python3 -c "
import requests
import json

url = '$TUNNEL_URL/predict'
payload = {
    'observation': {
        'observation.state': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
}

r = requests.post(url, json=payload)
result = r.json()
print(f\"✓ Predicted {result['num_steps']} action steps\")
print(f\"  Action shape: ({result['num_steps']}, {result['action_dim']})\")
print(f\"  First action: {result['actions'][0][:3]}...\")
"

if [ $? -eq 0 ]; then
    echo "✓ Inference test passed!"
else
    echo "✗ Inference test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo "Your tunnel is working correctly."
echo "You can now use this URL with the client:"
echo "  python client_example.py --server_url=$TUNNEL_URL"
echo "  python client_tunnel.py --tunnel_url=$TUNNEL_URL"
