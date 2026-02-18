# LeRobot Inference Server

A REST API server for serving LeRobot policy models for easy inference. This server provides a simple HTTP interface to interact with LeRobot models including SmolVLA, ACT, Diffusion Policy, and other supported policies.

## Features

- 🚀 **Easy to Use**: Simple REST API with FastAPI
- 🔌 **Multiple Policy Support**: Works with SmolVLA, ACT, Diffusion, PI0, and more
- 🖼️ **Image Support**: Handle camera observations with base64 encoding
- ⚡ **GPU Acceleration**: Automatic CUDA detection and usage
- 📊 **Model Info Endpoint**: Query model configuration and capabilities
- 🏥 **Health Checks**: Monitor server and model status
- 🔄 **CORS Enabled**: Easy integration with web frontends

## Installation

### 1. Install LeRobot

First, ensure LeRobot is installed:

```bash
cd /workspace/lerobot
pip install -e .[smolvla]
```

For other policy types, use the appropriate extras:
- `[act]` for ACT policy
- `[diffusion]` for Diffusion policy
- `[pi0]` for PI0 policy
- `[async]` for async inference support

### 2. Install Server Dependencies

```bash
cd /workspace/vla_evaluation
pip install -r requirements.txt
```

## Quick Start

### Starting the Server

**Basic usage:**
```bash
python3 inference_server.py --model_id=NLTuan/smolvla_red_block_in_tape --port=8000
```

**With VSCode Port Forwarding (for blocked networks):**
See [VSCODE_TUNNEL_SETUP.md](VSCODE_TUNNEL_SETUP.md) for detailed instructions on using VSCode DevTunnels.
```bash
# Start server
python3 inference_server.py --port=8000

# Forward port 8000 in VSCode PORTS tab (set to Public)
# Then use the tunnel URL with the client:
python client_tunnel.py --tunnel_url=https://YOUR-TUNNEL.use.devtunnels.ms
```

**With specific device:**
```bash
python inference_server.py \
    --model_id=NLTuan/smolvla_red_block_in_tape \
    --device=cuda \
    --port=8000
```

**With environment variables:**
```bash
export MODEL_ID=NLTuan/smolvla_red_block_in_tape
export PORT=8000
export DEVICE=cuda
python inference_server.py
```

### Using the Client

```bash
python client_example.py --server_url=http://localhost:8000
```

Run specific examples:
```bash
# Simple state-only inference
python client_example.py --example=simple

# Image + state inference
python client_example.py --example=image

# Simulated robot control loop
python client_example.py --example=loop
```

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### Endpoints

#### GET `/health`
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

#### GET `/model/info`
Get detailed model information.

**Response:**
```json
{
  "model_id": "NLTuan/smolvla_red_block_in_tape",
  "policy_type": "smolvla",
  "action_dim": 7,
  "chunk_size": 20,
  "max_state_dim": 14,
  "camera_inputs": ["top"],
  "device": "cuda"
}
```

#### POST `/predict`
Run inference on an observation.

**Request:**
```json
{
  "observation": {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "observation.images.top": "base64_encoded_image_string"
  },
  "action_steps": 10
}
```

**Response:**
```json
{
  "actions": [[0.1, 0.2, ...], [0.15, 0.21, ...], ...],
  "action_dim": 7,
  "num_steps": 10,
  "model_info": {
    "model_id": "NLTuan/smolvla_red_block_in_tape",
    "policy_type": "smolvla",
    ...
  }
}
```

## Usage Examples

### Python Client

```python
from client_example import LeRobotInferenceClient
import numpy as np

# Initialize client
client = LeRobotInferenceClient("http://localhost:8000")

# Get model info
info = client.get_model_info()
print(f"Model: {info['model_id']}")

# Simple state observation
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}
result = client.predict(observation)
print(f"Predicted actions: {result['actions']}")

# With image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "observation.images.top": client.encode_image(image)
}
result = client.predict(observation, action_steps=20)
```

### cURL Examples

**Health check:**
```bash
curl http://localhost:8000/health
```

**Model info:**
```bash
curl http://localhost:8000/model/info
```

**Inference:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
  }'
```

### JavaScript/TypeScript

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    observation: {
      'observation.state': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    },
    action_steps: 10
  })
});

const result = await response.json();
console.log('Predicted actions:', result.actions);
```

## Configuration

Edit `server_config.yaml` to customize server settings:

```yaml
model:
  model_id: "NLTuan/smolvla_red_block_in_tape"
  policy_type: "auto"
  device: null  # Auto-detect

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

inference:
  max_batch_size: 1
  timeout: 30
```

## Supported Models

The server supports all LeRobot policy types:

### Vision-Language-Action Models
- **SmolVLA**: `NLTuan/smolvla_red_block_in_tape` or your fine-tuned model
- **xVLA**: Vision-language-action models

### Action Chunking Models
- **ACT**: `lerobot/act_pusht_image`
- **Diffusion Policy**: `lerobot/diffusion_pusht`

### Generalist Models
- **PI0**: `physical-intelligence/pi0-pusht`
- **PI0.5**: Improved PI0 models
- **GR00T**: NVIDIA's generalist robot models

### Other Policies
- **TDMPC**: Temporal Difference Model Predictive Control
- **VQ-BeT**: Vector Quantized Behavior Transformers
- **SAC**: Soft Actor-Critic (for RL environments)

## Deployment

### Docker (Coming Soon)

```bash
docker build -t lerobot-inference-server .
docker run -p 8000:8000 \
  -e MODEL_ID=NLTuan/smolvla_red_block_in_tape \
  -e DEVICE=cuda \
  --gpus all \
  lerobot-inference-server
```

### Production Considerations

1. **Use a production ASGI server**:
   ```bash
   gunicorn inference_server:app \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8000
   ```

2. **Add authentication**: Implement API keys or OAuth

3. **Rate limiting**: Use middleware or reverse proxy

4. **Monitoring**: Add Prometheus metrics, logging aggregation

5. **Load balancing**: Deploy multiple instances behind nginx/traefik

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Use CPU or reduce batch size
```bash
python inference_server.py --device=cpu
```

### Issue: Model not found
**Solution**: Verify model ID and HuggingFace access
```bash
huggingface-cli login
python inference_server.py --model_id=your/model-id
```

### Issue: Port already in use
**Solution**: Use a different port
```bash
python inference_server.py --port=8001
```

### Issue: Slow inference
**Solution**: 
- Ensure GPU is being used: Check `/health` endpoint
- Reduce image resolution
- Use a smaller model or quantized version

## Performance Tips

1. **Use GPU**: Always prefer CUDA for inference
2. **Batch requests**: Group multiple observations when possible
3. **Image compression**: Use appropriate image sizes
4. **Model selection**: Choose appropriate model size for your hardware
5. **Caching**: Consider caching preprocessors for repeated use

## Development

### Running in development mode with auto-reload:
```bash
python inference_server.py --reload
```

### Running tests:
```bash
pytest tests/
```

## Integration with LeRobot's Async Inference

This REST API server complements LeRobot's native gRPC async inference system:

- **REST API** (this server): Best for web services, simple integrations, cross-language support
- **gRPC async**: Best for low-latency real-time robot control, high-frequency updates

You can use both:
```python
# For web/HTTP clients: Use REST API
# For robot control: Use native async inference with gRPC
```

See [LeRobot's async inference docs](https://github.com/huggingface/lerobot/blob/main/lerobot/src/lerobot/async_inference/) for gRPC-based control.

## License

This project follows the same license as LeRobot (Apache 2.0).

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

- **LeRobot Discord**: [Join here](https://discord.gg/s3KuuzsPFb)
- **Issues**: Open an issue on GitHub
- **Docs**: Check [LeRobot documentation](https://github.com/huggingface/lerobot)

## Acknowledgements

Built on top of the excellent [LeRobot](https://github.com/huggingface/lerobot) library by HuggingFace.
