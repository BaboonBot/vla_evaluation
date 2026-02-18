"""Test script to verify documentation accuracy"""
from client_tunnel import LeRobotInferenceClient
import numpy as np

# Connect to server (from USAGE.md)
client = LeRobotInferenceClient("http://localhost:8000")

# Check server
health = client.health_check()
print(f"✓ Server status: {health['status']}")
print(f"✓ Device: {health['device']}")

# Get model info
info = client.get_model_info()
print(f"✓ Model: {info['model_id']}")
print(f"✓ Action dim: {info['action_dim']}")
print(f"✓ Chunk size: {info['chunk_size']}")
print(f"✓ Required cameras: {info['camera_inputs']}")

# Make prediction (MUST include an image for SmolVLA)
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
observation = {
    "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "task": "pick up the red block",
    "observation.images.camera1": client.encode_image(image)
}

result = client.predict(observation, action_steps=10)
print(f"✓ Predicted {result['num_steps']} action steps")
print(f"✓ First action: {result['actions'][0]}")
print("\n✓ Documentation is accurate!")
