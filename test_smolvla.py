#!/usr/bin/env python3
"""
Quick test of the inference server with proper SmolVLA inputs
This shows how to prepare observations with camera images and task instructions
"""

import sys
sys.path.insert(0, '/workspace/lerobot/src')

import base64
import io
import requests
import numpy as np
from PIL import Image


def encode_image(image_array: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def main():
    # Server URL (use localhost for testing, or your tunnel URL)
    server_url = "http://localhost:8000"
    
    print("Testing LeRobot Inference Server with SmolVLA")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking server health...")
    health = requests.get(f"{server_url}/health").json()
    print(f"   Status: {health['status']}")
    print(f"   Model: {health['model_id']}")
    print(f"   Device: {health['device']}")
    
    # 2. Get model info
    print("\n2. Getting model info...")
    info = requests.get(f"{server_url}/model/info").json()
    print(f"   Policy type: {info['policy_type']}")
    print(f"   Action dim: {info['action_dim']}")
    print(f"   Chunk size: {info['chunk_size']}")
    print(f"   Camera inputs: {info['camera_inputs']}")
    
    # 3. Prepare observation with camera images
    print("\n3. Preparing observation with camera images...")
    
    # Create dummy camera images (replace with real camera images in practice)
    # SmolVLA expects at least one camera image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Prepare observation
    observation = {
        "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Robot state (must match model's action_dim)
        "task": "pick up the red block",  # Task instruction
        # Add camera images (match the camera names from model info)
        "observation.images.camera1": encode_image(dummy_image),
    }
    
    print(f"   Task: {observation['task']}")
    print(f"   State dim: {len(observation['observation.state'])}")
    print(f"   Images: {[k for k in observation.keys() if 'images' in k]}")
    
    # 4. Run inference
    print("\n4. Running inference...")
    response = requests.post(
        f"{server_url}/predict",
        json={"observation": observation}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success!")
        print(f"   Predicted {result['num_steps']} action steps")
        print(f"   Action dimension: {result['action_dim']}")
        print(f"   First action: {result['actions'][0][:3]}...")
    else:
        print(f"   ✗ Error: {response.json()}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nNow you can forward port 8000 in VSCode and use your tunnel URL!")


if __name__ == "__main__":
    main()
