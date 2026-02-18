#!/usr/bin/env python3
"""
Better test using actual dataset to see proper data format
This loads a real frame from the dataset to test the server
"""

import sys
sys.path.insert(0, '/workspace/lerobot/src')

import base64
import io
import requests
import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a torch tensor (C, H, W) to base64 string."""
    # Convert CHW to HWC for PIL
    if tensor.dim() == 3:
        img_array = tensor.permute(1, 2, 0).numpy()
    else:
        img_array = tensor.numpy()
    
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        # Assume values are in [0, 1] or [0, 255]
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    image = Image.fromarray(img_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def main():
    server_url = "http://localhost:8000"
    
    print("Loading dataset to get real examples...")
    print("=" * 60)
    
    # Load the actual dataset
    DATASET_ID = "ThavT/red_block_in_tape"
    dataset = LeRobotDataset(DATASET_ID)
    
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    print(f"  Features: {list(dataset.features.keys())}")
    
    # Get a sample frame
    frame = dataset[0]
    
    print(f"\n✓ Sample frame keys: {list(frame.keys())}")
    
    # Find image keys
    image_keys = [k for k in frame.keys() if k.startswith("observation.images.")]
    print(f"✓ Image keys: {image_keys}")
    
    # Check state
    if "observation.state" in frame:
        print(f"✓ State shape: {frame['observation.state'].shape}")
    
    # Check task
    if "task" in frame:
        print(f"✓ Task: {frame['task']}")
    
    print("\n" + "=" * 60)
    print("Testing inference server...")
    print("=" * 60)
    
    # 1. Check health
    health = requests.get(f"{server_url}/health").json()
    print(f"\n✓ Server status: {health['status']}")
    print(f"  Model: {health['model_id']}")
    
    # 2. Prepare observation from dataset
    observation = {}
    
    # Add state
    if "observation.state" in frame:
        observation["observation.state"] = frame["observation.state"].tolist()
    
    # Add task
    if "task" in frame:
        observation["task"] = frame["task"]
    
    # Add at least one camera image
    for img_key in image_keys[:1]:  # Just use first camera
        print(f"\n✓ Encoding image: {img_key}")
        img_tensor = frame[img_key]
        print(f"  Original tensor shape: {img_tensor.shape}")
        observation[img_key] = tensor_to_base64(img_tensor)
        print(f"  Base64 length: {len(observation[img_key])}")
    
    print(f"\n✓ Observation prepared:")
    print(f"  Keys: {list(observation.keys())}")
    
    # 3. Run inference
    print("\n" + "-" * 60)
    print("Running inference...")
    response = requests.post(
        f"{server_url}/predict",
        json={"observation": observation},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ SUCCESS!")
        print(f"  Predicted {result['num_steps']} action steps")
        print(f"  Action dimension: {result['action_dim']}")
        print(f"  First action: {result['actions'][0][:3]}...")
        print(f"  Last action:  {result['actions'][-1][:3]}...")
    else:
        print(f"\n✗ Error {response.status_code}:")
        print(f"  {response.json()}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nYou can now use the server with VSCode port forwarding!")
    print("Just encode your camera images to base64 and send them.")


if __name__ == "__main__":
    main()
