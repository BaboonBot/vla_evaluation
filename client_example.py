"""
Example client for LeRobot Inference Server

This script demonstrates how to connect to the inference server
and make prediction requests.

Usage:
    python client_example.py --server_url=http://localhost:8000
"""

import argparse
import base64
import io
import json
from typing import Any, Dict, List

import numpy as np
import requests
from PIL import Image


class LeRobotInferenceClient:
    """Client for LeRobot Inference Server."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            server_url: URL of the inference server
        """
        self.server_url = server_url.rstrip("/")
        self._check_server_health()
    
    def _check_server_health(self):
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if not health.get("model_loaded"):
                print("Warning: Server is running but model is not loaded")
            else:
                print(f"Connected to server. Model: {health.get('model_id')}, Device: {health.get('device')}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to server: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from the server."""
        response = requests.get(f"{self.server_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        observation: Dict[str, Any],
        action_steps: int = None
    ) -> Dict[str, Any]:
        """
        Send inference request to the server.
        
        Args:
            observation: Observation dictionary
            action_steps: Number of action steps to predict
            
        Returns:
            Response dictionary with predicted actions
        """
        payload = {
            "observation": observation,
            "action_steps": action_steps
        }
        
        response = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def encode_image(image: np.ndarray or Image.Image) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image as numpy array or PIL Image
            
        Returns:
            Base64 encoded image string
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    
    @staticmethod
    def decode_image(encoded_str: str) -> Image.Image:
        """
        Decode base64 string to PIL Image.
        
        Args:
            encoded_str: Base64 encoded image string
            
        Returns:
            PIL Image
        """
        image_data = base64.b64decode(encoded_str)
        return Image.open(io.BytesIO(image_data))


def example_simple_state_inference(client: LeRobotInferenceClient):
    """Example with simple state observation (no images)."""
    print("\n" + "="*80)
    print("Example 1: Simple State Observation")
    print("="*80)
    
    # Create a simple observation with robot state
    observation = {
        "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    
    print(f"Sending observation: {observation}")
    
    # Get prediction
    result = client.predict(observation)
    
    print(f"\nPrediction result:")
    print(f"  Actions shape: ({result['num_steps']}, {result['action_dim']})")
    print(f"  First action: {result['actions'][0]}")
    print(f"  Model info: {result['model_info']['model_id']}")


def example_image_inference(client: LeRobotInferenceClient):
    """Example with image and state observation."""
    print("\n" + "="*80)
    print("Example 2: Image + State Observation")
    print("="*80)
    
    # Create a dummy image (in practice, this would come from a camera)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    encoded_image = client.encode_image(dummy_image)
    
    # Create observation with image and state
    observation = {
        "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "observation.images.top": encoded_image
    }
    
    print(f"Sending observation with image (size: {dummy_image.shape}) and state")
    
    # Get prediction
    result = client.predict(observation, action_steps=10)
    
    print(f"\nPrediction result:")
    print(f"  Actions shape: ({result['num_steps']}, {result['action_dim']})")
    print(f"  Number of predicted steps: {result['num_steps']}")
    print(f"  Actions: {np.array(result['actions']).shape}")


def example_real_robot_loop(client: LeRobotInferenceClient):
    """Example of a real robot control loop."""
    print("\n" + "="*80)
    print("Example 3: Simulated Robot Control Loop")
    print("="*80)
    
    # Get model info
    model_info = client.get_model_info()
    print(f"Model: {model_info['model_id']}")
    print(f"Action dimension: {model_info['action_dim']}")
    print(f"Chunk size: {model_info['chunk_size']}")
    
    # Simulate robot control loop
    num_steps = 5
    current_state = [0.0] * model_info['max_state_dim']
    
    print(f"\nRunning {num_steps} control steps...")
    for step in range(num_steps):
        # Create observation (in practice, get from robot sensors)
        observation = {
            "observation.state": current_state[:model_info['max_state_dim']]
        }
        
        # Get action prediction
        result = client.predict(observation)
        actions = result['actions']
        
        # Execute first action (in practice, send to robot)
        next_action = actions[0] if actions else [0.0] * model_info['action_dim']
        
        print(f"Step {step + 1}: Predicted {len(actions)} actions, executing: {next_action[:3]}...")
        
        # Update state (in practice, read from robot)
        current_state = [s + 0.01 * a for s, a in zip(current_state, next_action + [0.0] * (len(current_state) - len(next_action)))]
    
    print("Control loop completed!")


def example_batch_inference(client: LeRobotInferenceClient):
    """Example of processing multiple observations."""
    print("\n" + "="*80)
    print("Example 4: Multiple Inference Requests")
    print("="*80)
    
    # Create multiple observations
    observations = [
        {"observation.state": [0.1 * i] * 6}
        for i in range(3)
    ]
    
    print(f"Processing {len(observations)} observations...")
    
    results = []
    for i, obs in enumerate(observations):
        result = client.predict(obs)
        results.append(result)
        print(f"  Observation {i+1}: Got {result['num_steps']} action steps")
    
    print(f"\nProcessed {len(results)} observations successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LeRobot Inference Client Example")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the inference server"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["simple", "image", "loop", "batch", "all"],
        default="all",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    # Create client
    print(f"Connecting to server at {args.server_url}...")
    client = LeRobotInferenceClient(args.server_url)
    
    # Run examples
    examples = {
        "simple": example_simple_state_inference,
        "image": example_image_inference,
        "loop": example_real_robot_loop,
        "batch": example_batch_inference,
    }
    
    if args.example == "all":
        for name, func in examples.items():
            try:
                func(client)
            except Exception as e:
                print(f"Error in example '{name}': {e}")
    else:
        examples[args.example](client)
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
