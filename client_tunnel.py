"""
Tunnel-aware client for LeRobot Inference Server

This client is optimized for use with VSCode DevTunnels or other
tunneling solutions where you need to use HTTPS forwarded URLs.

Usage:
    # Auto-detect tunnel URL from VSCode
    python client_tunnel.py
    
    # Specify tunnel URL
    python client_tunnel.py --tunnel_url=https://xyz-8000.use.devtunnels.ms
    
    # Test with local server too
    python client_tunnel.py --local
"""

import argparse
import base64
import io
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import requests
from PIL import Image


class LeRobotInferenceClient:
    """Client for LeRobot Inference Server."""
    
    def __init__(self, server_url: str):
        """
        Initialize client.
        
        Args:
            server_url: Base URL of the inference server
        """
        self.server_url = server_url.rstrip('/')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        response = requests.get(f"{self.server_url}/model/info", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = requests.get(f"{self.server_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        observation: Dict[str, Any],
        action_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run inference on an observation.
        
        Args:
            observation: Observation dict with state and/or images
            action_steps: Number of action steps to predict
            
        Returns:
            Prediction result with actions
        """
        payload = {"observation": observation}
        if action_steps is not None:
            payload["action_steps"] = action_steps
        
        response = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    @staticmethod
    def decode_image(base64_str: str) -> np.ndarray:
        """Decode base64 string to numpy image."""
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)


def example_simple_state_inference(client: LeRobotInferenceClient):
    """Example: Simple state-only inference."""
    print("\n" + "="*70)
    print("Example 1: Simple State Inference")
    print("="*70)
    
    observation = {
        "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "task": "pick up the red block"
    }
    
    print(f"Input state: {observation['observation.state']}")
    print(f"Task: {observation['task']}")
    
    try:
        result = client.predict(observation, action_steps=5)
        print(f"\nPredicted {result['num_steps']} action steps")
        print(f"Action dimension: {result['action_dim']}")
        print(f"First action: {result['actions'][0]}")
    except Exception as e:
        print(f"Error: {e}")


def example_image_inference(client: LeRobotInferenceClient):
    """Example: Image + state inference."""
    print("\n" + "="*70)
    print("Example 2: Image + State Inference")
    print("="*70)
    
    # Create dummy images (SmolVLA needs camera1)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    observation = {
        "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "task": "pick up the red block",
        "observation.images.camera1": client.encode_image(image)
    }
    
    print(f"Input: state + image ({image.shape})")
    
    try:
        result = client.predict(observation, action_steps=10)
        print(f"\nPredicted {result['num_steps']} action steps")
        print(f"First few actions: {result['actions'][:3]}")
    except Exception as e:
        print(f"Error: {e}")


def example_real_robot_loop(client: LeRobotInferenceClient):
    """Example: Simulated robot control loop."""
    print("\n" + "="*70)
    print("Example 3: Robot Control Loop (Simulated)")
    print("="*70)
    
    print("Simulating 3 control steps...")
    
    for step in range(3):
        # Simulated observation
        state = [0.1 * step, 0.2 * step, 0.3, 0.4, 0.5, 0.6]
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        observation = {
            "observation.state": state,
            "task": "pick up the red block",
            "observation.images.camera1": client.encode_image(image)
        }
        
        try:
            result = client.predict(observation, action_steps=10)
            action = result['actions'][0]
            print(f"  Step {step + 1}: predicted action = {action[:3]}...")
        except Exception as e:
            print(f"  Step {step + 1}: Error - {e}")


def example_batch_inference(client: LeRobotInferenceClient):
    """Example: Batch inference (sequential)."""
    print("\n" + "="*70)
    print("Example 4: Batch Inference")
    print("="*70)
    
    observations = [
        {
            "observation.state": [i * 0.1, i * 0.2, 0.3, 0.4, 0.5, 0.6],
            "task": "pick up the red block"
        }
        for i in range(3)
    ]
    
    print(f"Processing {len(observations)} observations...")
    
    for i, obs in enumerate(observations):
        try:
            result = client.predict(obs, action_steps=5)
            print(f"  Observation {i + 1}: predicted {result['num_steps']} actions")
        except Exception as e:
            print(f"  Observation {i + 1}: Error - {e}")


def detect_vscode_tunnel_url(port: int = 8000) -> str:
    """
    Try to auto-detect VSCode tunnel URL for the given port.
    
    This is best-effort and may not work in all environments.
    """
    # Check if we're in a VSCode terminal
    if not os.getenv("TERM_PROGRAM") == "vscode":
        return None
    
    try:
        # Try to read VSCode tunnel info (this is a simplified approach)
        # In practice, you'd need to check VSCode's internal state
        # For now, we'll just return None and require manual input
        pass
    except Exception:
        pass
    
    return None


def test_connection(url: str) -> bool:
    """Test if we can connect to the server."""
    try:
        client = LeRobotInferenceClient(url)
        info = client.get_model_info()
        print(f"✓ Connected successfully!")
        print(f"  Model: {info['model_id']}")
        print(f"  Device: {info['device']}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def get_tunnel_url_from_user() -> str:
    """Interactively get tunnel URL from user."""
    print("\n" + "="*70)
    print("VSCode Port Forwarding Setup")
    print("="*70)
    print("""
To forward a port in VSCode:
1. Open the PORTS tab (bottom panel)
2. Click 'Forward a Port'
3. Enter '8000'
4. Set visibility to 'Public'
5. Copy the forwarded URL

Example URL: https://c36pk8pn-8000.use.devtunnels.ms
""")
    
    url = input("Enter your forwarded tunnel URL: ").strip()
    
    # Clean up the URL
    url = url.rstrip('/')
    
    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        print("Warning: URL should start with http:// or https://")
        url = 'https://' + url
    
    return url


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tunnel-aware LeRobot Inference Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (will prompt for tunnel URL)
  python client_tunnel.py
  
  # Specify tunnel URL directly
  python client_tunnel.py --tunnel_url=https://xyz-8000.use.devtunnels.ms
  
  # Use local server
  python client_tunnel.py --local
  
  # Run specific example
  python client_tunnel.py --tunnel_url=https://xyz-8000.use.devtunnels.ms --example=simple
        """
    )
    parser.add_argument(
        "--tunnel_url",
        type=str,
        help="VSCode tunnel URL (e.g., https://xyz-8000.use.devtunnels.ms)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use localhost instead of tunnel (for same-machine testing)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["simple", "image", "loop", "batch", "all"],
        default="all",
        help="Which example to run"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test connection, don't run examples"
    )
    
    args = parser.parse_args()
    
    # Determine server URL
    if args.local:
        server_url = f"http://localhost:{args.port}"
        print(f"Using local server: {server_url}")
    elif args.tunnel_url:
        server_url = args.tunnel_url.rstrip('/')
        print(f"Using tunnel URL: {server_url}")
    else:
        # Try to auto-detect
        server_url = detect_vscode_tunnel_url(args.port)
        if not server_url:
            # Interactive mode
            server_url = get_tunnel_url_from_user()
    
    # Test connection
    print(f"\nTesting connection to {server_url}...")
    if not test_connection(server_url):
        print("\n❌ Could not connect to server!")
        print("\nTroubleshooting:")
        print("1. Make sure the inference server is running:")
        print("   python inference_server.py --port=8000")
        print("2. Verify the port is forwarded in VSCode PORTS tab")
        print("3. Check that visibility is set to 'Public'")
        print("4. Try accessing the health endpoint in a browser:")
        print(f"   {server_url}/health")
        sys.exit(1)
    
    if args.test_only:
        print("\n✓ Connection test passed!")
        sys.exit(0)
    
    # Create client
    print(f"\n{'='*70}")
    print("Running Examples with Tunnel Connection")
    print('='*70)
    
    client = LeRobotInferenceClient(server_url)
    
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
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print(f"\nYour tunnel URL: {server_url}")
    print("Save this URL to reuse in your code!")


if __name__ == "__main__":
    main()
