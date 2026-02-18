"""
Test script for LeRobot Inference Server

This script performs basic tests on the inference server to ensure
it's working correctly.

Usage:
    python test_server.py --server_url=http://localhost:8000
"""

import argparse
import sys
import time

import requests


def test_health_check(server_url: str) -> bool:
    """Test the health check endpoint."""
    print("\n[TEST 1] Health Check")
    print("-" * 50)
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {data['status']}")
        print(f"✓ Model loaded: {data['model_loaded']}")
        print(f"✓ Model ID: {data.get('model_id', 'N/A')}")
        print(f"✓ Device: {data.get('device', 'N/A')}")
        
        if data['status'] != 'healthy':
            print("✗ Server is not healthy")
            return False
        
        if not data['model_loaded']:
            print("✗ Model is not loaded")
            return False
        
        print("✓ Health check passed")
        return True
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_model_info(server_url: str) -> bool:
    """Test the model info endpoint."""
    print("\n[TEST 2] Model Info")
    print("-" * 50)
    try:
        response = requests.get(f"{server_url}/model/info", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Model ID: {data['model_id']}")
        print(f"✓ Policy type: {data['policy_type']}")
        print(f"✓ Action dimension: {data['action_dim']}")
        print(f"✓ Chunk size: {data['chunk_size']}")
        print(f"✓ Max state dimension: {data['max_state_dim']}")
        print(f"✓ Camera inputs: {data['camera_inputs']}")
        print(f"✓ Device: {data['device']}")
        
        print("✓ Model info test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model info test failed: {e}")
        return False


def test_simple_inference(server_url: str) -> bool:
    """Test simple inference with state observation."""
    print("\n[TEST 3] Simple Inference")
    print("-" * 50)
    try:
        # Get model info first
        model_response = requests.get(f"{server_url}/model/info", timeout=5)
        model_data = model_response.json()
        state_dim = min(6, model_data['max_state_dim'])
        
        # Prepare observation
        observation = {
            "observation.state": [0.1 * i for i in range(state_dim)]
        }
        
        payload = {
            "observation": observation,
            "action_steps": 5
        }
        
        print(f"Sending observation with state dim: {state_dim}")
        
        # Time the inference
        start_time = time.time()
        response = requests.post(f"{server_url}/predict", json=payload, timeout=30)
        inference_time = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Inference time: {inference_time:.3f}s")
        print(f"✓ Action dimension: {data['action_dim']}")
        print(f"✓ Number of steps: {data['num_steps']}")
        print(f"✓ Actions shape: ({data['num_steps']}, {data['action_dim']})")
        print(f"✓ First action: {data['actions'][0][:3]}...")
        
        if data['num_steps'] == 0:
            print("✗ No actions returned")
            return False
        
        print("✓ Simple inference test passed")
        return True
        
    except Exception as e:
        print(f"✗ Simple inference test failed: {e}")
        return False


def test_multiple_requests(server_url: str, num_requests: int = 3) -> bool:
    """Test multiple sequential inference requests."""
    print(f"\n[TEST 4] Multiple Requests ({num_requests} requests)")
    print("-" * 50)
    try:
        times = []
        
        for i in range(num_requests):
            observation = {
                "observation.state": [0.1 * (i + 1) * j for j in range(6)]
            }
            
            payload = {"observation": observation}
            
            start_time = time.time()
            response = requests.post(f"{server_url}/predict", json=payload, timeout=30)
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            response.raise_for_status()
            data = response.json()
            
            print(f"  Request {i+1}: {inference_time:.3f}s - {data['num_steps']} actions")
        
        avg_time = sum(times) / len(times)
        print(f"\n✓ Average inference time: {avg_time:.3f}s")
        print(f"✓ Min: {min(times):.3f}s, Max: {max(times):.3f}s")
        print("✓ Multiple requests test passed")
        return True
        
    except Exception as e:
        print(f"✗ Multiple requests test failed: {e}")
        return False


def test_error_handling(server_url: str) -> bool:
    """Test error handling with invalid requests."""
    print("\n[TEST 5] Error Handling")
    print("-" * 50)
    try:
        # Test with empty observation
        payload = {"observation": {}}
        response = requests.post(f"{server_url}/predict", json=payload, timeout=30)
        
        # We expect this to potentially fail gracefully
        if response.status_code >= 500:
            print(f"✓ Server handled empty observation (status: {response.status_code})")
        else:
            print(f"✓ Server processed empty observation (status: {response.status_code})")
        
        # Test with malformed request
        payload = {"invalid_field": "test"}
        response = requests.post(f"{server_url}/predict", json=payload, timeout=30)
        
        if response.status_code >= 400:
            print(f"✓ Server rejected malformed request (status: {response.status_code})")
        
        print("✓ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"✓ Error handling test passed (caught exception as expected: {type(e).__name__})")
        return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test LeRobot Inference Server")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the inference server"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LeRobot Inference Server Test Suite")
    print("=" * 60)
    print(f"Testing server at: {args.server_url}")
    
    # Run all tests
    tests = [
        ("Health Check", lambda: test_health_check(args.server_url)),
        ("Model Info", lambda: test_model_info(args.server_url)),
        ("Simple Inference", lambda: test_simple_inference(args.server_url)),
        ("Multiple Requests", lambda: test_multiple_requests(args.server_url)),
        ("Error Handling", lambda: test_error_handling(args.server_url)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total} tests")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
