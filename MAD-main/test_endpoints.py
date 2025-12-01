#!/usr/bin/env python3
"""
Test if litgpt endpoints are ready and responding
"""
import requests
import time
import json

def test_endpoint(port, name):
    """Test if an endpoint is responding"""
    url = f"http://localhost:{port}/predict"
    
    # Simple test prompt
    data = {
        "prompt": "Say 'Hello World' in one sentence.",
        "max_new_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        print(f"🧪 Testing {name} (port {port})...")
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {name} is working!")
            print(f"   Response: {result.get('output', 'No output')}")
            return True
        else:
            print(f"❌ {name} returned status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ {name} timed out (still loading?)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"🔌 {name} connection failed")
        return False
    except Exception as e:
        print(f"❌ {name} error: {e}")
        return False

def main():
    """Test all three endpoints"""
    print("🚀 Testing litgpt endpoints...")
    
    endpoints = [
        (8000, "Agent A"),
        (8001, "Agent B"), 
        (8003, "Judge")
    ]
    
    # Try each endpoint
    for port, name in endpoints:
        test_endpoint(port, name)
        print()
    
    print("🎯 All tests completed!")

if __name__ == "__main__":
    main()
