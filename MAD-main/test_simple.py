#!/usr/bin/env python3
"""
Simple test to check litgpt response limits
"""
import requests

def test_simple_prompt():
    """Test with a very simple prompt"""
    
    # Test 1: Very short prompt
    print("=== Test 1: Short Prompt ===")
    short_prompt = "Write a 5-sentence story about a cat."
    
    data = {
        "prompt": short_prompt,
        "max_new_tokens": 16384,  # Match server limit
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response length: {len(result.get('output', ''))}")
            print(f"Response: {result.get('output', '')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Medium prompt
    print("=== Test 2: Medium Prompt ===")
    medium_prompt = """You are a medical expert. Answer this question in 3 sentences:

Question: What causes diabetes?
Answer:"""
    
    data = {
        "prompt": medium_prompt,
        "max_new_tokens": 16384,  # Match server limit
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Response length: {len(result.get('output', ''))}")
            print(f"Response: {result.get('output', '')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_simple_prompt()
