#!/usr/bin/env python3
"""
Simple test script to verify the HoldOn server is working.
"""

import requests
import time
import sys

def test_server():
    """Test if the server is running and responding."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing HoldOn API Server...")
    print(f"📍 Server URL: {base_url}\n")
    
    # Test 1: Health check
    print("1️⃣ Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Server is running!")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ⚠️  Server responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server. Is it running?")
        print("   💡 Start the server with: python3 -m src.api.main")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: API docs
    print("\n2️⃣ Testing API documentation endpoint...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("   ✅ API documentation is available!")
            print(f"   📚 View at: {base_url}/docs")
        else:
            print(f"   ⚠️  Docs endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Could not access docs: {e}")
    
    print("\n✅ Server test complete!")
    print(f"\n📚 Full API documentation: {base_url}/docs")
    print(f"🔗 Alternative docs: {base_url}/redoc")
    return True

if __name__ == "__main__":
    # Wait a bit for server to start if just launched
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        print("⏳ Waiting 3 seconds for server to start...")
        time.sleep(3)
    
    success = test_server()
    sys.exit(0 if success else 1)

