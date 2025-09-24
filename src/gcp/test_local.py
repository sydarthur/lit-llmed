#!/usr/bin/env python3
"""
Local testing script for the GCP Cloud Run application
"""

import requests
import json
import time
import subprocess
import signal
import os
import sys
from multiprocessing import Process

def start_server():
    """Start the Flask server in a separate process"""
    os.chdir(os.path.dirname(__file__))
    subprocess.run([sys.executable, "app.py"])

def test_endpoints():
    """Test all available endpoints"""
    base_url = "http://localhost:8080"
    
    print("🧪 Testing GCP Cloud Run application locally...")
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server failed to start")
        return False
    
    print("✅ Server is running")
    
    # Test health check endpoint
    print("\n1️⃣ Testing health check endpoint (GET /)...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
        print("   ✅ Health check passed")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test status endpoint
    print("\n2️⃣ Testing status endpoint (GET /api/status)...")
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Status endpoint passed")
    except Exception as e:
        print(f"   ❌ Status endpoint failed: {e}")
        return False
    
    # Test process endpoint
    print("\n3️⃣ Testing process endpoint (POST /api/process)...")
    try:
        test_data = {"message": "test", "data": [1, 2, 3]}
        response = requests.post(f"{base_url}/api/process", json=test_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
        print("   ✅ Process endpoint passed")
    except Exception as e:
        print(f"   ❌ Process endpoint failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The application is ready for Cloud Run deployment.")
    return True

if __name__ == "__main__":
    print("🚀 Starting local test of GCP Cloud Run application")
    print("📋 This will:")
    print("   1. Start the Flask server on localhost:8080")
    print("   2. Test all endpoints")
    print("   3. Provide deployment readiness status")
    print()
    
    # Start server in background
    server_process = Process(target=start_server)
    server_process.start()
    
    try:
        time.sleep(2)  # Give server time to start
        success = test_endpoints()
        
        if success:
            print("\n✅ Application is ready for Cloud Run!")
            print("💡 Next steps:")
            print("   1. Install GCP CLI: https://cloud.google.com/sdk/docs/install")
            print("   2. Run: gcloud auth login")
            print("   3. Run: gcloud config set project YOUR_PROJECT_ID")
            print("   4. Deploy: cd ../../config && ./deploy.sh YOUR_PROJECT_ID")
        else:
            print("\n❌ Some tests failed. Please check the application before deploying.")
            
    finally:
        # Clean up
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()