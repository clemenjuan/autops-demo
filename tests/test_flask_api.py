#!/usr/bin/env python3
"""
Flask API Testing Script
Tests Flask endpoints step by step
"""

import requests
import json
import time
import os

BASE_URL = "http://localhost:5000"

def test_server_startup():
    """Test if Flask server can start"""
    print("Step 1: Testing Flask server startup...")
    try:
        # Try to import and start the app
        from app import app
        print("✅ Flask app imports successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False

def test_status_endpoint():
    """Test the status endpoint"""
    print("\nStep 2: Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Status endpoint working")
            data = response.json()
            print(f"   LLM Status: {data.get('llm_status', 'Unknown')}")
            print(f"   Tools loaded: {len(data.get('tools', {}))}")
            return True
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server - is it running?")
        return False
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\nStep 3: Testing chat endpoint...")
    try:
        data = {
            "message": "Hello, can you help me with satellite operations?",
            "context": {}
        }
        response = requests.post(f"{BASE_URL}/api/chat", json=data, timeout=30)
        if response.status_code == 200:
            print("✅ Chat endpoint working")
            result = response.json()
            print(f"   AI Response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False

def test_tool_endpoint():
    """Test a tool endpoint"""
    print("\nStep 4: Testing tool endpoint...")
    try:
        data = {"satellite_id": "SAT-123"}
        response = requests.post(f"{BASE_URL}/api/tools/collision_avoidance", json=data, timeout=30)
        if response.status_code == 200:
            print("✅ Tool endpoint working")
            result = response.json()
            print(f"   Tool result: {result.get('satellite_id', 'Unknown')}")
            return True
        else:
            print(f"❌ Tool endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Tool endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("Flask API Testing")
    print("=" * 30)
    
    # Test 1: Import
    if not test_server_startup():
        print("\n❌ Cannot proceed - Flask app import failed")
        return
    
    # Test 2: Status endpoint
    if not test_status_endpoint():
        print("\n❌ Status endpoint failed - check if server is running")
        return
    
    # Test 3: Chat endpoint
    if not test_chat_endpoint():
        print("\n❌ Chat endpoint failed")
        return
    
    # Test 4: Tool endpoint
    if not test_tool_endpoint():
        print("\n❌ Tool endpoint failed")
        return
    
    print("\n" + "=" * 30)
    print("✅ All Flask API tests passed!")
    print("Ready to create web interface for operators.")

if __name__ == "__main__":
    main()
