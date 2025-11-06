#!/usr/bin/env python3
"""
LLM Integration Test
Tests Ollama connection first, then OpenAI fallback if needed
"""

import asyncio
import os
import sys

def test_imports():
    """Test if we can import the LLM interface"""
    print("Step 1: Testing imports...")
    try:
        from agent.llm_interface import LLMInterface
        print("✅ LLMInterface imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ollama_connection():
    """Test if Ollama is available"""
    print("\nStep 2: Testing Ollama connection...")
    try:
        import ollama
        ollama_host = os.getenv('OLLAMA_HOST', 'https://ollama.sps.ed.tum.de')
        print(f"   Trying to connect to: {ollama_host}")
        
        # Try to list models with timeout
        import requests
        response = requests.get(f"{ollama_host}/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama server is available")
            return True
        else:
            print(f"❌ Ollama server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_openai_setup():
    """Test if OpenAI is configured"""
    print("\nStep 3: Testing OpenAI setup...")
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OpenAI API key is configured")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            print("✅ OpenAI client created successfully")
            return True
        except Exception as e:
            print(f"❌ OpenAI client creation failed: {e}")
            return False
    else:
        print("❌ OpenAI API key not found")
        return False

async def test_llm_interface():
    """Test the LLM interface with a simple prompt"""
    print("\nStep 4: Testing LLM interface...")
    
    try:
        from agent.llm_interface import LLMInterface
        llm = LLMInterface(role="general")
        
        simple_prompt = "Say hello and confirm you are working. Keep it short."
        print("   Sending simple test prompt...")
        
        response = await llm.reason(simple_prompt)
        print("✅ LLM interface working!")
        print(f"   Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ LLM interface test failed: {e}")
        return False

async def main():
    """Run all tests step by step"""
    print("Satellite Operations Agent - LLM Integration Test")
    print("=" * 50)
    
    # Step 1: Test imports
    if not test_imports():
        print("\n❌ Cannot proceed - import failed")
        return
    
    # Step 2: Test Ollama
    ollama_available = test_ollama_connection()
    
    # Step 3: Test OpenAI
    openai_available = test_openai_setup()
    
    if not ollama_available and not openai_available:
        print("\n❌ Neither Ollama nor OpenAI is available!")
        return
    
    # Step 4: Test LLM interface
    await test_llm_interface()
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
