import os
import asyncio
import json
import ollama
import requests
from typing import Optional, Dict, Any

# Configuration for LLM models
LLM_CONFIG = {
    "reasoning": {
        "ollama_model": "deepseek-r1:70b",
        "openai_model": "gpt-4",
        "openai_temp": 0.3,
        "system_prompt": "You are an advanced satellite operations reasoning agent. You excel at complex reasoning, multi-step analysis, and decision-making for satellite operations. Provide clear, step-by-step reasoning and justify your recommendations."
    },
    "general": {
        "ollama_model": "llama3.1:8b",
        "openai_model": "gpt-3.5-turbo",
        "openai_temp": 0.5,
        "system_prompt": "You are a professional satellite operations assistant. You help with quick task understanding, categorization, and clear communication. Always be direct, concise, and professional."
    },
    "default": {
        "system_prompt": "You are a professional satellite operations assistant. You are helpful, professional, and provide clear, concise responses."
    }
}

class LLMInterface:
    def __init__(self, preferred_model="auto", role="general"):
        self.role = role
        self.preferred_model = preferred_model
        self.ollama_host = os.getenv('OLLAMA_HOST', 'https://ollama.sps.ed.tum.de')
        self.ollama_available = False
        self.ollama_client = None
        self.openai_client = None
        self.openai_available = False
        self.actual_service = "Unknown"
        self._ollama_backoff_until = None
        
        self._init_openai()
        self._init_ollama()
        
        if not self.ollama_available and not self.openai_available:
            raise RuntimeError(f"No LLM service available for {role}. Set OPENAI_API_KEY or ensure Ollama is running.")
            
    def _init_openai(self):
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print(f"WARNING: OPENAI_API_KEY not set for {self.role} LLM")
            return

        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_key)
            self.openai_available = True
            if not self.ollama_available: # Default to OpenAI if Ollama not yet checked/available
                self.actual_service = "OpenAI API"
        except Exception as e:
            print(f"Failed to initialize OpenAI client for {self.role}: {e}")

    def _init_ollama(self):
        if self._check_ollama():
            self.ollama_available = True
            self.actual_service = "TUM Ollama (local)"
        
    def _check_ollama(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            if response.status_code == 200:
                self.ollama_client = ollama.Client(host=self.ollama_host)
                return True
        except Exception:
            pass
        return False
    
    async def reason(self, prompt: str, model_preference: str = "auto", show_thinking: bool = None) -> str:
        from datetime import datetime, timezone
        
        # Check backoff
        if self._ollama_backoff_until:
            if datetime.now(timezone.utc) < self._ollama_backoff_until:
                # In backoff, try OpenAI directly
                if self.openai_available:
                    self.actual_service = "OpenAI API"
                    return await self._call_openai(prompt)
                raise RuntimeError("LLM unavailable: Ollama in backoff and OpenAI unavailable.")
            else:
                self._ollama_backoff_until = None

        # Try Ollama first
        if self.ollama_available:
            try:
                self.actual_service = "TUM Ollama (local)"
                return await self._call_ollama(prompt, show_thinking)
            except Exception as e:
                from datetime import timedelta
                print(f"Ollama failed, backing off: {e}")
                self._ollama_backoff_until = datetime.now(timezone.utc) + timedelta(seconds=60)
                
                # Fallback to OpenAI
                if self.openai_available:
                    self.actual_service = "OpenAI API"
                    return await self._call_openai(prompt)
                raise RuntimeError(f"LLM unavailable: Ollama failed and OpenAI unavailable. Ollama error: {e}")
        
        # If Ollama not available, try OpenAI
        if self.openai_available:
            self.actual_service = "OpenAI API"
            return await self._call_openai(prompt)
        
        raise RuntimeError("LLM unavailable: No LLM models available.")

    def _get_config(self) -> Dict[str, Any]:
        return LLM_CONFIG.get(self.role, LLM_CONFIG["default"])

    async def _call_ollama(self, prompt: str, show_thinking: bool = None) -> str:
        config = self._get_config()
        model = config.get("ollama_model", "llama3.1:8b")
        
        # Determine thinking mode
        if show_thinking is None:
            show_thinking = (self.role == "reasoning")
            
        try:
            chat_params = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': config.get("system_prompt", "")},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False
            }
            
            # Add 'think' parameter for reasoning models if supported/requested
            if "deepseek-r1" in model or show_thinking:
                 # Note: The original code passed 'think' to chat, assuming the client supports it.
                 # We keep this behavior but guard it slightly or just pass it if the library allows **kwargs.
                 # The original code did: chat_params['think'] = show_thinking
                 # We will do the same.
                 chat_params['think'] = show_thinking
            
            response = self.ollama_client.chat(**chat_params)
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Ollama error: {e}")
    
    async def _call_openai(self, prompt: str) -> str:
        config = self._get_config()
        model = config.get("openai_model", "gpt-3.5-turbo")
        temperature = config.get("openai_temp", 0.5)
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": config.get("system_prompt", "")},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=temperature
        )
        return response.choices[0].message.content

    def get_current_status(self) -> str:
        return self.actual_service

