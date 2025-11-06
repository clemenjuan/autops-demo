import os
import asyncio
import json
import ollama
import requests

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
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print(f"WARNING: OPENAI_API_KEY not set for {role} LLM")
        else:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
                self.openai_available = True
            except Exception as e:
                print(f"Failed to initialize OpenAI client for {role}: {e}")
        
        self.ollama_available = self._check_ollama()
        if self.ollama_available:
            self.actual_service = "TUM Ollama (local)"
        elif self.openai_available:
            self.actual_service = "OpenAI API"
        
        if not self.ollama_available and not self.openai_available:
            raise RuntimeError(f"No LLM service available for {role}. Set OPENAI_API_KEY or ensure Ollama is running.")
        
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
        from datetime import datetime, timedelta, timezone
        
        in_backoff = False
        if self._ollama_backoff_until:
            if datetime.now(timezone.utc) < self._ollama_backoff_until:
                in_backoff = True
            else:
                self._ollama_backoff_until = None

        if self.ollama_available and not in_backoff:
            try:
                self.actual_service = "TUM Ollama (local)"
                return await self._call_ollama(prompt, show_thinking)
            except Exception as e:
                self._ollama_backoff_until = datetime.now(timezone.utc) + timedelta(seconds=60)
                
                if self.openai_available:
                    self.actual_service = "OpenAI API"
                    try:
                        return await self._call_openai(prompt)
                    except Exception as e2:
                        raise RuntimeError(f"LLM unavailable: Both Ollama and OpenAI failed. Ollama: {e}, OpenAI: {e2}")
                raise RuntimeError(f"LLM unavailable: Ollama failed and OpenAI unavailable. Ollama error: {e}")
        
        if self.openai_available:
            try:
                self.actual_service = "OpenAI API"
                return await self._call_openai(prompt)
            except Exception as e:
                raise RuntimeError(f"LLM unavailable: OpenAI failed. Error: {e}")
        
        raise RuntimeError("LLM unavailable: No LLM models available.")

    async def _call_ollama(self, prompt: str, show_thinking: bool = None) -> str:
        """Call Ollama model using the TUM server"""
        # Available models on TUM server (from curl output):
        # - deepseek-r1:70b (70B) - can show/hide reasoning
        # - llama3.3:latest (70B)
        # - gemma3:27b (27B)
        # - qwen3:32b (32B)
        # - phi4:14b (14B)
        # - llama3.1:8b (8B)
        # - mistral:7b (7B)
        # - deepseek-r1:latest (7B)
        # - qwen2.5:latest (7B)
        # - gemma2:latest (9B)
        # - mistral:latest (7B)
        # - phi4:latest (14B)
        
        # Choose model and thinking mode based on role
        if self.role == "reasoning":
            model = "deepseek-r1:70b"
            if show_thinking is None:
                show_thinking = True
        else:
            model = "llama3.1:8b"
            show_thinking = False
        
        try:
            chat_params = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': self._get_system_prompt()},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False
            }
            
            if "deepseek-r1" in model:
                chat_params['think'] = show_thinking
            
            response = self.ollama_client.chat(**chat_params)
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Ollama error: {e}")
    
    def get_current_status(self) -> str:
        return self.actual_service
    
    def _get_system_prompt(self) -> str:
        if self.role == "reasoning":
            return "You are an advanced satellite operations reasoning agent. You excel at complex reasoning, multi-step analysis, and decision-making for satellite operations. Provide clear, step-by-step reasoning and justify your recommendations."
        if self.role == "general":
            return "You are a professional satellite operations assistant. You help with quick task understanding, categorization, and clear communication. Always be direct, concise, and professional."
        return "You are a professional satellite operations assistant. You are helpful, professional, and provide clear, concise responses."

    async def _call_openai(self, prompt: str) -> str:
        if self.role == "reasoning":
            model, temperature = "gpt-4", 0.3
        else:
            model, temperature = "gpt-3.5-turbo", 0.5
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=temperature
        )
        return response.choices[0].message.content
