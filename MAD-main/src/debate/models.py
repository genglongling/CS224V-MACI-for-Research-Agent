"""
Model wrappers for different AI providers
"""
import os
from typing import Dict, Any, Optional
import json
import requests

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class BaseModel:
    """Base class for model wrappers"""
    
    def __init__(self, provider: str, model: str, temperature: float = 0.3, max_tokens: int = 1024):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, messages: list) -> Any:
        """Invoke the model with messages"""
        raise NotImplementedError

class OpenAIModel(BaseModel):
    """OpenAI model wrapper"""
    
    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__("openai", model, temperature, max_tokens)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
    
    def invoke(self, messages: list) -> Any:
        """Invoke OpenAI model"""
        try:
            # For some models, temperature is not supported, so we omit it
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens
                )
            except Exception as e:
                if "temperature" in str(e).lower():
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=self.max_tokens
                    )
                else:
                    raise
            return response.choices[0].message
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return type('MockResponse', (), {'content': '{"output": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "reason": {"A": "Error", "B": "Error", "C": "Error", "D": "Error"}}'})

class AnthropicModel(BaseModel):
    """Anthropic model wrapper"""
    
    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__("anthropic", model, temperature, max_tokens)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available. Install with: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def invoke(self, messages: list) -> Any:
        """Invoke Anthropic model"""
        try:
            # Convert messages to Anthropic format
            system_msg = None
            user_msg = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}]
            )
            return type('MockResponse', (), {'content': response.content[0].text})
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return type('MockResponse', (), {'content': '{"output": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "reason": {"A": "Error", "B": "Error", "C": "Error", "D": "Error"}}'})

class GoogleModel(BaseModel):
    """Google Gemini model wrapper"""
    
    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__("google", model, temperature, max_tokens)
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI library not available. Install with: pip install google-generativeai")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def invoke(self, messages: list) -> Any:
        """Invoke Google model"""
        try:
            # Convert messages to Google format
            system_msg = None
            user_msg = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
            
            if system_msg:
                prompt = f"{system_msg}\n\n{user_msg}"
            else:
                prompt = user_msg
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return type('MockResponse', (), {'content': response.text})
        except Exception as e:
            print(f"Google API error: {e}")
            return type('MockResponse', (), {'content': '{"output": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "reason": {"A": "Error", "B": "Error", "C": "Error", "D": "Error"}}'})

class LocalModel(BaseModel):
    """Local model wrapper using transformers"""
    
    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__("local", model, temperature, max_tokens)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch")
        
        # Load model and tokenizer
        print(f"Loading local model: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def invoke(self, messages: list) -> Any:
        """Invoke local model"""
        try:
            # Convert messages to prompt format
            system_msg = None
            user_msg = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
            
            # Format prompt based on model type
            if "Qwen" in str(self.model):
                # Qwen2.5 format
                if system_msg:
                    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            elif "Llama" in str(self.model):
                # Llama3.1 format
                if system_msg:
                    prompt = f"<|system|>\n{system_msg}<|end|>\n<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"
                else:
                    prompt = f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"
            else:
                # Default format - try to use chat template if available
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except:
                    # Fallback to simple format
                    if system_msg:
                        prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "
                    else:
                        prompt = f"User: {user_msg}\n\nAssistant: "
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            
            # Use simpler generation parameters to prevent hanging
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_tokens, 500),  # Use configured max_tokens but cap at 500 to prevent hanging
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clean up response and try to complete incomplete JSON
            response_text = response_text.strip()
            if not response_text:
                response_text = "I cannot provide a response at this time."
            
            # Try to complete incomplete JSON if it starts with { but doesn't end with }
            if response_text.startswith('{') and not response_text.endswith('}'):
                # Find the last complete key-value pair
                lines = response_text.split('\n')
                completed_lines = []
                for line in lines:
                    if line.strip() and ':' in line:
                        # Check if this line looks like a complete key-value pair
                        if line.strip().endswith(',') or line.strip().endswith('"'):
                            completed_lines.append(line)
                        elif line.strip().endswith('"') and not line.strip().endswith('",'):
                            # Incomplete string, skip this line and stop
                            break
                        else:
                            completed_lines.append(line)
                    else:
                        completed_lines.append(line)
                
                # Reconstruct and try to close the JSON
                response_text = '\n'.join(completed_lines)
                if response_text.endswith(','):
                    response_text = response_text[:-1]  # Remove trailing comma
                if not response_text.endswith('}'):
                    response_text += '}'
            
            return type('MockResponse', (), {'content': response_text})
        except Exception as e:
            print(f"Local model error: {e}")
            return type('MockResponse', (), {'content': '{"output": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "reason": {"A": "Error", "B": "Error", "C": "Error", "D": "Error"}}'})

class LitGPTModel(BaseModel):
    """Model wrapper for litgpt API endpoints"""
    
    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 1024):
        super().__init__("litgpt", model, temperature, max_tokens)
        # Extract port from model string (e.g., "8000" -> "http://localhost:8000")
        if model.isdigit():
            self.base_url = f"http://localhost:{model}"
        else:
            self.base_url = model.rstrip('/')
        self.session = requests.Session()
    
    def invoke(self, messages):
        """Send request to litgpt API endpoint"""
        try:
            # Convert messages to litgpt format
            if len(messages) == 2 and messages[0]["role"] == "system":
                # System + user message
                prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            else:
                # Just user message or other format
                prompt = messages[-1]["content"]
            
            # Use the correct litgpt API structure: /predict endpoint
            url = f"{self.base_url}/predict"
            data = {
                "prompt": prompt,
                "max_new_tokens": max(self.max_tokens, 16384),  # Increased to 16384 to match server limit
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": 50
            }
            
            print(f"   🔧 Debug: self.temperature = {self.temperature}, self.max_tokens = {self.max_tokens}")
            
            print(f"   Sending request to: {url}")
            print(f"   Prompt: {prompt[:100]}...")
            print(f"   Request data: {data}")
            
            # Send the POST request
            response = self.session.post(url, json=data, timeout=300)  # Increased timeout for longer responses
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success! Response received")
                print(f"   Raw response: {result}")
                print(f"   Raw response type: {type(result)}")
                print(f"   Raw response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                # Extract the generated text from response
                generated_text = result.get("output", "")
                if not generated_text:
                    print(f"   ⚠️ No 'output' field in response, using full response")
                    generated_text = str(result)
                
                print(f"   Extracted text: {generated_text[:200]}...")
                print(f"   Full extracted text length: {len(generated_text)}")
                print(f"   Last 100 chars: {generated_text[-100:] if len(generated_text) > 100 else generated_text}")
                
                # Try to clean up incomplete JSON responses
                if generated_text.startswith('```json') and not generated_text.endswith('```'):
                    print(f"   ⚠️ Incomplete JSON detected, attempting to complete...")
                    # Remove markdown formatting
                    generated_text = generated_text.replace('```json', '').replace('```', '').strip()
                
                # Try to complete incomplete JSON (with or without markdown)
                if generated_text.startswith('{') and not generated_text.endswith('}'):
                    print(f"   ⚠️ Incomplete JSON detected, attempting to complete...")
                    
                    # First, try to fix common JSON formatting issues
                    generated_text = generated_text.replace('":0.', '": "0.').replace('":0,', '": "0,').replace('":1,', '": "1,')
                    generated_text = generated_text.replace('":0}', '": "0"}').replace('":1}', '": "1"}')
                    
                    # Count brackets to understand the structure
                    open_braces = generated_text.count('{')
                    close_braces = generated_text.count('}')
                    open_brackets = generated_text.count('[')
                    close_brackets = generated_text.count(']')
                    
                    print(f"   📊 Bracket analysis: {{: {open_braces}, }}: {close_braces}, [: {open_brackets}, ]: {close_brackets}")
                    
                    # Simple approach: just close the JSON properly
                    if open_braces > close_braces:
                        # Add missing closing braces
                        missing_braces = open_braces - close_braces
                        generated_text += '}' * missing_braces
                        print(f"   🔧 Added {missing_braces} closing braces")
                    
                    if open_brackets > close_brackets:
                        # Add missing closing brackets
                        missing_brackets = open_brackets - close_brackets
                        generated_text += ']' * missing_brackets
                        print(f"   🔧 Added {missing_brackets} closing brackets")
                    
                    print(f"   🔧 Completed JSON: {generated_text[:200]}...")
                    print(f"   🔧 Final bracket count: {{: {generated_text.count('{')}, }}: {generated_text.count('}')}")
                
                return type('MockResponse', (), {'content': generated_text})
            else:
                print(f"   ❌ Error: {response.status_code}, {response.text}")
                return type('MockResponse', (), {'content': '{"output": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "reason": {"A": "Error", "B": "Error", "C": "Error", "D": "Error"}}'})
                
        except Exception as e:
            print(f"LitGPT API call failed: {e}")
            return type('MockResponse', (), {'content': '{"output": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "reason": {"A": "Error", "B": "Error", "C": "Error", "D": "Error"}}'})

class LLMFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def make(provider: str, model: str, temperature: float = 0.3, max_tokens: int = 1024) -> BaseModel:
        """Create a model instance based on provider"""
        if provider == "openai":
            return OpenAIModel(model, temperature, max_tokens)
        elif provider == "anthropic":
            return AnthropicModel(model, temperature, max_tokens)
        elif provider == "google":
            return GoogleModel(model, temperature, max_tokens)
        elif provider == "local":
            return LocalModel(model, temperature, max_tokens)
        elif provider == "litgpt":
            return LitGPTModel(model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
