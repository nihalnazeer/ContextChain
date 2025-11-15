# contextchain/src/llm.py
"""
Thin LLM client abstraction for ContextChain v2.1
Supports: Ollama, HF, OpenAI, Anthropic, Grok, OpenRouter
Features: streaming, caching, retries, metrics, budget-aware
"""

import asyncio
import logging
import time
import json
from typing import Dict, Optional, Any, List, AsyncGenerator, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# Optional imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None
    HF_AVAILABLE = False

from .acba import BudgetAllocation

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class LLMConfig:
    model_name: str = "tinyllama"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30
    enable_streaming: bool = False
    enable_caching: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    device: str = "cuda" if HF_AVAILABLE and torch and torch.cuda.is_available() else "cpu"

    def dict(self, exclude=None):
        d = asdict(self)
        if exclude:
            for key in exclude:
                d.pop(key, None)
        return d


@dataclass
class GenerationResult:
    content: str
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time: float = 0.0
    finish_reason: Optional[str] = None
    model_used: str = ""
    cached: bool = False
    retry_count: int = 0
    stream_chunks: Optional[List[str]] = None

@dataclass
class PerformanceMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_generation_time: float = 0.0
    total_tokens: int = 0
    cache_hit_rate: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)

# --------------------------------------------------------------------------- #
# Base LLM Client
# --------------------------------------------------------------------------- #
class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(5)
        self.cache: Optional[Dict[str, GenerationResult]] = {} if config.enable_caching else None
        self.metrics = PerformanceMetrics()
        logger.info(f"LLM client initialized: {self.__class__.__name__} - {config.model_name}")

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        pass

    async def generate_optimized(
        self,
        prompt: str,
        budget: BudgetAllocation,
        stream: bool = False,
        **kwargs
    ) -> Union[GenerationResult, AsyncGenerator[str, None]]:
        max_tokens = min(budget.generation_tokens, self.config.max_tokens)

        if self.cache:
            cache_key = self._get_cache_key(prompt, max_tokens, self.config.temperature)
            if cache_key in self.cache:
                result = self.cache[cache_key]
                result.cached = True
                self._update_metrics(result, success=True)
                return result

        if stream and self.config.enable_streaming:
            return self._stream_with_budget(prompt, max_tokens, budget, **kwargs)

        async with self.semaphore:
            start_time = time.time()
            for attempt in range(self.config.retry_attempts):
                try:
                    result = await self.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        **kwargs
                    )
                    result.generation_time = time.time() - start_time
                    result.model_used = self.config.model_name
                    result.retry_count = attempt

                    if self.cache and result.content:
                        cache_key = self._get_cache_key(prompt, max_tokens, self.config.temperature)
                        self.cache[cache_key] = result
                        if len(self.cache) > 1000:
                            old_keys = list(self.cache.keys())[:500]
                            for k in old_keys:
                                del self.cache[k]

                    self._update_metrics(result, success=True)
                    return result

                except Exception as e:
                    logger.warning(f"LLM generation failed (attempt {attempt + 1}): {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        error_result = GenerationResult(
                            content=f"[Generation failed: {str(e)}]",
                            generation_time=time.time() - start_time,
                            finish_reason="error",
                            retry_count=attempt + 1
                        )
                        self._update_metrics(error_result, success=False)
                        return error_result

    async def _stream_with_budget(self, prompt: str, max_tokens: int, budget: BudgetAllocation, **kwargs):
        chunks = []
        start_time = time.time()
        try:
            async for chunk in self.generate_stream(prompt, max_tokens, self.config.temperature, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"[Stream error: {e}]"

        full_content = "".join(chunks)
        result = GenerationResult(
            content=full_content,
            tokens_used=len(full_content.split()) * 1.3,
            generation_time=time.time() - start_time,
            stream_chunks=chunks,
            cached=False
        )
        if self.cache:
            cache_key = self._get_cache_key(prompt, max_tokens, self.config.temperature)
            self.cache[cache_key] = result
        self._update_metrics(result, success=True)

    def _get_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        key_data = {
            "prompt_hash": hashlib.md5(prompt.encode("utf-8")).hexdigest(),
            "model": self.config.model_name,
            "max_tokens": max_tokens,
            "temperature": round(temperature, 2)
        }
        return json.dumps(key_data, sort_keys=True)

    def _update_metrics(self, result: GenerationResult, success: bool):
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
            self.metrics.total_tokens += result.tokens_used
            n = self.metrics.successful_requests
            self.metrics.avg_generation_time = (
                (self.metrics.avg_generation_time * (n - 1) + result.generation_time) / n
            )
        else:
            self.metrics.failed_requests += 1
        if self.cache:
            self.metrics.cache_hit_rate = sum(1 for r in self.cache.values() if r.cached) / max(self.metrics.total_requests, 1)

    async def get_performance_stats(self) -> Dict[str, Any]:
        uptime = (datetime.utcnow() - self.metrics.start_time).total_seconds()
        return {
            "status": "active" if self.metrics.total_requests > 0 else "idle",
            "uptime_seconds": uptime,
            "total_requests": self.metrics.total_requests,
            "success_rate": self.metrics.successful_requests / max(self.metrics.total_requests, 1),
            "avg_generation_time": round(self.metrics.avg_generation_time, 3),
            "total_tokens": self.metrics.total_tokens,
            "cache_hit_rate": round(self.metrics.cache_hit_rate, 3),
            "model": self.config.model_name
        }

    async def close(self):
        pass

# --------------------------------------------------------------------------- #
# Provider Implementations
# --------------------------------------------------------------------------- #

class OllamaLLM(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        self.client = httpx.AsyncClient(
            base_url=config.api_base or "http://localhost:11434",
            timeout=120
        )

    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": False
        }
        response = await self.client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")
        return GenerationResult(
            content=content,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            tokens_used=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            finish_reason=data.get("done_reason")
        )

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs):
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": True
        }
        async with self.client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if data.get("done"): break
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    continue

    async def health_check(self) -> Dict[str, Any]:
        try:
            start = time.time()
            resp = await self.client.get("/api/tags")
            return {
                "status": "healthy" if resp.status_code == 200 else "degraded",
                "latency": time.time() - start,
                "model": self.config.model_name
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self):
        await self.client.aclose()


class HuggingFaceLLM(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch not installed. Run: pip install transformers torch")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )

    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        loop = asyncio.get_event_loop()
        def generate_sync():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text.replace(prompt, "").strip()
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            return response_text, input_tokens, output_tokens
        
        content, input_tokens, output_tokens = await loop.run_in_executor(None, generate_sync)
        return GenerationResult(
            content=content,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncGenerator[str, None]:
        result = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield result.content

    async def health_check(self) -> Dict[str, Any]:
        try:
            start_time = time.time()
            status = "healthy" if self.tokenizer and self.model else "unhealthy"
            latency = time.time() - start_time
            return {
                'status': status,
                'model': self.config.model_name,
                'device': self.config.device,
                'latency_seconds': latency,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.config.model_name,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }


class OpenAILLM(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library not installed. Run: pip install openai")
        if not self.config.api_key:
            raise ValueError("API key required for OpenAI")
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout
        )

    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        messages = kwargs.get('messages', [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ])
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        return GenerationResult(
            content=response.choices[0].message.content.strip(),
            tokens_used=response.usage.total_tokens,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            finish_reason=response.choices[0].finish_reason
        )

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            status = "healthy" if response.choices[0].message.content else "unhealthy"
            latency = time.time() - start_time
            return {
                'status': status,
                'model': self.config.model_name,
                'latency_seconds': latency,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.config.model_name,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }

    async def close(self):
        await self.client.close()


class AnthropicLLM(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library not installed. Run: pip install anthropic")
        if not self.config.api_key:
            raise ValueError("API key required for Anthropic")
        self.client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout
        )

    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        system_message = kwargs.get('system', "You are a helpful AI assistant.")
        response = await self.client.messages.create(
            model=self.config.model_name,
            system=system_message,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return GenerationResult(
            content=response.content[0].text.strip(),
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason
        )

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncGenerator[str, None]:
        result = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield result.content

    async def health_check(self) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = await self.client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            status = "healthy" if response.content[0].text else "unhealthy"
            latency = time.time() - start_time
            return {
                'status': status,
                'model': self.config.model_name,
                'latency_seconds': latency,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.config.model_name,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }

    async def close(self):
        await self.client.close()


class GrokLLM(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx library not installed. Run: pip install httpx")
        if not self.config.api_key:
            raise ValueError("API key required for Grok")
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base or "https://api.x.ai/v1",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout
        )

    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        result_data = response.json()
        choice = result_data['choices'][0]
        usage = result_data.get('usage', {})
        return GenerationResult(
            content=choice['message']['content'].strip(),
            tokens_used=usage.get('total_tokens', 0),
            input_tokens=usage.get('prompt_tokens', 0),
            output_tokens=usage.get('completion_tokens', 0),
            finish_reason=choice.get('finish_reason')
        )

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncGenerator[str, None]:
        result = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield result.content

    async def health_check(self) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = await self.client.get("/models")
            status = "healthy" if response.status_code == 200 else "unhealthy"
            latency = time.time() - start_time
            return {
                'status': status,
                'model': self.config.model_name,
                'latency_seconds': latency,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.config.model_name,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }

    async def close(self):
        await self.client.aclose()


class OpenRouterLLM(BaseLLMClient):
    """
    OpenRouter LLM client implementation.
    Supports routing to various models via OpenRouter API.
    """
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx library not installed. Run: pip install httpx")
        if not self.config.api_key:
            raise ValueError("API key required for OpenRouter")
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "HTTP-Referer": self.config.api_base or "",  # Optional: Your app's URL
                "X-Title": "ContextChain App",  # Optional: Your app's name
            },
            timeout=self.config.timeout
        )

    async def generate(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> GenerationResult:
        messages = kwargs.get('messages', [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ])
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        result_data = response.json()
        choice = result_data['choices'][0]
        usage = result_data.get('usage', {})
        return GenerationResult(
            content=choice['message']['content'].strip(),
            tokens_used=usage.get('total_tokens', 0),
            input_tokens=usage.get('prompt_tokens', 0),
            output_tokens=usage.get('completion_tokens', 0),
            finish_reason=choice.get('finish_reason')
        )

    async def generate_stream(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> AsyncGenerator[str, None]:
        messages = kwargs.get('messages', [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ])
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        async with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            full_content = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                chunk = delta["content"]
                                full_content += chunk
                                yield chunk
                    except json.JSONDecodeError:
                        continue
            # For compatibility, yield the full content if no chunks
            if not full_content:
                yield full_content

    async def health_check(self) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = await self.client.get("/models")
            status = "healthy" if response.status_code == 200 else "unhealthy"
            latency = time.time() - start_time
            return {
                'status': status,
                'model': self.config.model_name,
                'latency_seconds': latency,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model': self.config.model_name,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }

    async def close(self):
        await self.client.aclose()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_llm_client(provider: str, model: str, api_key: str = None, **kwargs) -> BaseLLMClient:
    config = LLMConfig(model_name=model, api_key=api_key, **kwargs)
    providers = {
        "ollama": OllamaLLM,
        "huggingface": HuggingFaceLLM,
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "grok": GrokLLM,
        "openrouter": OpenRouterLLM 
    }
    cls = providers.get(provider.lower())
    if not cls:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return cls(config)