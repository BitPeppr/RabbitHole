"""Multi-provider LLM support with automatic load balancing and failover.

Supports multiple API providers with:
- Multiple API keys per provider
- Per-provider task assignments (e.g., OLLAMA_TASKS=report,research)
- Automatic rerouting on rate limits
- Circuit breaker pattern for failed providers
"""

import os
import time
import json
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

from .logger import log


class ProviderStatus(Enum):
    """Health status of a provider instance."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some failures but still usable
    RATE_LIMITED = "rate_limited"  # Hit rate limit, retry after cooldown
    FAILED = "failed"  # Too many failures, circuit breaker open


class TaskType(Enum):
    """LLM task types for routing."""
    SUMMARIZATION = "summarization"
    SUBTOPIC = "subtopic"
    VALIDATION = "validation"
    REPORT = "report"
    RECOMMENDATIONS = "recommendations"
    RESEARCH = "research"
    
    @classmethod
    def all_types(cls) -> List[str]:
        return [t.value for t in cls]


class RateLimitError(Exception):
    """Raised when a provider hits rate limit."""
    def __init__(self, message: str, retry_after: int = 60, provider_id: str = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.provider_id = provider_id


@dataclass
class ProviderHealth:
    """Tracks health metrics for a provider instance."""
    status: ProviderStatus = ProviderStatus.HEALTHY
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    rate_limit_until: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    
    # Circuit breaker thresholds
    DEGRADED_THRESHOLD: int = 3
    FAILED_THRESHOLD: int = 5
    FAILED_COOLDOWN_SEC: int = 600  # 10 minutes


@dataclass
class ProviderUsage:
    """Tracks usage metrics for a provider instance."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ProviderBase(ABC):
    """Abstract base class for LLM providers.
    
    Each provider implementation must:
    - Handle API authentication
    - Implement call() for synchronous requests
    - Detect and raise RateLimitError on rate limits
    - Track usage metrics
    """
    
    def __init__(
        self,
        provider_id: str,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        backup_model: str = None,
        enabled_tasks: List[str] = None,
        log_requests: bool = False,
    ):
        self.provider_id = provider_id
        self.provider_type = self.__class__.__name__.replace("Provider", "").lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.backup_model = backup_model
        self.enabled_tasks = enabled_tasks or TaskType.all_types()
        self.log_requests = log_requests
        
        self.health = ProviderHealth()
        self.usage = ProviderUsage()
        self._lock = threading.Lock()
    
    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make a synchronous LLM call.
        
        Args:
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            model: Override default model
            
        Returns:
            Tuple of (response_content, usage_dict)
            
        Raises:
            RateLimitError: If rate limit hit (will trigger rerouting)
            Exception: For other errors
        """
        pass
    
    def is_enabled_for_task(self, task_type: str) -> bool:
        """Check if this provider is enabled for a task type."""
        return task_type in self.enabled_tasks
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy and can accept requests."""
        with self._lock:
            # Check if rate limit has expired
            if self.health.status == ProviderStatus.RATE_LIMITED:
                if time.time() >= (self.health.rate_limit_until or 0):
                    self.health.status = ProviderStatus.HEALTHY
                    self.health.rate_limit_until = None
                    log.config(f"Provider {self.provider_id} rate limit expired, marking healthy")
                else:
                    return False
            
            # Check if failed cooldown has expired
            if self.health.status == ProviderStatus.FAILED:
                if self.health.last_failure_time and \
                   time.time() - self.health.last_failure_time > ProviderHealth.FAILED_COOLDOWN_SEC:
                    self.health.status = ProviderStatus.HEALTHY
                    self.health.consecutive_failures = 0
                    log.config(f"Provider {self.provider_id} cooldown expired, marking healthy")
                else:
                    return False
            
            return self.health.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED)
    
    def mark_success(self):
        """Mark a successful call, reset failure counters."""
        with self._lock:
            self.health.consecutive_failures = 0
            self.health.status = ProviderStatus.HEALTHY
    
    def mark_failure(self):
        """Mark a failed call, update circuit breaker state."""
        with self._lock:
            self.health.consecutive_failures += 1
            self.health.total_failures += 1
            self.health.last_failure_time = time.time()
            
            if self.health.consecutive_failures >= ProviderHealth.FAILED_THRESHOLD:
                self.health.status = ProviderStatus.FAILED
                log.warning(f"Provider {self.provider_id} marked FAILED after {self.health.consecutive_failures} failures")
            elif self.health.consecutive_failures >= ProviderHealth.DEGRADED_THRESHOLD:
                self.health.status = ProviderStatus.DEGRADED
                log.warning(f"Provider {self.provider_id} marked DEGRADED")
    
    def mark_rate_limited(self, retry_after: int = 60):
        """Mark provider as rate limited."""
        with self._lock:
            self.health.status = ProviderStatus.RATE_LIMITED
            self.health.rate_limit_until = time.time() + retry_after
            log.warning(f"Provider {self.provider_id} rate limited for {retry_after}s")
    
    def record_usage(self, usage: Dict):
        """Record token usage from API response."""
        if not usage:
            return
        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.usage.completion_tokens += int(usage.get("completion_tokens", 0))
            self.usage.total_tokens += int(usage.get("total_tokens", 0))
            self.health.total_calls += 1
    
    def get_usage(self) -> Dict:
        """Get usage statistics."""
        with self._lock:
            return {
                "provider_id": self.provider_id,
                "provider_type": self.provider_type,
                "calls": self.usage.calls,
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
    
    def _detect_rate_limit(self, response=None, error=None) -> Optional[int]:
        """Detect rate limit from response or error. Returns retry_after seconds or None."""
        # Check HTTP status
        if response is not None:
            status_code = getattr(response, 'status_code', None)
            if status_code == 429:
                # Try to extract Retry-After header
                headers = getattr(response, 'headers', {})
                retry_after = headers.get('Retry-After') or headers.get('retry-after')
                if retry_after:
                    try:
                        return int(retry_after)
                    except ValueError:
                        pass
                return 60  # Default
        
        # Check error message
        if error:
            error_str = str(error).lower()
            rate_limit_indicators = [
                "rate limit", "rate_limit", "ratelimit",
                "too many requests", "quota exceeded",
                "resource_exhausted", "throttl"
            ]
            if any(indicator in error_str for indicator in rate_limit_indicators):
                return 60
        
        return None


class OpenRouterProvider(ProviderBase):
    """OpenRouter API provider."""
    
    DEFAULT_MODEL = "arcee-ai/trinity-large-preview:free"
    DEFAULT_BASE_URL = "https://openrouter.ai/api"
    ALTERNATE_BASE_URLS = ["https://api.openrouter.ai"]
    
    def __init__(
        self,
        provider_id: str,
        api_key: str,
        model: str = None,
        backup_model: str = None,
        base_url: str = None,
        enabled_tasks: List[str] = None,
        log_requests: bool = False,
    ):
        super().__init__(
            provider_id=provider_id,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
            backup_model=backup_model,
            enabled_tasks=enabled_tasks,
            log_requests=log_requests,
        )
        
        # Build list of base URLs to try
        self.base_urls = [self.base_url.rstrip("/")]
        for alt in self.ALTERNATE_BASE_URLS:
            if alt.rstrip("/") not in self.base_urls:
                self.base_urls.append(alt.rstrip("/"))
    
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make OpenRouter API call."""
        import requests
        from requests.exceptions import RequestException
        
        model = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        last_error = None
        for base_url in self.base_urls:
            url = f"{base_url}/v1/chat/completions"
            try:
                if self.log_requests:
                    log.progress(f"[{self.provider_id}] request model={model} base={base_url}")
                
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                
                # Check for rate limit
                retry_after = self._detect_rate_limit(response=resp)
                if retry_after:
                    raise RateLimitError(
                        f"Rate limit hit on {self.provider_id}",
                        retry_after=retry_after,
                        provider_id=self.provider_id
                    )
                
                resp.raise_for_status()
                
                data = resp.json()
                content = self._extract_content(data)
                usage = data.get("usage")
                
                self.record_usage(usage)
                
                if self.log_requests:
                    tokens = usage.get("total_tokens", 0) if usage else 0
                    log.progress(f"[{self.provider_id}] response ok tokens={tokens}")
                
                return content, usage
                
            except RateLimitError:
                raise
            except RequestException as e:
                retry_after = self._detect_rate_limit(error=e)
                if retry_after:
                    raise RateLimitError(
                        f"Rate limit hit on {self.provider_id}: {e}",
                        retry_after=retry_after,
                        provider_id=self.provider_id
                    )
                last_error = e
                log.error(f"[{self.provider_id}] request failed for {base_url}: {e}")
                continue
        
        raise last_error or Exception(f"OpenRouter request failed on {self.provider_id}")
    
    def _extract_content(self, data: Dict) -> Optional[str]:
        """Extract content from OpenRouter response."""
        if not isinstance(data, dict):
            return None
        
        choices = data.get("choices", [])
        if choices and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                if "message" in first and isinstance(first["message"], dict):
                    return first["message"].get("content")
                elif "text" in first:
                    return first.get("text")
        
        return data.get("output") or data.get("result") or data.get("generated_text")


class GroqProvider(ProviderBase):
    """Groq API provider for fast inference."""
    
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_BASE_URL = "https://api.groq.com/openai"
    
    def __init__(
        self,
        provider_id: str,
        api_key: str,
        model: str = None,
        backup_model: str = None,
        base_url: str = None,
        enabled_tasks: List[str] = None,
        log_requests: bool = False,
    ):
        super().__init__(
            provider_id=provider_id,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
            backup_model=backup_model,
            enabled_tasks=enabled_tasks,
            log_requests=log_requests,
        )
    
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make Groq API call (OpenAI-compatible)."""
        import requests
        from requests.exceptions import RequestException
        
        model = model or self.model
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        try:
            if self.log_requests:
                log.progress(f"[{self.provider_id}] request model={model}")
            
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            
            # Check for rate limit
            retry_after = self._detect_rate_limit(response=resp)
            if retry_after:
                raise RateLimitError(
                    f"Rate limit hit on {self.provider_id}",
                    retry_after=retry_after,
                    provider_id=self.provider_id
                )
            
            resp.raise_for_status()
            
            data = resp.json()
            content = None
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
            
            usage = data.get("usage")
            self.record_usage(usage)
            
            if self.log_requests:
                tokens = usage.get("total_tokens", 0) if usage else 0
                log.progress(f"[{self.provider_id}] response ok tokens={tokens}")
            
            return content, usage
            
        except RateLimitError:
            log.warning(f"[{self.provider_id}] rate limit hit")
            raise
        except RequestException as e:
            log.error(f"[{self.provider_id}] request failed: {e}")
            retry_after = self._detect_rate_limit(error=e)
            if retry_after:
                raise RateLimitError(
                    f"Rate limit hit on {self.provider_id}: {e}",
                    retry_after=retry_after,
                    provider_id=self.provider_id
                )
            raise
        except Exception as e:
            log.error(f"[{self.provider_id}] error: {e}")
            raise


class GoogleAIProvider(ProviderBase):
    """Google AI Studio (Gemini) provider."""
    
    DEFAULT_MODEL = "gemini-2.0-flash-exp"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
    
    def __init__(
        self,
        provider_id: str,
        api_key: str,
        model: str = None,
        backup_model: str = None,
        base_url: str = None,
        enabled_tasks: List[str] = None,
        log_requests: bool = False,
    ):
        super().__init__(
            provider_id=provider_id,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
            backup_model=backup_model,
            enabled_tasks=enabled_tasks,
            log_requests=log_requests,
        )
    
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make Google AI API call."""
        import requests
        from requests.exceptions import RequestException
        
        model = model or self.model
        url = f"{self.base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
        
        # Convert OpenAI message format to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg.get("role") in ("user", "system") else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        
        try:
            if self.log_requests:
                log.progress(f"[{self.provider_id}] request model={model}")
            
            resp = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                params={"key": self.api_key},
                timeout=60
            )
            
            # Check for rate limit
            retry_after = self._detect_rate_limit(response=resp)
            if retry_after:
                raise RateLimitError(
                    f"Rate limit hit on {self.provider_id}",
                    retry_after=retry_after,
                    provider_id=self.provider_id
                )
            
            resp.raise_for_status()
            
            data = resp.json()
            content = None
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    content = parts[0].get("text")
            
            # Google AI doesn't provide standard usage, estimate it
            usage = data.get("usageMetadata", {})
            if usage:
                usage = {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0),
                }
            self.record_usage(usage)
            
            if self.log_requests:
                tokens = usage.get("total_tokens", 0) if usage else 0
                log.progress(f"[{self.provider_id}] response ok tokens={tokens}")
            
            return content, usage
            
        except RateLimitError:
            log.warning(f"[{self.provider_id}] rate limit hit")
            raise
        except RequestException as e:
            log.error(f"[{self.provider_id}] request failed: {e}")
            retry_after = self._detect_rate_limit(error=e)
            if retry_after:
                raise RateLimitError(
                    f"Rate limit hit on {self.provider_id}: {e}",
                    retry_after=retry_after,
                    provider_id=self.provider_id
                )
            raise
        except Exception as e:
            log.error(f"[{self.provider_id}] error: {e}")
            raise


class OllamaProvider(ProviderBase):
    """Ollama local LLM provider."""
    
    DEFAULT_MODEL = "llama3.1"
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(
        self,
        provider_id: str,
        base_url: str = None,
        model: str = None,
        backup_model: str = None,
        enabled_tasks: List[str] = None,
        log_requests: bool = False,
    ):
        super().__init__(
            provider_id=provider_id,
            api_key=None,  # Ollama doesn't use API keys
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
            backup_model=backup_model,
            enabled_tasks=enabled_tasks,
            log_requests=log_requests,
        )
    
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make Ollama API call (OpenAI-compatible endpoint)."""
        import requests
        from requests.exceptions import RequestException
        
        model = model or self.model
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        
        try:
            if self.log_requests:
                log.progress(f"[{self.provider_id}] request model={model}")
            
            resp = requests.post(url, json=payload, headers=headers, timeout=300)  # Longer timeout for local
            
            # Check for rate limit (Ollama can return 503 if busy)
            if resp.status_code == 503:
                raise RateLimitError(
                    f"Ollama busy on {self.provider_id}",
                    retry_after=10,  # Short retry for local
                    provider_id=self.provider_id
                )
            
            resp.raise_for_status()
            
            data = resp.json()
            content = None
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
            
            usage = data.get("usage")
            self.record_usage(usage)
            
            if self.log_requests:
                tokens = usage.get("total_tokens", 0) if usage else 0
                log.progress(f"[{self.provider_id}] response ok tokens={tokens}")
            
            return content, usage
            
        except RateLimitError:
            log.warning(f"[{self.provider_id}] busy/rate limited")
            raise
        except RequestException as e:
            log.error(f"[{self.provider_id}] request failed: {e}")
            if "503" in str(e) or "busy" in str(e).lower():
                raise RateLimitError(
                    f"Ollama busy on {self.provider_id}: {e}",
                    retry_after=10,
                    provider_id=self.provider_id
                )
            raise
        except Exception as e:
            log.error(f"[{self.provider_id}] error: {e}")
            raise


class OpenAIProvider(ProviderBase):
    """OpenAI API provider."""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_BASE_URL = "https://api.openai.com"
    
    def __init__(
        self,
        provider_id: str,
        api_key: str,
        model: str = None,
        backup_model: str = None,
        base_url: str = None,
        enabled_tasks: List[str] = None,
        log_requests: bool = False,
    ):
        super().__init__(
            provider_id=provider_id,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            model=model or self.DEFAULT_MODEL,
            backup_model=backup_model,
            enabled_tasks=enabled_tasks,
            log_requests=log_requests,
        )
    
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make OpenAI API call."""
        import requests
        from requests.exceptions import RequestException
        
        model = model or self.model
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        try:
            if self.log_requests:
                log.progress(f"[{self.provider_id}] request model={model}")
            
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            
            # Check for rate limit
            retry_after = self._detect_rate_limit(response=resp)
            if retry_after:
                raise RateLimitError(
                    f"Rate limit hit on {self.provider_id}",
                    retry_after=retry_after,
                    provider_id=self.provider_id
                )
            
            resp.raise_for_status()
            
            data = resp.json()
            content = None
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
            
            usage = data.get("usage")
            self.record_usage(usage)
            
            if self.log_requests:
                tokens = usage.get("total_tokens", 0) if usage else 0
                log.progress(f"[{self.provider_id}] response ok tokens={tokens}")
            
            return content, usage
            
        except RateLimitError:
            log.warning(f"[{self.provider_id}] rate limit hit")
            raise
        except RequestException as e:
            log.error(f"[{self.provider_id}] request failed: {e}")
            retry_after = self._detect_rate_limit(error=e)
            if retry_after:
                raise RateLimitError(
                    f"Rate limit hit on {self.provider_id}: {e}",
                    retry_after=retry_after,
                    provider_id=self.provider_id
                )
            raise
        except Exception as e:
            log.error(f"[{self.provider_id}] error: {e}")
            raise


def parse_provider_tasks(provider_name: str) -> List[str]:
    """Parse PROVIDER_TASKS env var to get enabled task types.
    
    Examples:
        OPENROUTER_TASKS=all → all task types
        GROQ_TASKS=none → []
        OLLAMA_TASKS=report,research → ["report", "research"]
    """
    env_var = f"{provider_name.upper()}_TASKS"
    tasks_str = os.environ.get(env_var, "").lower().strip()
    
    if not tasks_str or tasks_str == "all":
        return TaskType.all_types()
    elif tasks_str == "none":
        return []
    else:
        return [t.strip() for t in tasks_str.split(",") if t.strip()]
