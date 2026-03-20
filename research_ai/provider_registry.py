"""Provider registry with load balancing, circuit breaker, and automatic rerouting.

Manages multiple LLM provider instances and routes requests to healthy providers.
"""

import os
import time
import threading
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

from .logger import log
from .providers import (
    ProviderBase, 
    ProviderStatus,
    TaskType,
    RateLimitError,
    OpenRouterProvider,
    GroqProvider,
    GoogleAIProvider,
    OllamaProvider,
    OpenAIProvider,
    parse_provider_tasks,
)


@dataclass
class ProviderConfig:
    """Configuration for initializing providers from environment."""
    provider_class: type
    env_keys_var: str  # e.g., "OPENROUTER_API_KEYS"
    env_model_var: str  # e.g., "OPENROUTER_MODELS"
    env_backup_model_var: str  # e.g., "OPENROUTER_BACKUP_MODEL"
    env_log_var: str  # e.g., "OPENROUTER_LOG"
    env_base_url_var: Optional[str] = None  # e.g., "OPENROUTER_API_BASE"
    requires_api_key: bool = True


# Provider configurations for initialization
PROVIDER_CONFIGS = {
    "openrouter": ProviderConfig(
        provider_class=OpenRouterProvider,
        env_keys_var="OPENROUTER_API_KEYS",
        env_model_var="OPENROUTER_MODELS",
        env_backup_model_var="OPENROUTER_BACKUP_MODEL",
        env_log_var="OPENROUTER_LOG",
        env_base_url_var="OPENROUTER_API_BASE",
    ),
    "groq": ProviderConfig(
        provider_class=GroqProvider,
        env_keys_var="GROQ_API_KEYS",
        env_model_var="GROQ_MODELS",
        env_backup_model_var="GROQ_BACKUP_MODEL",
        env_log_var="GROQ_LOG",
        env_base_url_var="GROQ_BASE_URL",
    ),
    "google_ai": ProviderConfig(
        provider_class=GoogleAIProvider,
        env_keys_var="GOOGLE_AI_KEYS",
        env_model_var="GOOGLE_AI_MODELS",
        env_backup_model_var="GOOGLE_AI_BACKUP_MODEL",
        env_log_var="GOOGLE_AI_LOG",
        env_base_url_var="GOOGLE_AI_BASE_URL",
    ),
    "ollama": ProviderConfig(
        provider_class=OllamaProvider,
        env_keys_var="OLLAMA_BASE_URLS",  # Ollama uses base URLs instead of API keys
        env_model_var="OLLAMA_MODELS",
        env_backup_model_var="OLLAMA_BACKUP_MODEL",
        env_log_var="OLLAMA_LOG",
        requires_api_key=False,
    ),
    "openai": ProviderConfig(
        provider_class=OpenAIProvider,
        env_keys_var="OPENAI_API_KEYS",
        env_model_var="OPENAI_MODELS",
        env_backup_model_var="OPENAI_BACKUP_MODEL",
        env_log_var="OPENAI_LOG",
        env_base_url_var="OPENAI_BASE_URL",
    ),
}


class ProviderRegistry:
    """Manages LLM providers with load balancing and automatic failover.
    
    Features:
    - Multiple providers and keys per provider type
    - Per-provider task assignments (PROVIDER_TASKS env var)
    - Round-robin load balancing across healthy providers
    - Circuit breaker pattern for failed providers
    - Automatic rerouting on rate limits
    """
    
    MAX_REROUTES = 10  # Maximum reroute attempts before giving up
    
    def __init__(self):
        self.providers: Dict[str, ProviderBase] = {}  # provider_id -> provider
        self.provider_order: List[str] = []  # For round-robin
        self.fallback_chain: List[str] = []  # Provider types in fallback order
        self._call_counter = 0
        self._lock = threading.Lock()
    
    def register(self, provider: ProviderBase):
        """Register a provider instance."""
        with self._lock:
            self.providers[provider.provider_id] = provider
            if provider.provider_id not in self.provider_order:
                self.provider_order.append(provider.provider_id)
            log.config(f"Registered provider: {provider.provider_id} (tasks: {provider.enabled_tasks})")
    
    def unregister(self, provider_id: str):
        """Unregister a provider instance."""
        with self._lock:
            if provider_id in self.providers:
                del self.providers[provider_id]
            if provider_id in self.provider_order:
                self.provider_order.remove(provider_id)
    
    def set_fallback_chain(self, chain: List[str]):
        """Set the fallback chain (provider types in priority order)."""
        self.fallback_chain = chain
    
    def get_providers_for_task(self, task_type: str) -> List[ProviderBase]:
        """Get all providers enabled for a task type."""
        with self._lock:
            return [
                p for p in self.providers.values()
                if p.is_enabled_for_task(task_type)
            ]
    
    def get_healthy_providers_for_task(self, task_type: str) -> List[ProviderBase]:
        """Get healthy providers enabled for a task type."""
        providers = self.get_providers_for_task(task_type)
        return [p for p in providers if p.is_healthy()]
    
    def select_provider(self, task_type: str, exclude: List[str] = None) -> Optional[ProviderBase]:
        """Select best available provider for a task using round-robin.
        
        Args:
            task_type: The task type to route
            exclude: Provider IDs to exclude (e.g., ones that just failed)
            
        Returns:
            Selected provider or None if no healthy providers available
        """
        exclude = exclude or []
        healthy = self.get_healthy_providers_for_task(task_type)
        available = [p for p in healthy if p.provider_id not in exclude]
        
        if not available:
            # Try fallback chain
            for provider_type in self.fallback_chain:
                for p in self.providers.values():
                    if (p.provider_type == provider_type and 
                        p.is_healthy() and 
                        p.provider_id not in exclude):
                        log.config(f"Using fallback provider {p.provider_id} for task {task_type}")
                        return p
            return None
        
        # Round-robin selection
        with self._lock:
            self._call_counter += 1
            idx = self._call_counter % len(available)
        
        return available[idx]
    
    def call_with_reroute(
        self,
        task_type: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: str = None,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Make LLM call with automatic rerouting on failures.
        
        This method ensures the call is never forgotten:
        1. Select provider for task type
        2. Attempt call
        3. On rate limit: mark provider, reroute to another
        4. On other failure: mark provider degraded, reroute
        5. Continue until success or all providers exhausted
        
        Args:
            task_type: The task type for routing
            messages: Chat messages
            temperature: Sampling temperature
            model: Optional model override
            
        Returns:
            Tuple of (content, usage)
            
        Raises:
            Exception: If all providers fail
        """
        attempted = []  # Track attempted provider IDs
        last_error = None
        
        for attempt in range(self.MAX_REROUTES):
            provider = self.select_provider(task_type, exclude=attempted)
            
            if not provider:
                # No healthy providers available
                if attempted:
                    raise Exception(
                        f"All providers exhausted for task {task_type}. "
                        f"Attempted: {attempted}. Last error: {last_error}"
                    )
                else:
                    raise Exception(f"No providers configured for task {task_type}")
            
            attempted.append(provider.provider_id)
            
            try:
                log.progress(f"[registry] Attempting {provider.provider_id} for {task_type}")
                result = provider.call(messages, temperature=temperature, model=model)
                provider.mark_success()
                return result
                
            except RateLimitError as e:
                log.warning(f"Rate limit on {provider.provider_id}, rerouting... ({e})")
                provider.mark_rate_limited(e.retry_after)
                last_error = e
                # Continue to try next provider
                
            except Exception as e:
                log.error(f"Provider {provider.provider_id} failed: {e}")
                provider.mark_failure()
                last_error = e
                # Continue to try next provider
        
        raise Exception(
            f"Failed after {self.MAX_REROUTES} reroute attempts for task {task_type}. "
            f"Attempted: {attempted}. Last error: {last_error}"
        )
    
    def get_all_usage(self) -> Dict[str, Dict]:
        """Get usage statistics for all providers."""
        with self._lock:
            return {pid: p.get_usage() for pid, p in self.providers.items()}
    
    def get_total_usage(self) -> Dict:
        """Get aggregated usage across all providers."""
        all_usage = self.get_all_usage()
        total = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for usage in all_usage.values():
            total["calls"] += usage.get("calls", 0)
            total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total["completion_tokens"] += usage.get("completion_tokens", 0)
            total["total_tokens"] += usage.get("total_tokens", 0)
        return total
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status for all providers."""
        with self._lock:
            return {pid: p.health.status.value for pid, p in self.providers.items()}
    
    def __len__(self) -> int:
        return len(self.providers)
    
    def __bool__(self) -> bool:
        return len(self.providers) > 0


def init_registry_from_env() -> ProviderRegistry:
    """Initialize provider registry from environment variables.
    
    Reads configuration for all supported providers and creates instances
    based on PROVIDER_TASKS settings.
    
    Environment variables:
        OPENROUTER_API_KEYS: Comma-separated API keys
        OPENROUTER_MODELS: Model to use
        OPENROUTER_TASKS: Task assignments (all/none/list)
        
        GROQ_API_KEYS: Comma-separated API keys
        GROQ_MODELS: Model to use
        GROQ_TASKS: Task assignments
        
        GOOGLE_AI_KEYS: Comma-separated API keys
        GOOGLE_AI_MODELS: Model to use
        GOOGLE_AI_TASKS: Task assignments
        
        OLLAMA_BASE_URLS: Comma-separated base URLs
        OLLAMA_MODELS: Model to use
        OLLAMA_TASKS: Task assignments
        
        OPENAI_API_KEYS: Comma-separated API keys
        OPENAI_MODELS: Model to use
        OPENAI_TASKS: Task assignments
        
        LLM_FALLBACK_CHAIN: Fallback provider types (e.g., "openrouter,groq,ollama")
    
    Returns:
        Configured ProviderRegistry instance
    """
    registry = ProviderRegistry()
    
    for provider_type, config in PROVIDER_CONFIGS.items():
        tasks = parse_provider_tasks(provider_type)
        
        if not tasks:
            # TASKS=none, skip this provider
            log.config(f"Provider {provider_type} disabled (TASKS=none)")
            continue
        
        # Get keys/URLs
        keys_str = os.environ.get(config.env_keys_var, "")
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
        if config.requires_api_key and not keys:
            # No API keys configured
            continue
        
        if not config.requires_api_key and not keys:
            # Ollama uses base URLs, default to localhost
            keys = ["http://localhost:11434"]
        
        # Get model
        model = os.environ.get(config.env_model_var, "")
        
        # Get backup model
        backup_model = os.environ.get(config.env_backup_model_var, "") or None
        
        # Get log setting
        log_requests = os.environ.get(config.env_log_var, "0") == "1"
        
        # Get base URL override
        base_url = None
        if config.env_base_url_var:
            base_url = os.environ.get(config.env_base_url_var, "")
        
        # Create provider instances
        for i, key in enumerate(keys):
            provider_id = f"{provider_type}_{i+1}" if len(keys) > 1 else provider_type
            
            try:
                if config.requires_api_key:
                    provider = config.provider_class(
                        provider_id=provider_id,
                        api_key=key,
                        model=model or None,
                        backup_model=backup_model,
                        base_url=base_url or None,
                        enabled_tasks=tasks,
                        log_requests=log_requests,
                    )
                else:
                    # Ollama - key is actually base_url
                    provider = config.provider_class(
                        provider_id=provider_id,
                        base_url=key,
                        model=model or None,
                        backup_model=backup_model,
                        enabled_tasks=tasks,
                        log_requests=log_requests,
                    )
                
                registry.register(provider)
                
            except Exception as e:
                log.error(f"Failed to create provider {provider_id}: {e}")
    
    # Set fallback chain
    fallback_str = os.environ.get("LLM_FALLBACK_CHAIN", "openrouter,groq,google_ai,openai,ollama")
    fallback_chain = [f.strip() for f in fallback_str.split(",") if f.strip()]
    registry.set_fallback_chain(fallback_chain)
    
    log.config(f"Provider registry initialized with {len(registry)} providers")
    log.config(f"Fallback chain: {fallback_chain}")
    
    return registry


def is_multi_provider_config() -> bool:
    """Check if multi-provider configuration is being used.
    
    Returns True if any of these conditions are met:
    - Multiple API keys for any provider (comma-separated)
    - Any PROVIDER_TASKS env var is set
    - LLM_FALLBACK_CHAIN is set
    """
    # Check for multiple keys
    for config in PROVIDER_CONFIGS.values():
        keys_str = os.environ.get(config.env_keys_var, "")
        if "," in keys_str:
            return True
    
    # Check for task assignments
    for provider_type in PROVIDER_CONFIGS.keys():
        if os.environ.get(f"{provider_type.upper()}_TASKS"):
            return True
    
    # Check for fallback chain
    if os.environ.get("LLM_FALLBACK_CHAIN"):
        return True
    
    return False
