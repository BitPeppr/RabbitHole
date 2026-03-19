"""Global executor and semaphore for strict sequential execution.

This module provides a singleton ThreadPoolExecutor and asyncio Semaphore to ensure
only MAX_CONCURRENT_TASKS operations run simultaneously across the entire system.

When MAX_CONCURRENT_TASKS=1 (default), the system becomes fully sequential:
- Only 1 agent runs at a time
- Only 1 HTTP request at a time  
- Only 1 LLM API call at a time

This ensures that as you scale depth/layers/children, only TIME increases, not memory or CPU usage.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class ExecutorLimiter:
    """Singleton that manages a global thread pool and semaphore."""
    
    _instance: Optional['ExecutorLimiter'] = None
    
    def __init__(self):
        # Read MAX_CONCURRENT_TASKS from env (default 1 for fully sequential execution)
        try:
            max_tasks = int(os.environ.get("MAX_CONCURRENT_TASKS", "1"))
            max_tasks = max(1, max_tasks)  # at least 1
        except Exception:
            max_tasks = 1
        
        self.max_tasks = max_tasks
        # ThreadPoolExecutor with exactly max_tasks workers
        self.executor = ThreadPoolExecutor(max_workers=max_tasks, thread_name_prefix="research")
        # Asyncio semaphore to limit concurrent async operations
        self.semaphore = asyncio.Semaphore(max_tasks)
        
    @classmethod
    def get_instance(cls) -> 'ExecutorLimiter':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = ExecutorLimiter()
        return cls._instance
    
    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get the global limited executor."""
        return cls.get_instance().executor
    
    @classmethod
    def get_semaphore(cls) -> asyncio.Semaphore:
        """Get the global semaphore."""
        return cls.get_instance().semaphore
    
    @classmethod
    def get_max_tasks(cls) -> int:
        """Get the configured max concurrent tasks."""
        return cls.get_instance().max_tasks


# Convenience functions for easy import
def get_executor() -> ThreadPoolExecutor:
    """Get the global limited thread pool executor."""
    return ExecutorLimiter.get_executor()


def get_semaphore() -> asyncio.Semaphore:
    """Get the global concurrency-limiting semaphore."""
    return ExecutorLimiter.get_semaphore()


def get_max_tasks() -> int:
    """Get MAX_CONCURRENT_TASKS setting."""
    return ExecutorLimiter.get_max_tasks()
