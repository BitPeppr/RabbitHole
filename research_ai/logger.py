"""Colored logging utility for ResearchAI.

Provides thread-safe colored console logging with category-based formatting.
Respects NO_COLOR and FORCE_COLOR environment variables.
"""

import os
import sys
import threading
from datetime import datetime


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    BRIGHT_GREEN = "\033[92m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"


def _supports_color() -> bool:
    """Detect if terminal supports colors."""
    # FORCE_COLOR=1 always enables colors
    if os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes"):
        return True
    # NO_COLOR disables colors (https://no-color.org/)
    if os.environ.get("NO_COLOR", ""):
        return False
    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def _should_timestamp() -> bool:
    """Check if timestamps should be included."""
    return os.environ.get("LOG_TIMESTAMPS", "").lower() in ("1", "true", "yes")


def format_tokens(count: int) -> str:
    """Format token counts nicely (e.g., 1,234 or 1.2K)."""
    if count < 1000:
        return str(count)
    elif count < 10000:
        return f"{count:,}"
    elif count < 1000000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1000000:.1f}M"


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max length, adding ellipsis if needed."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def make_progress_bar(percent: float, width: int = 20, use_color: bool = True) -> str:
    """Create a text-based progress bar.
    
    Args:
        percent: Progress percentage (0-100)
        width: Width of the bar in characters
        use_color: Whether to use ANSI colors
    
    Returns:
        A string like "[████████░░░░░░░░░░░░] 42%"
    """
    percent = max(0, min(100, percent))
    filled = int(width * percent / 100)
    empty = width - filled
    
    bar_filled = "█" * filled
    bar_empty = "░" * empty
    
    if use_color:
        # Color the filled portion green, empty portion dim
        bar = f"{Colors.GREEN}{bar_filled}{Colors.DIM}{bar_empty}{Colors.RESET}"
    else:
        bar = f"{bar_filled}{bar_empty}"
    
    return f"[{bar}] {percent:3.0f}%"


class Logger:
    """Thread-safe colored logger with category-based formatting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._use_color = _supports_color()

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self._use_color:
            return text
        return f"{color}{text}{Colors.RESET}"

    def _timestamp(self) -> str:
        """Get timestamp prefix if enabled."""
        if not _should_timestamp():
            return ""
        return datetime.now().strftime("%H:%M:%S ")

    def _print(self, message: str):
        """Thread-safe print."""
        with self._lock:
            print(message)

    def config(self, message: str):
        """Log configuration/startup info (blue)."""
        prefix = self._colorize("[config]", Colors.BLUE)
        ts = self._timestamp()
        self._print(f"{ts}{prefix} {message}")

    def progress(self, message: str):
        """Log progress heartbeat updates (cyan)."""
        prefix = self._colorize("[progress]", Colors.CYAN)
        ts = self._timestamp()
        self._print(f"{ts}{prefix} {message}")

    def worker(self, worker_id: int, action: str, task_id: str, topic: str = None):
        """Log worker start/done messages (green).

        Args:
            worker_id: The worker number
            action: Action like 'start' or 'done'
            task_id: The task identifier
            topic: Optional topic (will be truncated)
        """
        prefix = self._colorize(f"[worker:{worker_id}]", Colors.GREEN)
        ts = self._timestamp()
        msg = f"{ts}{prefix} {action} task={task_id}"
        if topic:
            msg += f" topic={truncate(topic, 60)}"
        self._print(msg)

    def worker_done(self, worker_id: int, task_id: str, docs: int = 0, spawned: int = 0):
        """Log worker completion with stats (green).

        Args:
            worker_id: The worker number
            task_id: The task identifier
            docs: Number of documents processed
            spawned: Number of child tasks spawned
        """
        prefix = self._colorize(f"[worker:{worker_id}]", Colors.GREEN)
        ts = self._timestamp()
        self._print(f"{ts}{prefix} done task={task_id} docs={docs} spawned={spawned}")

    def queue(self, operation: str, task_id: str, **kwargs):
        """Log queue operations (yellow).

        Args:
            operation: Operation like '+task' or '-task'
            task_id: The task identifier
            **kwargs: Additional key=value pairs to log
        """
        prefix = self._colorize("[queue]", Colors.YELLOW)
        ts = self._timestamp()
        msg = f"{ts}{prefix} {operation}={task_id}"
        for key, value in kwargs.items():
            if key == "topic" and value:
                value = truncate(str(value), 60)
            msg += f" {key}={value}"
        self._print(msg)

    def error(self, message: str):
        """Log errors (red)."""
        prefix = self._colorize("[error]", Colors.RED)
        ts = self._timestamp()
        self._print(f"{ts}{prefix} {message}")

    def success(self, message: str):
        """Log successful completions (bright green)."""
        prefix = self._colorize("[success]", Colors.BRIGHT_GREEN)
        ts = self._timestamp()
        self._print(f"{ts}{prefix} {message}")

    def warning(self, message: str):
        """Log warnings (yellow)."""
        prefix = self._colorize("[warning]", Colors.YELLOW)
        ts = self._timestamp()
        self._print(f"{ts}{prefix} {message}")

    def progress_stats(
        self,
        pending: int = 0,
        in_progress: int = 0,
        done: int = 0,
        llm_calls: int = 0,
        llm_tokens: int = 0,
        estimated_total: int = None,
    ):
        """Log formatted progress statistics with progress bar.

        Args:
            pending: Number of pending tasks
            in_progress: Number of in-progress tasks
            done: Number of completed tasks
            llm_calls: Number of LLM API calls
            llm_tokens: Total LLM tokens used
            estimated_total: Estimated total tasks (for percentage calculation)
        """
        tokens_fmt = format_tokens(llm_tokens)
        
        # Calculate progress percentage
        if estimated_total and estimated_total > 0:
            # Use actual progress but cap at 90% until truly done
            actual_total = pending + in_progress + done
            if actual_total > estimated_total:
                estimated_total = actual_total
            percent = (done / estimated_total) * 100
            # Cap at 90% while work is still pending/in_progress
            if pending > 0 or in_progress > 0:
                percent = min(percent, 90)
            bar = make_progress_bar(percent, width=20, use_color=self._use_color)
        else:
            # Fallback: estimate based on done vs total known tasks
            total_known = pending + in_progress + done
            if total_known > 0:
                percent = (done / total_known) * 100
                if pending > 0 or in_progress > 0:
                    percent = min(percent, 90)
                bar = make_progress_bar(percent, width=20, use_color=self._use_color)
            else:
                bar = make_progress_bar(0, width=20, use_color=self._use_color)
        
        msg = (
            f"{bar} "
            f"tasks: {done}/{pending + in_progress + done} "
            f"(+{in_progress} active) "
            f"llm: {llm_calls} calls, {tokens_fmt} tokens"
        )
        self.progress(msg)

    def job_started(
        self,
        job_id: str,
        depth: int,
        children: int,
        concurrency: int,
        max_tasks: int = None,
    ):
        """Log job start information.

        Args:
            job_id: The job identifier
            depth: Maximum depth setting
            children: Maximum children per task
            concurrency: Worker concurrency level
            max_tasks: Maximum concurrent tasks (optional)
        """
        msg = f"job={job_id} started depth={depth} children={children} concurrency={concurrency}"
        if max_tasks is not None:
            msg += f" max_concurrent_tasks={max_tasks}"
        self.progress(msg)

    def job_completed(self, job_id: str, agents: int, output: str):
        """Log job completion.

        Args:
            job_id: The job identifier
            agents: Number of agents that ran
            output: Output file path
        """
        self.progress(f"job={job_id} completed agents={agents} output={output}")


# Singleton logger instance
log = Logger()
