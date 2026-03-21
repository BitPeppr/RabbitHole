"""Colored logging utility for ResearchAI.

Provides thread-safe colored console logging with category-based formatting.
Respects NO_COLOR and FORCE_COLOR environment variables.

Supports two display modes via PROGRESS_DISPLAY_MODE env var:
- "tui": Sticky progress bar at bottom of terminal (updated in-place)
- "scroll": Standard scrolling logs (default)
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


# ANSI cursor control codes
class Cursor:
    SAVE = "\033[s"           # Save cursor position
    RESTORE = "\033[u"        # Restore cursor position
    MOVE_TO_BOTTOM = "\033[999;1H"  # Move to row 999 (will clamp to bottom)
    CLEAR_LINE = "\033[2K"    # Clear entire line
    MOVE_UP = "\033[1A"       # Move cursor up one line
    HIDE = "\033[?25l"        # Hide cursor
    SHOW = "\033[?25h"        # Show cursor


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


def _get_display_mode() -> str:
    """Get progress display mode: 'tui' for sticky bar, 'scroll' for normal."""
    mode = os.environ.get("PROGRESS_DISPLAY_MODE", "scroll").lower()
    return mode if mode in ("tui", "scroll") else "scroll"


def _get_terminal_size() -> tuple:
    """Get terminal size (columns, rows). Returns (80, 24) as fallback."""
    try:
        import shutil
        size = shutil.get_terminal_size((80, 24))
        return (size.columns, size.lines)
    except Exception:
        return (80, 24)


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
    """Thread-safe colored logger with category-based formatting.
    
    Supports two display modes:
    - scroll: Standard logging with each message on a new line
    - tui: Progress bar stays at bottom of terminal, updated in-place
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._use_color = _supports_color()
        self._tui_mode = None  # Lazy-evaluated on first use
        self._progress_line = ""  # Last progress bar content for TUI mode
        self._tui_initialized = False

    def _is_tui_mode(self) -> bool:
        """Check if TUI mode is enabled (lazy evaluation to allow .env loading)."""
        if self._tui_mode is None:
            self._tui_mode = _get_display_mode() == "tui"
        return self._tui_mode

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
        """Thread-safe print. In TUI mode, prints above the sticky progress bar."""
        with self._lock:
            if self._is_tui_mode() and self._tui_initialized and self._progress_line:
                # Clear current line, print message with newline, then redraw progress bar
                sys.stdout.write(f"\r{Cursor.CLEAR_LINE}{message}\n{self._progress_line}")
                sys.stdout.flush()
            else:
                print(message)

    def _print_progress_tui(self, message: str):
        """Print progress bar in TUI mode (sticky at current line, updated in-place)."""
        with self._lock:
            self._progress_line = message
            # Always update in place: carriage return, clear line, print new content
            sys.stdout.write(f"\r{Cursor.CLEAR_LINE}{message}")
            sys.stdout.flush()
            self._tui_initialized = True

    def finalize_tui(self):
        """Call at end of job to finalize TUI mode (print newline after progress bar)."""
        if self._is_tui_mode() and self._tui_initialized:
            with self._lock:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._tui_initialized = False

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
        elapsed_sec: float = 0,
    ):
        """Log formatted progress statistics with progress bar and ETA.

        Args:
            pending: Number of pending tasks
            in_progress: Number of in-progress tasks
            done: Number of completed tasks
            llm_calls: Number of LLM API calls
            llm_tokens: Total LLM tokens used
            estimated_total: Estimated total tasks (for percentage calculation)
            elapsed_sec: Seconds elapsed since job start (for ETA calculation)
        """
        tokens_fmt = format_tokens(llm_tokens)
        
        # Calculate progress percentage
        percent = 0
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
        
        # Calculate ETA based on elapsed time and progress
        # Use a "work done" metric that includes partial credit for in_progress tasks
        eta_str = ""
        work_done = done + (in_progress * 0.5)  # Count in-progress as 50% done
        total_work = pending + in_progress + done
        work_percent = (work_done / total_work * 100) if total_work > 0 else 0
        
        if elapsed_sec > 30 and work_percent > 1:  # Need some data to estimate
            # Time per percent of work, extrapolate to remaining
            remaining_percent = 100 - work_percent
            time_per_percent = elapsed_sec / work_percent
            eta_sec = time_per_percent * remaining_percent
            eta_str = f" ETA: {self._format_duration(eta_sec)}"
        elif elapsed_sec > 0:
            # Show elapsed time while building estimate
            eta_str = f" ({self._format_duration(elapsed_sec)} elapsed)"
        
        msg = (
            f"{bar} "
            f"tasks: {done}/{pending + in_progress + done} "
            f"(+{in_progress} active) "
            f"llm: {llm_calls} calls, {tokens_fmt} tokens"
            f"{eta_str}"
        )
        # In TUI mode, update progress bar in-place; otherwise use standard logging
        if self._is_tui_mode():
            self._print_progress_tui(msg)
        else:
            self.progress(msg)

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration (e.g., '5m 30s', '1h 15m')."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s" if secs > 0 else f"{mins}m"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

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
