"""Command-line entrypoint for RabbitHole.

Configuration is loaded from ~/.config/rabbithole/rabbithole.conf (or path specified 
by RABBITHOLE_CONFIG env var). On first run, creates a template config file and exits.

Supports:
- Legacy single-provider mode (OPENROUTER_API_KEY)
- Multi-provider mode with automatic load balancing (OPENROUTER_API_KEYS, GROQ_API_KEYS, etc.)

Only a single positional topic argument is accepted; all other settings are loaded from config.
"""

import asyncio
import os
import shutil
import signal
import sys
import tempfile

from .datastore import Datastore
from .llm import LLM
from .web_search import WebSearchConnector
from .orchestrator import Orchestrator
from .logger import log


# ---------------------------------------------------------------------------
# Config file handling
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.expanduser("~/.config/rabbithole")
CONFIG_FILE = os.path.join(CONFIG_DIR, "rabbithole.conf")

CONFIG_TEMPLATE = """\
# RabbitHole configuration file
# Replace placeholder values with your real API keys.

# =============================================================================
# LLM PROVIDERS
# =============================================================================
# Configure one or more providers. The system uses all available providers
# with load balancing and automatic failover on rate limits.
#
# Task types: summarization, subtopic, validation, report, recommendations, research

# --- OpenRouter (recommended - many free models) ---
OPENROUTER_API_KEYS=your_openrouter_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api
OPENROUTER_MODELS=stepfun/step-3.5-flash:free
OPENROUTER_BACKUP_MODEL=arcee-ai/trinity-large-preview:free
OPENROUTER_TASKS=all
OPENROUTER_LOG=1

# --- Groq (fast inference) ---
# GROQ_API_KEYS=your_groq_api_key_here
# GROQ_MODELS=moonshotai/kimi-k2-instruct-0905
# GROQ_BACKUP_MODEL=llama-3.3-70b-versatile
# GROQ_TASKS=report
# GROQ_LOG=1

# --- Google AI Studio ---
# GOOGLE_AI_KEYS=your_google_ai_key_here
# GOOGLE_AI_MODELS=gemini-2.0-flash-exp
# GOOGLE_AI_TASKS=none

# --- Ollama (local models) ---
# OLLAMA_BASE_URLS=http://localhost:11434
# OLLAMA_MODELS=llama3.1:70b
# OLLAMA_TASKS=none

# --- OpenAI (direct API) ---
# OPENAI_API_KEYS=your_openai_api_key_here
# OPENAI_MODELS=gpt-4o-mini
# OPENAI_TASKS=none

# --- Provider fallback order ---
LLM_FALLBACK_CHAIN=openrouter,groq,google_ai,openai,ollama

# =============================================================================
# RUNTIME SETTINGS
# =============================================================================
AUTO_CLEANUP=1
DB_PATH=state.db
OUTPUT_PATH=research_report.md

# Orchestrator defaults
MAX_DEPTH=3
MAX_CHILDREN=4
CONCURRENCY=30
MAX_CONCURRENT_TASKS=35

# Web search
CONNECTOR=http
SOURCE_COUNT=7
SEARCH_PROVIDER=bing
SEARCH_FALLBACK_CHAIN=wikipedia,arxiv
WEB_SEARCH_DELAY_SEC=1.0
USE_PLAYWRIGHT=1

# Progress display
PROGRESS_LOG=1
PROGRESS_VERBOSE_TASKS=1
PROGRESS_INTERVAL_SEC=5
PROGRESS_DISPLAY_MODE=tui

# Output length
SUMMARY_WORD_COUNT=600
REPORT_MIN_WORDS=2000
REPORT_MAX_WORDS=8000
"""


def _ensure_config() -> str:
    """Ensure config file exists. Returns path to config file.
    
    On first run (no config file), creates template and exits with instructions.
    """
    # Allow override via environment variable
    config_path = os.environ.get("RABBITHOLE_CONFIG", CONFIG_FILE)
    
    # If using custom path and it exists, use it
    if config_path != CONFIG_FILE and os.path.exists(config_path):
        return config_path
    
    # Check if default config exists
    if os.path.exists(CONFIG_FILE):
        return CONFIG_FILE
    
    # First run - create config directory and template
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(CONFIG_TEMPLATE)
    
    print("\n" + "=" * 60)
    print("🐰 RabbitHole - First Run Setup")
    print("=" * 60)
    print(f"\nCreated config file at:\n  {CONFIG_FILE}")
    print("\nPlease edit this file and add your API key(s).")
    print("At minimum, you need ONE provider configured.")
    print("\nRecommended: Get a free OpenRouter API key at:")
    print("  https://openrouter.ai/keys")
    print("\nThen run rabbithole again with your topic.")
    print("=" * 60 + "\n")
    sys.exit(0)


def _validate_config(config_path: str) -> bool:
    """Check if config has real API keys (not placeholders).
    
    Returns True if valid, False if only placeholders found.
    """
    placeholders = [
        "your_openrouter_api_key_here",
        "your_groq_api_key_here", 
        "your_google_ai_key_here",
        "your_openai_api_key_here",
        "your_brave_api_key_here",
        "your_serpapi_api_key_here",
        "your_tavily_api_key_here",
        "your_exa_api_key_here",
    ]
    
    has_real_key = False
    
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            
            # Check for API key fields
            if any(x in key.upper() for x in ["API_KEY", "API_KEYS"]):
                if value and value not in placeholders:
                    has_real_key = True
                    break
            
            # Also check for Ollama base URLs (doesn't need API key)
            if key == "OLLAMA_BASE_URLS" and value:
                has_real_key = True
                break
    
    return has_real_key


def _get_runtime_dir() -> str:
    """Get the runtime directory for transient files.
    
    Uses RUNTIME_DIR env var if set, otherwise uses system temp directory.
    For packaging (pipx/homebrew), this ensures no files are written to the project.
    """
    if os.environ.get("RUNTIME_DIR"):
        return os.environ["RUNTIME_DIR"]
    # Use system temp directory for clean packaging
    base_tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
    runtime_dir = os.path.join(base_tmp, "rabbithole")
    os.makedirs(runtime_dir, exist_ok=True)
    return runtime_dir


def _cleanup_runtime(runtime_dir: str, keep_report: bool = True):
    """Clean up all runtime artifacts (db, cache, artifacts) after job completion.
    
    Args:
        runtime_dir: The runtime directory to clean
        keep_report: If True, don't delete anything (report is kept at OUTPUT_PATH)
    """
    # Clear in-memory caches
    try:
        from .llm import _summary_cache
        _summary_cache.clear()
    except Exception:
        pass
    try:
        from .web_search import _url_cache
        _url_cache.clear()
    except Exception:
        pass
    
    # Delete runtime directory contents (db, artifacts, logs)
    if runtime_dir and os.path.isdir(runtime_dir):
        try:
            # Only delete if it's in a temp location (safety check)
            base_tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
            if runtime_dir.startswith(base_tmp) or runtime_dir.startswith("/tmp"):
                shutil.rmtree(runtime_dir, ignore_errors=True)
                log.progress("[cleanup] Runtime directory cleared")
            else:
                # For custom RUNTIME_DIR, just clear contents but keep the dir
                for item in os.listdir(runtime_dir):
                    item_path = os.path.join(runtime_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)
                log.progress("[cleanup] Runtime contents cleared")
        except Exception as e:
            log.warning(f"Cleanup warning: {e}")


def _handle_interrupt(signum, frame):
    """Handle Ctrl+C cleanly - immediate exit with clean message."""
    # Finalize TUI mode if active (print newline after progress bar)
    log.finalize_tui()
    print("\n\033[33m⚡ Task cancelled\033[0m")
    # Use os._exit to forcefully terminate without waiting for threads
    os._exit(0)


# Install signal handler early
signal.signal(signal.SIGINT, _handle_interrupt)


def _load_config(path: str):
    """Load config file into environment variables."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Handle inline comments (e.g., "value # comment")
            if "#" in v:
                v = v.split("#")[0].strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            # do not override existing env vars
            if k not in os.environ:
                os.environ[k] = v


async def main_async(topic: str):
    # Set up runtime directory (uses /tmp/rabbithole by default for clean packaging)
    runtime_dir = _get_runtime_dir()
    os.environ["RUNTIME_DIR"] = runtime_dir  # Propagate to datastore
    
    # Check if auto-cleanup is enabled (default: True)
    auto_cleanup = os.environ.get("AUTO_CLEANUP", "1").lower() not in ("0", "false", "no")
    
    db_path = os.environ.get("DB_PATH", "state.db")  # Relative path goes into runtime_dir
    ds = Datastore(db_path)
    ds.init()

    # Initialize executor limiter and log concurrency settings
    from .executor_limiter import get_max_tasks
    max_tasks = get_max_tasks()
    log.config(f"MAX_CONCURRENT_TASKS={max_tasks} (controls system-wide parallelism)")
    log.config(f"RUNTIME_DIR={runtime_dir}")

    # Initialize LLM (auto-detects single vs multi-provider mode)
    llm = LLM(provider="openrouter")
    
    # Log provider configuration
    if llm.use_multi_provider:
        log.config(f"Multi-provider mode enabled with {len(llm.registry)} provider instances")
        health = llm.get_provider_health()
        for pid, status in health.items():
            log.config(f"  - {pid}: {status}")
    elif llm.use_openrouter:
        log.config("Legacy mode: OpenRouter provider")
    elif llm.use_openai:
        log.config("Legacy mode: OpenAI provider")
    else:
        log.warning("No LLM provider configured - will use local fallbacks")

    connector = WebSearchConnector()
    log.config("connector=web_search")

    # optional lightweight embeddings for local retrieval
    try:
        from .embeddings import Embeddings

        embeddings = Embeddings()
    except Exception:
        embeddings = None

    concurrency = int(os.environ.get("CONCURRENCY", "1"))
    max_depth = int(os.environ.get("MAX_DEPTH", "2"))
    max_children = int(os.environ.get("MAX_CHILDREN", "2"))
    output = os.environ.get("OUTPUT_PATH", "research_report.md")

    orch = Orchestrator(datastore=ds, llm=llm, connector=connector, concurrency=concurrency, embeddings=embeddings)
    out = await orch.run_job(topic=topic, max_depth=max_depth, max_children=max_children, output_path=output, job_id=None)
    log.success(f"Report written to {out}")
    if getattr(orch, "job_id", None):
        print("Job ID:", orch.job_id)
    
    # Auto-cleanup runtime artifacts after successful completion
    if auto_cleanup:
        # Close database connection first
        try:
            if ds.conn:
                ds.conn.close()
        except Exception:
            pass
        _cleanup_runtime(runtime_dir)
        log.success("Runtime cleanup complete (db, cache, artifacts cleared)")


def main():
    # Handle --help and --version before config check
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: rabbithole <topic> [--no-cleanup]")
        print("\nRun deep research on a topic and generate a comprehensive report.")
        print(f"\nConfig: {CONFIG_FILE}")
        print("        (override with RABBITHOLE_CONFIG env var)")
        print("\nOptions:")
        print("  --no-cleanup    Keep runtime files (db, cache) after completion")
        print("  --help, -h      Show this message")
        print("  --version, -V   Show version")
        return
    
    if "--version" in sys.argv or "-V" in sys.argv:
        from . import __version__
        print(f"rabbithole {__version__}")
        return
    
    # Ensure config exists (creates template on first run and exits)
    config_path = _ensure_config()
    
    # Validate config has real API keys
    if not _validate_config(config_path):
        print("\n" + "=" * 60)
        print("⚠️  RabbitHole - Configuration Required")
        print("=" * 60)
        print(f"\nNo valid API keys found in:\n  {config_path}")
        print("\nPlease edit the config file and add at least one API key.")
        print("\nGet a free OpenRouter API key at:")
        print("  https://openrouter.ai/keys")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # Load config into environment
    _load_config(config_path)
    
    # Simple CLI: rabbithole "topic" [--no-cleanup]
    args = sys.argv[1:]
    no_cleanup = "--no-cleanup" in args
    if no_cleanup:
        args.remove("--no-cleanup")
        os.environ["AUTO_CLEANUP"] = "0"
    
    topic = args[0] if args else "Sample topic: impacts of AI on labor"
    asyncio.run(main_async(topic))


if __name__ == "__main__":
    main()
