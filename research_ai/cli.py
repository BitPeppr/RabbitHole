"""Command-line entrypoint for ResearchAI.

This simplified CLI uses a .env file for configuration and only supports the OpenRouter provider.
Only a single positional topic argument is accepted; all other settings are loaded from the .env file.
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


def _get_runtime_dir() -> str:
    """Get the runtime directory for transient files.
    
    Uses RUNTIME_DIR env var if set, otherwise uses system temp directory.
    For packaging (pipx/homebrew), this ensures no files are written to the project.
    """
    if os.environ.get("RUNTIME_DIR"):
        return os.environ["RUNTIME_DIR"]
    # Use system temp directory for clean packaging
    base_tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
    runtime_dir = os.path.join(base_tmp, "researchai")
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


def _load_dotenv(path: str = ".env"):
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
    _load_dotenv()
    
    # Set up runtime directory (uses /tmp/researchai by default for clean packaging)
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

    # Force OpenRouter provider
    llm = LLM(provider="openrouter")

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
    # Simple CLI: research_ai "topic" [--no-cleanup]
    args = sys.argv[1:]
    no_cleanup = "--no-cleanup" in args
    if no_cleanup:
        args.remove("--no-cleanup")
        os.environ["AUTO_CLEANUP"] = "0"
    
    topic = args[0] if args else "Sample topic: impacts of AI on labor"
    asyncio.run(main_async(topic))


if __name__ == "__main__":
    main()
