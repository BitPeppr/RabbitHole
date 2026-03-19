"""Command-line entrypoint for ResearchAI.

This simplified CLI uses a .env file for configuration and only supports the OpenRouter provider.
Only a single positional topic argument is accepted; all other settings are loaded from the .env file.
"""

import asyncio
import os
import sys

from .datastore import Datastore
from .llm import LLM
from .web_search import WebSearchConnector
from .orchestrator import Orchestrator
from .logger import log


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
    root = os.path.abspath(os.getcwd())
    db_path = os.environ.get("DB_PATH", os.path.join(root, "research_ai", "state.db"))
    ds = Datastore(db_path)
    ds.init()

    # Initialize executor limiter and log concurrency settings
    from .executor_limiter import get_max_tasks
    max_tasks = get_max_tasks()
    log.config(f"MAX_CONCURRENT_TASKS={max_tasks} (controls system-wide parallelism)")

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


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "Sample topic: impacts of AI on labor"
    try:
        asyncio.run(main_async(topic))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
