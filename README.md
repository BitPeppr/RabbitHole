# ResearchAI — Deep Research Orchestrator

> A recursive, multi-agent research system that performs deep research on any topic and generates comprehensive reports. Features fully sequential execution for constant memory usage.

## Table of Contents

- [Quick Start](#quick-start)
- [Example Outputs](#example-outputs)
- [What This Does](#what-this-does)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Understanding Sequential Execution](#understanding-sequential-execution)
- [Project Structure](#project-structure)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

### 1. Install (30 seconds)

```bash
cd ResearchAI
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure (Edit .env file)

```bash
# Required: Get your key at https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Research scope
MAX_DEPTH=5         # Recursion depth (2-10)
MAX_CHILDREN=12     # Children per agent (2-20)
SOURCE_COUNT=50     # Sources per topic (10-100)

# Resource control (IMPORTANT!)
MAX_CONCURRENT_TASKS=1   # 1=sequential, 4=parallel
```

### 3. Run Research

```bash
python -m research_ai.cli "Your research topic here"

# Keep runtime files for debugging (db, cache, artifacts):
python -m research_ai.cli "Your topic" --no-cleanup
```

Output saved to: `research_report.md`

Runtime files (db, cache) are stored in `$TMPDIR/researchai` and automatically cleaned up after each run. This keeps the project directory clean for packaging (pipx/homebrew).

---

## Example Outputs

See `example output/` for sample reports. Note that "AI Ethics (2 hours)" used deeper recursion with more agents but a lower target word count, while "Uses for LLMs (6 minutes)" generated to a higher word target in 6 minutes with less recursion. Both are of quite good quality and depth.

---

## What This Does

ResearchAI creates a **tree of specialized research agents** that recursively explore your topic:

```
Your Topic: "History of the Byzantine Empire"
    ↓
Root Agent: Fetches 50 sources, summarizes, derives subtopics
    ├─ "Early Byzantine Period (330-610)"
    │   ├─ "Constantine's founding of Constantinople"
    │   ├─ "Justinian's reconquest campaigns"
    │   └─ "Codification of Roman law"
    ├─ "Byzantine military tactics"
    │   ├─ "Greek fire technology"
    │   └─ "Theme system organization"
    └─ "Byzantine art and architecture"
        ├─ "Hagia Sophia construction"
        └─ "Icon veneration controversies"
```

Each agent:

1. Fetches N sources (default: 50) from the web
2. Summarizes each source with LLM
3. Derives child topics from summaries
4. Spawns child agents (up to MAX_CHILDREN)
5. Continues recursively up to MAX_DEPTH levels

**Final Output**: A comprehensive Markdown report with:

- 200-word executive summary
- Table of contents
- Detailed sections for each researched topic
- Source citations with URLs
- Full provenance appendix

---

## Key Features

### 🎯 Deep, Recursive Research

- Agents spawn sub-agents up to configurable depth
- Each agent specializes in a subtopic
- Potentially thousands of sources analyzed

### 🔒 Sequential Execution (Default)

- **Only 1 operation at a time** when `MAX_CONCURRENT_TASKS=1`
- Constant memory usage (~500MB) regardless of depth
- Scales by TIME only, not resources
- Safe for limited hardware (4GB RAM)

### ⚡ Configurable Parallelism

- Increase `MAX_CONCURRENT_TASKS` for faster results
- Trade memory for speed when you have RAM
- Up to 8× faster with more concurrent tasks

### 🧹 Clean Runtime

- Runtime files (db, cache) stored in system temp directory (`$TMPDIR/researchai`)
- Automatic cleanup after each run (configurable via `AUTO_CLEANUP`)
- Project directory stays clean — ready for pipx/homebrew packaging
- Content-hash caching for summaries avoids redundant LLM calls

### 🌐 Real Web Search

- **Multiple search backends**: Brave, SerpAPI, Tavily, Exa, Bing, Wikipedia, arXiv
- **Configurable fallback chain**: Define provider order (e.g., `brave,serpapi,bing,wikipedia`)
- Fetches actual online sources, not simulated data

### 📊 Progress Tracking

- Real-time logs of pending/in-progress/done tasks
- API call and token usage monitoring
- Detailed task start/completion logs
- **Colored console output** with category-based formatting
- Configurable via `NO_COLOR`, `FORCE_COLOR`, `LOG_TIMESTAMPS`

### 💰 Budget Controls

- Optional limits on tokens, calls, time, cost
- Automatic stopping when budgets exceeded

---

## Installation

### Prerequisites

- Python 3.9 or higher
- 1GB RAM minimum (2-4GB+ recommended for parallel execution)
- Internet connection
- OpenRouter API key (free tier available)

### Steps

1. **Clone or navigate to the project**:

   ```bash
   cd /path/to/ResearchAI
   ```

2. **Create virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Get OpenRouter API key**:
   - Visit https://openrouter.ai/keys
   - Sign up (free tier available)
   - Generate an API key
   - Add to `.env` file (see Configuration)

---

## Configuration

All configuration is done through the `.env` file in the project root.

### Essential Settings

```bash
# OpenRouter API (Required)
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_API_BASE=https://openrouter.ai/api
OPENROUTER_MODEL=arcee-ai/trinity-large-preview:free
OPENROUTER_LOG=1  # Log API requests (1=on, 0=off)

# Research Scope
MAX_DEPTH=5         # Recursion depth (2-10)
MAX_CHILDREN=12     # Children per agent (2-20)
SOURCE_COUNT=50     # Sources to fetch per topic (10-100)

# Resource Control (CRITICAL!)
MAX_CONCURRENT_TASKS=1    # System-wide parallelism limit
# 1 = Fully sequential (default, safest)
# 2-4 = Limited parallel (requires 8-16GB RAM)
# 8+ = High parallel (requires 32GB+ RAM)

CONCURRENCY=1       # Number of worker coroutines (usually 1)

# Output and Storage
OUTPUT_PATH=research_report.md
DB_PATH=runtime/research_ai/state.db

# Progress Logging
PROGRESS_LOG=1                  # Enable progress logs
PROGRESS_VERBOSE_TASKS=1        # Show detailed task logs
PROGRESS_INTERVAL_SEC=5         # Heartbeat interval

# Console Output Formatting
NO_COLOR=0                      # Set to 1 to disable colored output
FORCE_COLOR=0                   # Set to 1 to force colors (even in non-TTY)
LOG_TIMESTAMPS=0                # Set to 1 to add timestamps to logs
```

### Popular OpenRouter Models

```bash
# Free models
OPENROUTER_MODEL=arcee-ai/trinity-large-preview:free
OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free

# Paid models (fast & cheap)
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_MODEL=anthropic/claude-3-haiku

# Paid models (high quality)
OPENROUTER_MODEL=openai/gpt-4o
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

See all models: https://openrouter.ai/models

### Configuration Presets

#### Quick Test (5 min, 500MB RAM)

```bash
MAX_DEPTH=2
MAX_CHILDREN=3
MAX_CONCURRENT_TASKS=1
SOURCE_COUNT=10
```

#### Standard Research (1 hour, 500MB RAM)

```bash
MAX_DEPTH=3
MAX_CHILDREN=5
MAX_CONCURRENT_TASKS=1
SOURCE_COUNT=25
```

#### Deep Dive (4+ hours, 500MB RAM)

```bash
MAX_DEPTH=4
MAX_CHILDREN=8
MAX_CONCURRENT_TASKS=1
SOURCE_COUNT=50
```

#### Fast Research (30 min, 2GB RAM)

```bash
MAX_DEPTH=3
MAX_CHILDREN=5
MAX_CONCURRENT_TASKS=4
SOURCE_COUNT=25
```

### Multi-Provider Mode (Advanced)

For higher throughput and automatic failover, you can configure multiple LLM providers with per-provider task assignments:

```bash
# Multiple OpenRouter keys (load balanced)
OPENROUTER_API_KEYS=sk-or-v1-key1,sk-or-v1-key2,sk-or-v1-key3
OPENROUTER_MODELS=arcee-ai/trinity-large-preview:free
OPENROUTER_TASKS=all  # Enable for all task types

# Groq (fast inference, disabled in this example)
GROQ_API_KEYS=gsk_key1,gsk_key2
GROQ_MODELS=llama-3.3-70b-versatile
GROQ_TASKS=none  # Keys stored but not used

# Google AI Studio (specific tasks only)
GOOGLE_AI_KEYS=AIza...key1
GOOGLE_AI_MODELS=gemini-2.0-flash-exp
GOOGLE_AI_TASKS=summarization,validation  # Only these tasks

# Ollama (local model for high-quality report generation)
OLLAMA_BASE_URLS=http://localhost:11434
OLLAMA_MODELS=llama3.1:70b
OLLAMA_TASKS=report  # Only final report synthesis

# Provider fallback order
LLM_FALLBACK_CHAIN=openrouter,groq,google_ai,ollama
```

**Task Types**: `all`, `none`, `summarization`, `subtopic`, `validation`, `report`, `recommendations`, `research`

**Features**:
- **Load balancing**: Calls distributed across all healthy providers for a task type
- **Automatic failover**: Rate limits trigger instant rerouting to other providers
- **Per-provider tasks**: Assign different providers to different task types
- **Circuit breaker**: Failed providers temporarily disabled to prevent cascading failures

### Web Search Providers

Configure which search backend(s) to use for fetching web sources:

```bash
# Primary search provider
SEARCH_PROVIDER=bing              # Default (free, no API key)

# Fallback chain (tried in order when primary fails or returns insufficient results)
SEARCH_FALLBACK_CHAIN=wikipedia,arxiv
```

#### Available Providers

| Provider | API Key Required | Free Tier | Best For |
|----------|------------------|-----------|----------|
| `bing` | No | Unlimited | General search (default) |
| `brave` | `BRAVE_API_KEY` | $5/month | Quality results, privacy |
| `serpapi` | `SERPAPI_API_KEY` | 100/month | Google results |
| `tavily` | `TAVILY_API_KEY` | 1000/month | AI-optimized search |
| `exa` | `EXA_API_KEY` | $5 credits | Semantic/embeddings search |
| `wikipedia` | No | Unlimited | Encyclopedia content |
| `arxiv` | No | Unlimited | Research papers |

#### Example Configurations

```bash
# Free setup (default)
SEARCH_PROVIDER=bing
SEARCH_FALLBACK_CHAIN=wikipedia,arxiv

# Premium setup with multiple fallbacks
SEARCH_PROVIDER=brave
SEARCH_FALLBACK_CHAIN=tavily,serpapi,bing,wikipedia,arxiv
BRAVE_API_KEY=BSA...
TAVILY_API_KEY=tvly-...
SERPAPI_API_KEY=...

# Research-focused (academic papers priority)
SEARCH_PROVIDER=arxiv
SEARCH_FALLBACK_CHAIN=wikipedia,bing

# AI-optimized search
SEARCH_PROVIDER=tavily
SEARCH_FALLBACK_CHAIN=exa,brave,bing,wikipedia
TAVILY_API_KEY=tvly-...
EXA_API_KEY=...
```

The system tries providers in order until it has enough results. If `brave` returns 3 results but you need 5, it continues to `tavily`, then `serpapi`, etc.

---

## Usage

### Basic Usage

```bash
python -m research_ai.cli "Your research topic here"
```

The system will:

1. Load configuration from `.env`
2. Initialize database and connectors
3. Create root agent for your topic
4. Recursively spawn and process agents
5. Generate final report in `research_report.md`

### Example Topics

```bash
# History
python -m research_ai.cli "What caused the fall of the Roman Empire?"

# Technology comparison
python -m research_ai.cli "Compare cloud providers AWS, Azure, and GCP for startups"

# Scientific research
python -m research_ai.cli "Recent advances in quantum computing error correction"

# Product research
python -m research_ai.cli "Best noise-cancelling headphones under $300 in 2024"

# Philosophy
python -m research_ai.cli "Effective altruism philosophical arguments"
```

### Monitoring Progress

When `PROGRESS_LOG=1` (default), you'll see:

```bash
[config] MAX_CONCURRENT_TASKS=1 (controls system-wide parallelism)
[config] connector=web_search
[progress] job=job-abc123 started depth=5 children=12 concurrency=1 max_concurrent_tasks=1

[worker:0] start task=task-xyz depth=1 topic=Machine learning applications
[openrouter] request model=openai/gpt-4o-mini base=https://openrouter.ai/api
[openrouter] response ok model=openai/gpt-4o-mini base=https://openrouter.ai/api tokens=245
[queue] +task=task-abc depth=2 topic=Neural networks in medical imaging
[worker:0] done task=task-xyz docs=50 spawned=12

[progress] [██████████░░░░░░░░░░]  50% tasks: 23/46 (+1 active) llm: 47 calls, 12.5K tokens ETA: 5m 30s
```

**Key metrics**:

- **Progress bar**: Visual indicator with percentage complete
- **ETA**: Estimated time remaining based on current progress
- **pending**: Tasks waiting to be processed
- **in_progress**: Currently running tasks (≤ MAX_CONCURRENT_TASKS)
- **done**: Completed tasks
- **llm_calls**: Total API calls made
- **llm_tokens**: Total tokens used (cost indicator)

**Web Search Stats** (logged every 5 failures):

```bash
[web_stats] searches=100 bing=85% fallback=10% failed=5% | fetches=500 ok=92%
```

- **bing%**: Searches that succeeded from Bing directly
- **fallback%**: Searches that needed Wikipedia/arXiv fallback
- **failed%**: Searches with no sources found
- **fetches ok%**: Individual URL content fetch success rate

### Output Format

The generated `research_report.md` contains:

1. **Executive Summary** (~200 words)
   - LLM-synthesized overview of all findings

2. **Table of Contents**
   - Links to all researched topics

3. **Detailed Sections**
   - Each topic gets a section
   - Multiple sources per topic (title, URL)
   - Summarized content for each source

4. **Appendix: Provenance**
   - Full metadata (topics, depth, sources)

---

## Understanding Sequential Execution

### The Problem: Uncontrolled Parallelism

Traditional parallel systems scale resources with tree depth:

```
Without limits:
├── Agent 1 (+ HTTP + LLM × 50)
├── Agent 2 (+ HTTP + LLM × 50)  } All running
├── Agent 3 (+ HTTP + LLM × 50)  } simultaneously
└── ...potentially hundreds...

Result: Memory × agents = CRASH with deep trees
```

### The Solution: Global Concurrency Gate

ResearchAI uses `MAX_CONCURRENT_TASKS` to limit operations:

```
With MAX_CONCURRENT_TASKS=1:
Agent 1 [Fetch][LLM×50]
                       Agent 2 [Fetch][LLM×50]
                                              Agent 3...

Result: Memory constant (~500MB) regardless of depth
```

### How It Works

1. **ExecutorLimiter** (`executor_limiter.py`)
   - Singleton managing global thread pool
   - ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
   - asyncio.Semaphore(MAX_CONCURRENT_TASKS)

2. **All I/O operations** go through this gate:
   - Connector.fetch() → acquires semaphore → executes → releases
   - LLM.summarize_async() → acquires semaphore → executes → releases

3. **Sequential execution** (MAX_CONCURRENT_TASKS=1):
   - Only 1 semaphore slot available
   - Operations queue and wait for the slot
   - No parallel execution possible

### Sequential vs Parallel

| MAX_CONCURRENT_TASKS | Behavior         | Memory | Speed | Use Case            |
| -------------------- | ---------------- | ------ | ----- | ------------------- |
| 1                    | Fully sequential | 500MB  | 1×    | <1GB RAM, stability |
| 2                    | Limited parallel | 1GB    | 1.8×  | <2GB RAM, balanced  |
| 4                    | Limited parallel | 2GB    | 3.5×  | <4GB RAM, faster    |
| 8                    | High parallel    | 4GB    | 6×    | <8GB RAM, fastest   |

### Memory Scaling

**Key insight**: Memory is constant for a given MAX_CONCURRENT_TASKS, regardless of depth.

```
depth=2, children=2, MAX_CONCURRENT_TASKS=1
→ ~7 agents, 5 minutes, 500MB

depth=100, children=1000, MAX_CONCURRENT_TASKS=1
→ ~10^300 agents, years, 500MB (still constant!)
```

Only **time** increases with depth, not resources.

### Scaling Characteristics

| Agents | MAX_CONCURRENT_TASKS=1 | =2      | =4      | =8      |
| ------ | ---------------------- | ------- | ------- | ------- |
| 10     | 8 min                  | 4 min   | 2 min   | 1 min   |
| 40     | 30 min                 | 15 min  | 8 min   | 4 min   |
| 100    | 75 min                 | 38 min  | 20 min  | 10 min  |
| 1000   | 12.5 hrs               | 6.3 hrs | 3.1 hrs | 1.6 hrs |

_Assumes 45 seconds per agent average_

---

## Project Structure

```
ResearchAI/
├── .env                          # Configuration (YOU EDIT THIS)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── research_ai/                  # Core package
│   ├── __init__.py              # Package marker
│   ├── cli.py                   # Entry point, loads .env
│   ├── orchestrator.py          # Job manager, worker pool
│   ├── agent.py                 # Agent logic, subtopic derivation
│   ├── llm.py                   # LLM wrapper (OpenRouter/OpenAI)
│   ├── datastore.py             # SQLite persistence
│   ├── report.py                # Markdown report generator
│   ├── embeddings.py            # Optional vector embeddings
│   ├── executor_limiter.py      # Global concurrency control ★
│   ├── logger.py                # Colored console logging utility ★
│   ├── runner.py                # Standalone single-job runner
│   └── web_search.py            # Web search connector
│
├── runtime/                      # Runtime data (auto-created)
│   ├── research_ai/
│   │   ├── state.db             # SQLite database (auto-created)
│   │   └── artifacts/           # Cached source documents (SHA-256 filenames)
│   │       └── *.txt
│   └── artifacts/               # Additional artifact storage
│
├── Example_output/               # Example generated reports
│   └── *.md                     # Sample research reports
│
└── venv/                         # Python virtual environment
```

### What Each Folder Contains

**`research_ai/`** - Core application code

- Main modules for orchestration, agents, LLM, storage
- **executor_limiter.py**: Controls sequential execution
- **logger.py**: Colored console logging with category-based formatting
- **runner.py**: Standalone single-job runner for quick testing
- **web_search.py**: Web search connector for fetching sources

**`runtime/`** - Runtime data and caches

- **state.db**: SQLite database with jobs, tasks, agents, results
- **artifacts/**: Cached raw source documents (named by SHA-256 hash)
- Auto-created on first run
- Safe to delete (will regenerate, but loses history)

**`Example_output/`** - Example research reports

- Sample generated reports for reference

**`venv/`** - Python virtual environment

- Isolated Python packages
- Created with `python -m venv venv`
- Activate before running

<details>
<summary>📁 Project Organization (click to expand)</summary>

### Files You Should Edit

| File               | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| **`.env`**         | Your main configuration — API key, research parameters, resource limits |
| **Topic argument** | When running: `python -m research_ai.cli "Your topic here"`             |

### Files You Might Edit (Advanced)

| File               | Purpose                              |
| ------------------ | ------------------------------------ |
| `research_ai/*.py` | If extending the system              |
| `.gitignore`       | If you want to track different files |

### Files/Folders You Shouldn't Touch

| Folder            | Purpose                 | Can Delete?              |
| ----------------- | ----------------------- | ------------------------ |
| `runtime/`        | Runtime data and caches | ✅ Yes (will regenerate) |
| `Example_output/` | Example reports         | ✅ Yes                   |
| `venv/`           | Managed by pip          | ✅ Yes (must recreate)   |
| `__pycache__/`    | Python bytecode cache   | ✅ Yes                   |

### What is `state.db`?

SQLite database storing jobs, tasks, agents, and results. Located in `runtime/research_ai/`. Created automatically on first run. Safe to delete (will recreate, but loses history).

### Cleaning Up

```bash
# Clear all runtime data (reports, cache, database)
rm -rf runtime/

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear virtual environment (must reinstall after)
rm -rf venv/
```

### Do NOT Delete

- `.env` — Your configuration and API key
- `research_ai/` — Core application code
- `requirements.txt` — Dependency list
- `README.md` — Documentation

### Quick Reference

| Folder/File        | Purpose             | Edit?    | Delete? |
| ------------------ | ------------------- | -------- | ------- |
| `research_ai/`     | Application code    | Advanced | No      |
| `runtime/`         | Runtime data/caches | No       | Yes     |
| `Example_output/`  | Example reports     | No       | Yes     |
| `venv/`            | Virtual environment | No       | Yes\*   |
| `.env`             | Configuration       | **Yes**  | No      |
| `requirements.txt` | Dependencies        | Advanced | No      |

\*Can delete venv/ but must recreate and reinstall packages

</details>

---

## Performance Tuning

### For Limited RAM (4-8GB)

```bash
MAX_CONCURRENT_TASKS=1
MAX_DEPTH=2
MAX_CHILDREN=3
SOURCE_COUNT=10
CONCURRENCY=1
```

**Result**: ~13 agents, ~10 minutes, ~400MB RAM

### For Faster Results (16GB+)

```bash
MAX_CONCURRENT_TASKS=4
MAX_DEPTH=3
MAX_CHILDREN=5
SOURCE_COUNT=20
CONCURRENCY=2
```

**Result**: ~156 agents, ~30 minutes, ~2GB RAM

### For Comprehensive Research (32GB+)

```bash
MAX_CONCURRENT_TASKS=8
MAX_DEPTH=4
MAX_CHILDREN=8
SOURCE_COUNT=50
CONCURRENCY=4
```

**Result**: ~4,000+ agents, ~2 hours, ~4GB RAM

### Agent Count Formula

Total agents ≈ `SUM(MAX_CHILDREN^depth for depth in 0..MAX_DEPTH)`

Examples:

- depth=2, children=2: 1 + 2 + 4 = **7 agents**
- depth=3, children=3: 1 + 3 + 9 + 27 = **40 agents**
- depth=3, children=5: 1 + 5 + 25 + 125 = **156 agents**
- depth=5, children=12: 1 + 12 + 144 + ... = **~248,832 agents**

### Estimating Runtime

```
Time = (Total Agents / MAX_CONCURRENT_TASKS) × Average Time Per Agent
Average Time Per Agent ≈ 30-60 seconds (fetch + LLM calls)
```

Examples:

- 7 agents, MAX_CONCURRENT_TASKS=1: 7 × 45s = **~5 minutes**
- 156 agents, MAX_CONCURRENT_TASKS=1: 156 × 45s = **~2 hours**
- 156 agents, MAX_CONCURRENT_TASKS=4: 156/4 × 45s = **~30 minutes**

---

## Troubleshooting

### No Sources Retrieved

**Symptoms**: Report says "No online sources were retrieved for this topic"

**Solutions**:

1. Check OpenRouter API key is valid in `.env`
2. Verify network connectivity: `ping openrouter.ai`
3. Check OpenRouter privacy settings: https://openrouter.ai/settings/privacy
   - Ensure data policy allows external requests
4. Check console logs for specific error messages

### Blank or Very Short Reports

**Symptoms**: Report has few sections or mostly empty content

**Solutions**:

1. Increase `SOURCE_COUNT` to 50 or more
2. Check console for errors during source fetching
3. Verify LLM is working: `llm_calls > 0` in progress logs
4. Increase `MAX_DEPTH` to get more coverage

### Out of Memory Errors

**Symptoms**: Process crashes, system becomes unresponsive

**Solutions**:

1. Ensure `MAX_CONCURRENT_TASKS=1` in `.env`
2. Reduce `MAX_DEPTH` to 2 or 3
3. Reduce `MAX_CHILDREN` to 2 or 3
4. Lower `SOURCE_COUNT` to 10-20
5. Close other applications to free RAM
6. Consider upgrading RAM if you need deeper research

### API Rate Limit Errors

**Symptoms**: "Rate limit exceeded" or HTTP 429 errors, or "All providers rate limited" warnings

**Automatic Handling**: The system has coordinated rate limit throttling:
- When one request hits a rate limit, ALL concurrent requests pause together
- This prevents "thundering herd" where multiple requests independently discover limits
- Automatic retry with exponential backoff (up to 60 seconds) before failing

**Solutions** (if automatic retry fails):

1. Set `MAX_CONCURRENT_TASKS=1` (slower but stays under limits)
2. Use free models with higher limits: `arcee-ai/trinity-large-preview:free`
3. Check OpenRouter dashboard for your account limits
4. Configure multiple API keys to increase effective rate limits
5. Upgrade to paid OpenRouter tier for higher limits

### Very Slow Progress

**Symptoms**: Hours passing with minimal completed agents

**Solutions**:

1. Check `MAX_DEPTH` isn't too high (each level multiplies agents exponentially)
2. Monitor `pending` count in logs (should decrease over time)
3. Verify network isn't dropping connections: `ping openrouter.ai`
4. Consider increasing `MAX_CONCURRENT_TASKS` if you have RAM
5. Reduce `SOURCE_COUNT` to speed up each agent
6. Use faster model: `openai/gpt-4o-mini` instead of larger models

### Repeated Content in Report

**Symptoms**: Same information appears multiple times

**Solutions**:

1. System has automatic deduplication, but some repetition is expected
2. Agents exploring similar topics may find overlapping sources
3. This is normal for broad topics
4. Consider reducing `MAX_CHILDREN` to diversify topics more
5. Use more specific research topics

### in_progress > MAX_CONCURRENT_TASKS

**Symptoms**: Progress logs show more tasks running than configured

**Diagnosis**: Bug in concurrency control implementation

**Solution**: File an issue with full logs and configuration

### Progress Stuck at 0

**Symptoms**: No tasks being processed, `in_progress` always 0

**Solutions**:

1. Check console for error messages
2. Check OpenRouter API key is correct
3. Ensure `.env` file is being loaded (check startup logs)

---

## Architecture

### High-Level Flow

```
CLI → Load .env → Initialize ExecutorLimiter
                         ↓
              ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
              asyncio.Semaphore(MAX_CONCURRENT_TASKS)
                         ↓
              Orchestrator (spawn workers)
                         ↓
              Worker(s) claim tasks from database
                         ↓
                    Agent.run()
                    /         \
                   /           \
          connector.fetch()  llm.summarize_async() × N
              ↓                    ↓
          Semaphore            Semaphore
          (acquire)            (acquire)
              ↓                    ↓
          Executor             Executor
          (limited)            (limited)
              ↓                    ↓
          HTTP Request         OpenRouter API
              ↓                    ↓
          Release              Release
          Semaphore            Semaphore
```

### Component Responsibilities

**ExecutorLimiter** (`executor_limiter.py`)

- Singleton pattern
- Creates ThreadPoolExecutor with `max_workers=MAX_CONCURRENT_TASKS`
- Creates asyncio.Semaphore with same limit
- Provides `get_executor()`, `get_semaphore()`, `get_max_tasks()`

**Orchestrator** (`orchestrator.py`)

- Spawns `CONCURRENCY` worker coroutines
- Workers claim tasks from datastore
- Workers run agents (blocked by semaphore)
- Tracks progress, handles budgets

**Agent** (`agent.py`)

- Fetches sources via connector (blocked by semaphore)
- Summarizes each source via LLM (blocked by semaphore)
- Derives subtopics from summaries
- Spawns child agents via orchestrator.enqueue()

**LLM** (`llm.py`)

- Wraps OpenRouter/OpenAI API calls
- `summarize_async()`: Async with concurrency control
- `summarize_to_200_words_async()`: For report exec summary
- Tracks usage: calls, tokens

**Web Search** (`web_search.py`)

- Multi-provider search with configurable fallback chain
- Supports: Brave, SerpAPI, Tavily, Exa, Bing, Wikipedia, arXiv
- Uses global executor and semaphore for concurrency control

**Datastore** (`datastore.py`)

- SQLite persistence layer
- Tables: jobs, tasks, agents, artifacts, embeddings
- `claim_next_task()`: Race-free task claiming (BEGIN IMMEDIATE)

**ReportGenerator** (`report.py`)

- Aggregates agent results
- Generates executive summary
- Writes streaming Markdown report
- Deduplicates topics and summaries

### Sequential Execution Timeline

```
Time (seconds) →
0──────────10─────────20─────────30─────────40─────────50

Agent 1    [Fetch][LLM][LLM][LLM]...[LLM]
                                           Agent 2 [Fetch][LLM]...
                                                                   Agent 3...

Note: Only ONE operation (fetch or LLM) active at any moment.
      Agent 2 waits for Agent 1 to complete all its operations.
```

### Parallel Execution Timeline

```
Time (seconds) →
0──────────10─────────20

Agent 1    [Fetch][LLM][LLM][LLM]...[LLM]
Agent 2    [Fetch][LLM][LLM][LLM]...[LLM]
Agent 3    [Fetch][LLM][LLM][LLM]...[LLM]
Agent 4    [Fetch][LLM][LLM][LLM]...[LLM]
                                         Agent 5...

Note: Up to MAX_CONCURRENT_TASKS operations can happen simultaneously.
      Agents 5+ wait for a slot to open up.
```

### Memory Usage Comparison

```
Traditional (No Limits):
RAM = N_concurrent_agents × RAM_per_agent × (1 + N_sources)
Example: 100 agents × 10MB × 51 = 51GB → CRASH!

Sequential (MAX_CONCURRENT_TASKS=1):
RAM = 1 × 10MB × 51 = 510MB → Constant!

Parallel (MAX_CONCURRENT_TASKS=4):
RAM = 4 × 10MB × 51 = 2GB → Manageable!
```

### Key Insights

1. **Semaphore gates ALL I/O operations** (fetch, LLM calls)
2. **ThreadPoolExecutor** prevents blocking event loop
3. **max_workers=1** → sequential via thread serialization
4. **Semaphore(1)** → only 1 async op passes at a time
5. **Together**: complete serialization of all operations
6. **Memory stays constant** regardless of tree depth
7. **Only time scales** with agent count
8. **Can safely set** depth=100, children=1000 without crash

---

## Advanced Topics

### Budget Controls

Add to `.env` to limit costs:

```bash
MAX_TOKENS=1000000        # Stop after N tokens
MAX_CALLS=500             # Stop after N API calls
MAX_TIME=3600             # Stop after N seconds
MAX_COST=10.00            # Stop after $N USD (if supported)
```

When a budget is exceeded, the job stops gracefully and generates a report with completed agents.

### Custom Connectors

Create your own connector by implementing:

```python
class CustomConnector:
    async def fetch(self, topic: str, n: int = 6):
        """Fetch n documents for topic.
        Returns: [{"title": str, "url": str, "text": str}, ...]
        """
        from research_ai.executor_limiter import get_executor, get_semaphore
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self._sync_fetch, topic, n)

    def _sync_fetch(self, topic: str, n: int):
        # Your sync implementation here
        return [{"title": "...", "url": "...", "text": "..."}]
```

### Embeddings (Optional)

Enable vector embeddings for local retrieval:

```python
from research_ai.embeddings import Embeddings

embeddings = Embeddings()
orch = Orchestrator(..., embeddings=embeddings)
```

Embeddings are stored in the database for similarity search.

### Database Schema

```sql
-- Jobs
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    topic TEXT,
    config TEXT,  -- JSON
    status TEXT,
    created_at TEXT,
    updated_at TEXT
);

-- Tasks (pending work items)
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    parent_id TEXT,
    topic TEXT,
    depth INTEGER,
    max_depth INTEGER,
    max_children INTEGER,
    status TEXT,  -- pending, in_progress, done
    created_at TEXT
);

-- Agents (completed results)
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    job_id TEXT,
    parent_id TEXT,
    topic TEXT,
    depth INTEGER,
    result TEXT,  -- JSON
    created_at TEXT
);

-- Raw artifacts (cached sources)
CREATE TABLE artifacts (
    path TEXT PRIMARY KEY,
    content TEXT,
    created_at TEXT
);

-- Embeddings (optional)
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    doc_id TEXT,
    embedding BLOB,
    metadata TEXT,  -- JSON
    created_at TEXT
);

-- Usage tracking
CREATE TABLE job_usage (
    job_id TEXT PRIMARY KEY,
    usage TEXT,  -- JSON with calls, tokens, etc.
    created_at TEXT,
    updated_at TEXT
);
```

### Extending the System

**Add new LLM provider**:
Edit `llm.py` to add provider detection and API call logic.

**Add new report format**:
Create new generator class similar to `ReportGenerator` in `report.py`.

**Add new storage backend**:
Implement interface from `datastore.py` with your storage system.

**Add custom agent logic**:
Subclass `Agent` in `agent.py` and override `run()` or `_derive_subtopics()`.

---

## Tips and Best Practices

1. **Start small**: Always test with depth=2, children=2 first
2. **Monitor logs**: Watch pending/in_progress/done counts
3. **Be patient**: Deep research takes time in sequential mode
4. **Use free models**: Test with free models before using paid ones
5. **Sequential is safest**: Default MAX_CONCURRENT_TASKS=1 prevents crashes
6. **Read the output**: Check research_report.md quality before scaling up
7. **Iterate**: Adjust depth/children based on initial results
8. **Save .env**: Back up configuration before experimenting
9. **Budget wisely**: Set MAX_TOKENS or MAX_CALLS to prevent runaway costs
10. **Review topics**: Ensure your research topic is specific enough

---

## Support and Resources

- **OpenRouter Documentation**: https://openrouter.ai/docs
- **OpenRouter Models**: https://openrouter.ai/models
- **OpenRouter API Keys**: https://openrouter.ai/keys
- **Account Settings**: https://openrouter.ai/settings/privacy
- **GitHub Issues**: (your repo URL here)

---

## Acknowledgments

Built with:

- OpenRouter API for LLM access
- SQLite for persistence
- asyncio for concurrency control
- requests for HTTP
- BeautifulSoup for HTML parsing

---

**Ready to start?** Edit `.env` with your API key and run:

```bash
python -m research_ai.cli "Your fascinating research topic"
```

Happy researching! 🚀
