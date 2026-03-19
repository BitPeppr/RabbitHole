import asyncio
import os
import uuid

from .agent import Agent
from .report import ReportGenerator
from .logger import log


class Orchestrator:
    def __init__(self, datastore, llm, connector, concurrency: int = 2, embeddings=None):
        self.datastore = datastore
        self.llm = llm
        self.connector = connector
        self.concurrency = concurrency
        self.embeddings = embeddings
        self.workers = []
        self.job_id = None
        self._stop = False
        self._max_depth = 2
        self._max_children = 2
        self.progress_enabled = os.environ.get("PROGRESS_LOG", "1").lower() not in ("0", "false", "no")
        self.progress_verbose_tasks = os.environ.get("PROGRESS_VERBOSE_TASKS", "1").lower() not in ("0", "false", "no")
        try:
            self.progress_interval_sec = max(1.0, float(os.environ.get("PROGRESS_INTERVAL_SEC", "5")))
        except Exception:
            self.progress_interval_sec = 5.0

    def _estimate_total_tasks(self, max_depth: int, max_children: int) -> int:
        """Estimate total number of tasks based on tree structure.
        
        For a tree with max_depth=d and max_children=c:
        Total nodes = 1 + c + c^2 + ... + c^d = (c^(d+1) - 1) / (c - 1) for c > 1
        """
        if max_children <= 1:
            return max_depth + 1
        # Geometric series sum
        total = 0
        for d in range(max_depth + 1):
            total += max_children ** d
        return total

    async def run_job(self, topic: str, max_depth: int = 2, max_children: int = 2, output_path: str = "research_report.md", job_id: str = None):
        # Store for progress estimation
        self._max_depth = max_depth
        self._max_children = max_children
        # ensure datastore is initialized
        if not getattr(self.datastore, "conn", None):
            try:
                self.datastore.init()
            except Exception:
                pass
        if job_id is None:
            job_id = f"job-{uuid.uuid4().hex[:8]}"
            self.job_id = job_id
            config = {"max_depth": max_depth, "max_children": max_children}
            self.datastore.create_job(job_id, topic, config)
            # create root task
            root_id = f"task-{uuid.uuid4().hex}"
            # store job-level context with the root task so children inherit the original prompt
            self.datastore.add_task(task_id=root_id, job_id=job_id, parent_id=None, topic=topic, depth=0, max_depth=max_depth, max_children=max_children, status="pending", context=topic)
        else:
            self.job_id = job_id
        # start workers
        self._stop = False
        if self.progress_enabled:
            from .executor_limiter import get_max_tasks
            max_tasks = get_max_tasks()
            log.job_started(self.job_id, max_depth, max_children, self.concurrency, max_tasks)
        for i in range(self.concurrency):
            w = asyncio.create_task(self._worker(i))
            self.workers.append(w)
        heartbeat = asyncio.create_task(self._heartbeat())
        # wait until all tasks done or stopped by budget
        try:
            while True:
                remaining = self.datastore.count_remaining_tasks(self.job_id)
                job = self.datastore.get_job(self.job_id)
                if job and job.get("status") == "stopped":
                    log.progress(f"job={self.job_id} marked stopped; exiting")
                    break
                if remaining == 0:
                    break
                await asyncio.sleep(0.2)
        finally:
            self._stop = True
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            for w in self.workers:
                w.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers = []
        # Finalize TUI mode (print newline after progress bar)
        log.finalize_tui()
        # collect results
        results = self.datastore.get_all_agent_results(job_id=self.job_id)
        report = ReportGenerator(llm=self.llm, results=results, root_topic=topic)
        out_path = await report.generate(output_path)
        if self.progress_enabled:
            log.job_completed(self.job_id, len(results), out_path)
        return out_path

    async def _worker(self, worker_id: int):
        while not self._stop:
            task = self.datastore.claim_next_task(self.job_id)
            if not task:
                await asyncio.sleep(0.1)
                continue
            if self.progress_enabled and self.progress_verbose_tasks:
                log.worker(worker_id, f"start depth={task['depth']}", task['id'], task["topic"])
            # build an Agent from task
            agent = Agent(topic=task["topic"], depth=task["depth"], max_depth=task["max_depth"], max_children=task["max_children"], parent_id=task["parent_id"], parent_context=task.get("context"))
            agent.id = task["id"]  # preserve task id for persistence
            try:
                result = await agent.run(self)
                try:
                    self.datastore.update_task_result(task["id"], self.job_id, result)
                except Exception as e:
                    log.error(f"Error saving agent result: {e}")
                if self.progress_enabled and self.progress_verbose_tasks:
                    log.worker_done(worker_id, task['id'], result.get('doc_count', 0), result.get('spawned_children', 0))
                # record LLM usage and enforce simple budgets
                try:
                    usage = self.llm.get_usage() if hasattr(self.llm, "get_usage") else None
                    if usage:
                        self.datastore.record_job_usage(self.job_id, usage)
                        job = self.datastore.get_job(self.job_id)
                        cfg = job.get("config") if job else {}
                        max_tokens = cfg.get("max_tokens") or cfg.get("token_budget")
                        max_calls = cfg.get("max_calls")
                        if max_tokens and usage.get("total_tokens", 0) > max_tokens:
                            log.warning(f"Token budget exceeded for job {self.job_id}; stopping job.")
                            self.datastore.update_job_status(self.job_id, "stopped")
                            self._stop = True
                            break
                        if max_calls and usage.get("calls", 0) >= max_calls:
                            log.warning(f"Call budget exceeded for job {self.job_id}; stopping job.")
                            self.datastore.update_job_status(self.job_id, "stopped")
                            self._stop = True
                            break
                except Exception:
                    pass
            except Exception as e:
                log.error(f"Worker {worker_id} error running agent {getattr(agent, 'topic', None)}: {e}")

    async def enqueue(self, agent: Agent):
        # persist a new task for this job
        if not getattr(self, "job_id", None):
            raise RuntimeError("No active job id for enqueue")
        task_id = f"task-{uuid.uuid4().hex}"
        # inherit context from agent if available, otherwise use job root topic
        context = getattr(agent, "parent_context", None) or (self.datastore.get_job(self.job_id) or {}).get("topic")
        self.datastore.add_task(task_id=task_id, job_id=self.job_id, parent_id=agent.parent_id, topic=agent.topic, depth=agent.depth, max_depth=agent.max_depth, max_children=agent.max_children, status="pending", context=context)
        if self.progress_enabled and self.progress_verbose_tasks:
            log.queue("+task", task_id, depth=agent.depth, topic=agent.topic)
        return task_id

    async def _heartbeat(self):
        while not self._stop:
            if self.progress_enabled and self.job_id:
                try:
                    counts = self.datastore.get_task_counts(self.job_id)
                    usage = self.llm.get_usage() if hasattr(self.llm, "get_usage") else {}
                    calls = usage.get("calls", 0) if isinstance(usage, dict) else 0
                    tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                    estimated = self._estimate_total_tasks(self._max_depth, self._max_children)
                    log.progress_stats(
                        counts.get('pending', 0),
                        counts.get('in_progress', 0),
                        counts.get('done', 0),
                        calls,
                        tokens,
                        estimated_total=estimated
                    )
                except Exception as e:
                    log.error(f"heartbeat error: {e}")
            await asyncio.sleep(self.progress_interval_sec)
