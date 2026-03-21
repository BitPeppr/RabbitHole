import sqlite3
import json
import os
import hashlib
import uuid
from typing import Optional, Dict, Any


class Datastore:
    """Persistent SQLite-backed datastore with jobs and tasks."""

    def __init__(self, db_path: str):
        # prefer a runtime directory for all transient files
        runtime_dir = os.environ.get('RUNTIME_DIR', 'runtime')
        if os.path.isabs(db_path):
            self.db_path = db_path
        else:
            # place relative db paths inside runtime_dir
            os.makedirs(runtime_dir, exist_ok=True)
            self.db_path = os.path.join(runtime_dir, db_path)
        self.conn = None
        base_dir = os.path.dirname(self.db_path) or runtime_dir
        self.artifacts_dir = os.path.join(base_dir, "artifacts")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def init(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            parent_id TEXT,
            topic TEXT,
            depth INTEGER,
            result_json TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            path TEXT,
            type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            topic TEXT,
            config_json TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            parent_id TEXT,
            topic TEXT,
            depth INTEGER,
            max_depth INTEGER,
            max_children INTEGER,
            status TEXT,
            result_json TEXT,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS job_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            usage_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            doc_id TEXT,
            vector_json TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        # prompt_logs: store LLM prompts and responses for provenance and debugging
        cur.execute("""CREATE TABLE IF NOT EXISTS prompt_logs (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            task_id TEXT,
            role TEXT,
            prompt_text TEXT,
            response_text TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""")
        self.conn.commit()
        # Migration: ensure 'job_id' exists on agents table for older DBs
        try:
            cur.execute("PRAGMA table_info(agents)")
            cols = [r[1] for r in cur.fetchall()]
            if 'job_id' not in cols:
                try:
                    cur.execute("ALTER TABLE agents ADD COLUMN job_id TEXT")
                except Exception:
                    pass
        except Exception:
            pass
        # Migration: ensure 'context' column exists on tasks table
        try:
            cur.execute("PRAGMA table_info(tasks)")
            tcols = [r[1] for r in cur.fetchall()]
            if 'context' not in tcols:
                try:
                    cur.execute("ALTER TABLE tasks ADD COLUMN context TEXT")
                except Exception:
                    pass
        except Exception:
            pass
        self.conn.commit()

    def create_job(self, job_id: str, topic: str, config: dict):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO jobs (id, topic, config_json, status, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (job_id, topic, json.dumps(config), "running"),
        )
        self.conn.commit()

    def get_job(self, job_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT id, topic, config_json, status, created_at FROM jobs WHERE id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "topic": row[1], "config": json.loads(row[2]) if row[2] else {}, "status": row[3], "created_at": row[4]}

    def update_job_status(self, job_id: str, status: str):
        cur = self.conn.cursor()
        cur.execute("UPDATE jobs SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (status, job_id))
        self.conn.commit()

    def record_job_usage(self, job_id: str, usage: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO job_usage (job_id, usage_json) VALUES (?, ?)", (job_id, json.dumps(usage)))
        cur.execute("UPDATE jobs SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        self.conn.commit()

    def get_latest_job_usage(self, job_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT usage_json FROM job_usage WHERE job_id=? ORDER BY created_at DESC LIMIT 1", (job_id,))
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def add_task(self, task_id: str, job_id: str, parent_id: Optional[str], topic: str, depth: int, max_depth: int, max_children: int, status: str = "pending", context: Optional[str] = None):
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO tasks
            (id, job_id, parent_id, topic, depth, max_depth, max_children, status, context, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (task_id, job_id, parent_id, topic, depth, max_depth, max_children, status, context),
        )
        self.conn.commit()

    def claim_next_task(self, job_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                "SELECT id, parent_id, topic, depth, max_depth, max_children, context FROM tasks WHERE job_id=? AND status='pending' ORDER BY created_at LIMIT 1",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                self.conn.rollback()
                return None
            task_id, parent_id, topic, depth, max_depth, max_children, context = row
            cur.execute("UPDATE tasks SET status='in_progress', updated_at=CURRENT_TIMESTAMP WHERE id=?", (task_id,))
            self.conn.commit()
            return {
                "id": task_id,
                "parent_id": parent_id,
                "topic": topic,
                "depth": depth,
                "max_depth": max_depth,
                "max_children": max_children,
                "context": context,
            }
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return None

    def update_task_result(self, task_id: str, job_id: str, result: dict):
        cur = self.conn.cursor()
        cur.execute("UPDATE tasks SET result_json=?, status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(result), task_id))
        # also store in agents table for easy reading
        parent_id = result.get("parent_id") if isinstance(result, dict) else None
        topic = result.get("topic") if isinstance(result, dict) else None
        depth = result.get("depth") if isinstance(result, dict) else None
        cur.execute(
            """INSERT OR REPLACE INTO agents (id, job_id, parent_id, topic, depth, result_json, status, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (task_id, job_id, parent_id, topic, depth, json.dumps(result), "done"),
        )
        self.conn.commit()

    def get_all_agent_results(self, job_id: str = None):
        cur = self.conn.cursor()
        if job_id:
            cur.execute("SELECT result_json FROM agents WHERE job_id=? ORDER BY created_at", (job_id,))
        else:
            cur.execute("SELECT result_json FROM agents ORDER BY created_at")
        rows = cur.fetchall()
        results = []
        for (r,) in rows:
            try:
                results.append(json.loads(r))
            except Exception:
                pass
        return results

    def count_remaining_tasks(self, job_id: str) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(1) FROM tasks WHERE job_id=? AND status!='done'", (job_id,))
        row = cur.fetchone()
        return row[0] if row else 0

    def get_task_counts(self, job_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT status, COUNT(1) FROM tasks WHERE job_id=? GROUP BY status", (job_id,))
        rows = cur.fetchall()
        counts = {"pending": 0, "in_progress": 0, "done": 0}
        for status, n in rows:
            counts[status] = int(n)
        return counts

    def save_raw_artifact(self, content: str, filename: str = None, artifact_type: str = "text"):
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        path = filename or f"{h}.txt"
        full = os.path.join(self.artifacts_dir, path)
        if not os.path.exists(full):
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO artifacts (id, path, type) VALUES (?, ?, ?)", (h, full, artifact_type))
        self.conn.commit()
        return full

    def save_prompt_response(self, job_id: str, task_id: str, role: str, prompt_text: str, response_text: str, metadata: dict = None):
        cur = self.conn.cursor()
        pid = uuid.uuid4().hex
        cur.execute("INSERT INTO prompt_logs (id, job_id, task_id, role, prompt_text, response_text, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (pid, job_id, task_id, role, prompt_text, response_text, json.dumps(metadata or {})))
        self.conn.commit()
        return pid

    def save_embedding(self, emb_id: str, job_id: str, doc_id: str, vector: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO embeddings (id, job_id, doc_id, vector_json, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (emb_id, job_id, doc_id, json.dumps(vector), json.dumps(metadata or {})))
        self.conn.commit()

    def get_embeddings_for_job(self, job_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id, vector_json, metadata_json FROM embeddings WHERE job_id=?", (job_id,))
        rows = cur.fetchall()
        out = []
        for doc_id, vec_json, meta_json in rows:
            try:
                vec = json.loads(vec_json)
            except Exception:
                vec = {}
            try:
                meta = json.loads(meta_json)
            except Exception:
                meta = {}
            out.append((doc_id, vec, meta))
        return out

    def find_similar(self, job_id: str, query_vector: Dict[str, float], top_k: int = 5):
        # naive linear scan similarity
        import math
        def dot(a, b):
            return sum(a.get(k, 0) * b.get(k, 0) for k in a)
        def norm(a):
            return math.sqrt(sum(v * v for v in a.values()))
        qn = norm(query_vector)
        rows = self.get_embeddings_for_job(job_id)
        scored = []
        for doc_id, vec, meta in rows:
            vn = norm(vec)
            denom = (qn * vn)
            score = dot(query_vector, vec) / denom if denom else 0.0
            scored.append((score, doc_id, meta))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:top_k]
