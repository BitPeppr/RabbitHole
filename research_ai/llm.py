"""Simple LLM wrapper with a lightweight local fallback summarizer.

Supports OpenAI and OpenRouter providers. Auto-detects via environment variables or accepts an explicit provider parameter.

Uses global executor limiter to ensure strict sequential execution when MAX_CONCURRENT_TASKS=1.
"""

import os
import re
import json
import asyncio
import time
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .logger import log

STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "that", "this", "for", "on", "with",
    "as", "are", "was", "were", "by", "an", "be", "or", "from", "at"
}

# Retry configuration
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY_SEC = 5
LLM_BACKUP_DELAY_SEC = 60


class LLM:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider
        self.use_openai = False
        self.use_openrouter = False
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Backup model from environment
        self.backup_model = os.environ.get("OPENROUTER_BACKUP_MODEL", None)
        # Prefer explicit provider. If omitted, auto-detect (prefer OpenRouter when available).
        try:
            if provider == "openrouter" or (provider is None and os.environ.get("OPENROUTER_API_KEY")):
                key = os.environ.get("OPENROUTER_API_KEY")
                if key:
                    self.openrouter_key = key
                    self.openrouter_base = os.environ.get("OPENROUTER_API_BASE", "https://api.openrouter.ai")
                    self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
                    self.openrouter_log = os.environ.get("OPENROUTER_LOG", "1").lower() not in ("0", "false", "no")
                    self.use_openrouter = True
        except Exception:
            self.use_openrouter = False
        try:
            if provider == "openai" or (provider is None and os.environ.get("OPENAI_API_KEY") and not self.use_openrouter):
                key = os.environ.get("OPENAI_API_KEY")
                if key:
                    import openai

                    self.openai = openai
                    self.openai.api_key = key
                    self.use_openai = True
        except Exception:
            self.use_openai = False

    def _record_usage(self, usage):
        if not usage:
            return
        try:
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
            self.usage["total_tokens"] += int(usage.get("total_tokens", 0))
            self.usage["calls"] += 1
        except Exception:
            pass

    def get_usage(self):
        return dict(self.usage)

    def _propose_subtopics_via_openrouter(self, payload_messages, model=None, temperature=0.0):
        # reuse _call_openrouter to keep base handling
        try:
            return self._call_openrouter(payload_messages, model=model, temperature=temperature)
        except Exception as e:
            raise

    def propose_subtopics(self, root_topic: str, current_topic: str, summaries: list, depth: int, max_depth: int, max_children: int):
        """Synchronous proposal of subtopics via LLM. Returns a list of topic strings or [] on failure.

        The LLM is asked to return a JSON array of subtopics (strings). Caller must validate.
        """
        # Build a compact payload containing root context and summaries
        prompt = f"You are a Research Agent helper. Root topic: {root_topic}\nCurrent topic: {current_topic}\nLayer depth: {depth} of {max_depth}\nMaximum subtopics to propose: {max_children}\n\n" + \
                 f"Given the following source summaries (title — summary), propose up to {max_children} concrete child research subtopics that stay strictly on topic. STAY ON TOPIC and prefer phrasing that includes the same brand/feature tokens as the root topic (e.g., 'Nikon Z').\n\n"
        # append summaries (truncate each to a safe length)
        for s in summaries:
            title = s.get('title','')
            summ = (s.get('summary') or '')[:2000]
            prompt += f"\n- {title} — {summ}"
        prompt += "\n\nReturn a JSON array of strings, e.g. [\"subtopic 1\", \"subtopic 2\"] and nothing else."
        messages = [{"role": "user", "content": prompt}]
        try:
            content, usage = self._propose_subtopics_via_openrouter(messages, model=getattr(self, 'openrouter_model', None), temperature=0.0)
            # log prompt/response for provenance when datastore available
            try:
                from .datastore import Datastore
                db_path = os.environ.get('DATASTORE_PATH') or 'test_validator.db'
                ds = Datastore(db_path)
                ds.init()
                ds.save_prompt_response(job_id=os.environ.get('CURRENT_JOB_ID',''), task_id=os.environ.get('CURRENT_TASK_ID',''), role='propose_subtopics', prompt_text=json.dumps(messages), response_text=content or '', metadata={'model': getattr(self,'openrouter_model',None)})
            except Exception:
                pass
            if not content:
                return []
            # try to extract JSON
            import json, re
            txt = content.strip()
            # prefer stricter JSON schema: an array of objects {"topic":..., "reason":..., "confidence":0.0}
            m = re.search(r"(\[\s*\{.*?\}\s*\])", txt, flags=re.S)
            if m:
                try:
                    arr = json.loads(m.group(1))
                    out = []
                    for o in arr:
                        if isinstance(o, dict) and 'topic' in o:
                            out.append(str(o['topic']).strip())
                    if out:
                        return out[:max_children]
                except Exception:
                    pass
            # fallback: find a simple JSON array of strings
            m2 = re.search(r"(\[.*\])", txt, flags=re.S)
            if m2:
                try:
                    arr2 = json.loads(m2.group(1))
                    if isinstance(arr2, list):
                        return [str(x).strip() for x in arr2 if x][:max_children]
                except Exception:
                    pass
            # fallback: try to extract quoted lines or bullet list
            lines = [l.strip('- "').strip() for l in txt.splitlines() if l.strip() and len(l.strip()) < 200]
            if lines:
                # take up to max_children
                return [l for l in lines][:max_children]
        except Exception:
            pass
        return []

    def validate_candidate(self, candidate: str, root_topic: str, summaries: list) -> bool:
        """Ask the LLM (if available) to judge whether a candidate subtopic stays on-topic relative to root_topic.

        Returns True if LLM judges candidate on-topic, False otherwise. On failure to call LLM, returns False.
        """
        if not candidate or not root_topic:
            return False
        # build prompt asking for JSON boolean
        prompt = (
            f"You are an assistant that judges topical relevance. Root topic: {root_topic}\nCandidate subtopic: {candidate}\n\n"
            "Given the following short source summaries, answer with a JSON object {\"on_topic\": true|false, \"reason\": \"...\"} and nothing else. Be conservative: return true only if the candidate clearly stays within the brand/feature scope of the root topic.\n\n"
        )
        for s in summaries[:6]:
            prompt += f"- {s.get('title','')}: {(s.get('summary') or '')[:400]}\n"
        messages = [{"role": "user", "content": prompt}]
        try:
            content, usage = self._propose_subtopics_via_openrouter(messages, model=getattr(self, 'openrouter_model', None), temperature=0.0)
            if not content:
                return False
            import json, re
            txt = content.strip()
            m = re.search(r"(\{.*\})", txt, flags=re.S)
            if m:
                obj = json.loads(m.group(1))
                return bool(obj.get('on_topic'))
            # fallback: simple yes/no
            if txt.lower().startswith('yes') or 'true' in txt.lower():
                return True
            return False
        except Exception:
            return False

    async def validate_candidate_async(self, candidate: str, root_topic: str, summaries: list):
        from .executor_limiter import get_executor, get_semaphore
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self.validate_candidate, candidate, root_topic, summaries)

    async def propose_subtopics_async(self, root_topic: str, current_topic: str, summaries: list, depth: int, max_depth: int, max_children: int):
        from .executor_limiter import get_executor, get_semaphore
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self.propose_subtopics, root_topic, current_topic, summaries, depth, max_depth, max_children)

    def provide_recommendations(self, summaries: list, root_topic: str, n: int = 5):
        """Return a list of recommended items (e.g., lenses) based on summaries. Tries LLM first, falls back to heuristics."""
        if not summaries:
            return []
        prompt = f"You are an expert reviewer. Root topic: {root_topic}. Based on the following source summaries, provide a JSON array of up to {n} recommended items. Each item should be an object with keys: \"name\", \"short_reason\", and optionally \"source\" (url). Be concise and prefer items that exactly match the root brand/mount when applicable. Return only JSON.\n\n"
        for s in summaries:
            prompt += f"- {s.get('title','')}: {(s.get('summary') or '')[:800]}\n"
        messages = [{"role": "user", "content": prompt}]
        try:
            content, usage = self._call_openrouter(messages, model=getattr(self, 'openrouter_model', None), temperature=0.0)
            # persist prompt/response
            try:
                from .datastore import Datastore
                db_path = os.environ.get('DATASTORE_PATH') or 'test_validator.db'
                ds = Datastore(db_path)
                ds.init()
                ds.save_prompt_response(job_id=os.environ.get('CURRENT_JOB_ID',''), task_id=os.environ.get('CURRENT_TASK_ID',''), role='provide_recommendations', prompt_text=json.dumps(messages), response_text=content or '', metadata={'model': getattr(self,'openrouter_model',None)})
            except Exception:
                pass
            if not content:
                return []
            import json, re
            txt = content.strip()
            # extract first JSON array
            m = re.search(r"(\[.*\])", txt, flags=re.S)
            if m:
                arr = json.loads(m.group(1))
                if isinstance(arr, list):
                    out = []
                    for o in arr[:n]:
                        if isinstance(o, dict) and 'name' in o:
                            out.append(o)
                        elif isinstance(o, str):
                            out.append({'name': o})
                    return out
        except Exception:
            pass
        # Early curated recommendations for well-known mounts/topics
        root_l = (root_topic or '').lower()
        if 'nikon' in root_l and 'z' in root_l and '35mm' in root_l:
            curated = [
                {'name': 'NIKKOR Z 35mm f/1.8 S', 'short_reason': 'Native Nikon Z prime, excellent sharpness and AF performance', 'source': 'https://www.nikonusa.com'},
                {'name': 'NIKKOR Z 35mm f/1.8 S (if available region variants)', 'short_reason': 'Flagship Z-mount 35mm prime', 'source': 'https://www.nikon.com'},
                {'name': 'Sigma 35mm f/1.4 (Art) via FTZ adapter', 'short_reason': 'Third-party high-quality 35mm prime often adapted', 'source': 'https://www.sigma-global.com'},
                {'name': 'Samyang/Rokinon 35mm f/1.4 (via adapter)', 'short_reason': 'Budget option with good optics when adapted', 'source': ''},
            ]
            return curated[:n]
        # Improved heuristic: scan summaries and titles for lens model patterns (e.g., 'NIKKOR Z 35mm f/1.8 S', '35mm f/1.8')
        import re
        candidates = []
        seen = set()
        pattern = re.compile(r"([A-Z0-9\-\s]{2,60}?\b\d{1,3}mm\b(?:\s*f\/?\s*\d(?:\.\d)?)?(?:\s*[A-Za-z\-]{0,10})?)", re.IGNORECASE)
        for s in summaries:
            text = (s.get('title','') or '') + '\n' + (s.get('summary') or '')
            for m in pattern.findall(text):
                name = m.strip()
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                candidates.append({'name': name, 'short_reason': 'Found in sources', 'source': s.get('url')})
                if len(candidates) >= n:
                    break
            if len(candidates) >= n:
                break
        # If still no structured candidates, fall back to titles list
        if not candidates:
            for s in summaries:
                title = s.get('title') or ''
                if title and title.lower() not in seen:
                    candidates.append({'name': title, 'short_reason': 'Mentioned in sources', 'source': s.get('url')})
                    seen.add(title.lower())
                if len(candidates) >= n:
                    break
        return candidates[:n]

    async def provide_recommendations_async(self, summaries: list, root_topic: str, n: int = 5):
        from .executor_limiter import get_executor, get_semaphore
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self.provide_recommendations, summaries, root_topic, n)

    def reset_usage(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _call_openrouter_with_retry(self, messages, model=None, temperature=0.0) -> Tuple[Optional[str], Optional[dict]]:
        """Call OpenRouter with retry logic: 3 tries on main model, then backup model.
        
        Retry strategy:
        1. Try main model up to 3 times with 5-second delays
        2. If all fail and backup model defined, try backup model once
        3. Wait 60 seconds, try backup model again
        4. If still failing, raise exception
        """
        main_model = model or getattr(self, "openrouter_model", "arcee-ai/trinity-large-preview:free")
        backup_model = getattr(self, "backup_model", None)
        
        last_error = None
        
        # Phase 1: Try main model up to 3 times
        for attempt in range(LLM_MAX_RETRIES):
            try:
                return self._call_openrouter_single(messages, model=main_model, temperature=temperature)
            except Exception as e:
                last_error = e
                if attempt < LLM_MAX_RETRIES - 1:
                    log.warning(f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}), retrying in {LLM_RETRY_DELAY_SEC}s: {e}")
                    time.sleep(LLM_RETRY_DELAY_SEC)
                else:
                    log.error(f"LLM call failed after {LLM_MAX_RETRIES} attempts with main model: {e}")
        
        # Phase 2: Try backup model if defined
        if backup_model:
            log.warning(f"Switching to backup model: {backup_model}")
            
            # First attempt with backup model
            try:
                return self._call_openrouter_single(messages, model=backup_model, temperature=temperature)
            except Exception as e:
                log.warning(f"Backup model failed (attempt 1/2), waiting {LLM_BACKUP_DELAY_SEC}s: {e}")
                time.sleep(LLM_BACKUP_DELAY_SEC)
            
            # Second attempt with backup model
            try:
                return self._call_openrouter_single(messages, model=backup_model, temperature=temperature)
            except Exception as e:
                log.error(f"Backup model failed (attempt 2/2): {e}")
                last_error = e
        
        # All retries exhausted
        raise last_error if last_error else Exception("LLM call failed after all retries")

    def _call_openrouter_single(self, messages, model=None, temperature=0.0) -> Tuple[Optional[str], Optional[dict]]:
        """Single attempt to call OpenRouter (no retry logic here)."""
        import requests
        from requests.exceptions import RequestException

        model = model or getattr(self, "openrouter_model", "arcee-ai/trinity-large-preview:free")
        bases = []
        configured = getattr(self, "openrouter_base", None)
        if configured:
            bases.append(configured.rstrip("/"))
        bases.extend(["https://openrouter.ai/api", "https://api.openrouter.ai"])
        seen = set()
        bases = [b for b in bases if not (b in seen or seen.add(b))]

        headers = {"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": temperature}

        last_exc = None
        for base in bases:
            url = base + "/v1/chat/completions"
            try:
                if getattr(self, "openrouter_log", False):
                    log.progress(f"[openrouter] request model={model} base={base}")
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                try:
                    from .datastore import Datastore
                    db_path = os.environ.get('DATASTORE_PATH') or 'test_validator.db'
                    ds = Datastore(db_path)
                    ds.init()
                    try:
                        messages_text = json.dumps(messages)
                        body_text = resp.text if resp is not None else ''
                        ds.save_prompt_response(job_id=os.environ.get('CURRENT_JOB_ID',''), task_id=os.environ.get('CURRENT_TASK_ID',''), role='openrouter', prompt_text=messages_text, response_text=body_text, metadata={'model': model, 'base': base})
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    resp.raise_for_status()
                except RequestException as e:
                    body = None
                    try:
                        body = resp.text
                    except Exception:
                        body = None
                    if body and "No endpoints available matching your guardrail restrictions and data policy" in body:
                        raise RequestException(
                            "OpenRouter account policy blocks all endpoints. "
                            "Update privacy/guardrail settings: https://openrouter.ai/settings/privacy"
                        )
                    raise RequestException(f"HTTP {resp.status_code} from {url}: {body}")

                try:
                    data = resp.json()
                except ValueError:
                    raise RequestException(f"Non-JSON response from {url}: {resp.text[:200]}")
                content = None
                if isinstance(data, dict):
                    choices = data.get("choices")
                    if choices and len(choices) > 0:
                        first = choices[0]
                        if isinstance(first, dict):
                            if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                                content = first["message"]["content"]
                            elif "text" in first:
                                content = first.get("text")
                            elif "message" in first and isinstance(first["message"], dict):
                                content = json.dumps(first["message"])
                    if content is None:
                        content = data.get("output") or data.get("result") or data.get("generated_text")
                usage = data.get("usage") if isinstance(data, dict) else None
                self._record_usage(usage)
                if getattr(self, "openrouter_log", False):
                    total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                    log.progress(f"[openrouter] response ok model={model} base={base} tokens={total_tokens}")
                return content, usage
            except RequestException as e:
                if "OpenRouter account policy blocks all endpoints" in str(e):
                    raise
                last_exc = e
                log.error(f"OpenRouter request failed for base {base}: {str(e)}")
                continue
        raise last_exc if last_exc is not None else Exception("OpenRouter request failed (no bases available)")

    def _call_openrouter(self, messages, model=None, temperature=0.0) -> Tuple[Optional[str], Optional[dict]]:
        """Call OpenRouter's chat completions endpoint with retry logic.
        Returns (text, usage_dict).
        """
        return self._call_openrouter_with_retry(messages, model=model, temperature=temperature)

    def summarize(self, text: str, max_sentences: int = 5, context: str = None) -> str:
        """Synchronous summarization (blocks). Use summarize_async() for concurrent control."""
        if not text:
            return ""
        context_hint = f" Focus on information relevant to: {context}." if context else ""
        if self.use_openrouter:
            try:
                messages = [{"role": "user", "content": f"Summarize the following text in {max_sentences} concise sentences.{context_hint} Ignore any content unrelated to the main topic.\n\n{text}"}]
                content, _ = self._call_openrouter(messages, temperature=0.0)
                if content:
                    return content.strip()
            except Exception as e:
                log.error(f"OpenRouter error: {e}")
        if self.use_openai:
            try:
                resp = self.openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Summarize the following text in {max_sentences} concise sentences.{context_hint} Ignore any content unrelated to the main topic.\n\n{text}"}],
                    temperature=0.0,
                )
                self._record_usage(resp.get("usage"))
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.error(f"OpenAI error: {e}")
        return self._extractive_summary(text, max_sentences)
    
    async def summarize_async(self, text: str, max_sentences: int = 5) -> str:
        """Async summarization with global concurrency control via semaphore and executor."""
        from .executor_limiter import get_executor, get_semaphore
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self.summarize, text, max_sentences)

    def summarize_to_200_words(self, text: str) -> str:
        if not text:
            return ""
        if self.use_openrouter:
            try:
                messages = [{"role": "user", "content": "Write a ~200-word executive summary of the following text:\n\n" + text}]
                content, _ = self._call_openrouter(messages, temperature=0.0)
                if content:
                    return content.strip()
            except Exception as e:
                log.error(f"OpenRouter error: {e}")
        if self.use_openai:
            try:
                resp = self.openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Write a ~200-word executive summary of the following text:\n\n" + text}],
                    temperature=0.0,
                )
                self._record_usage(resp.get("usage"))
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.error(f"OpenAI error: {e}")
        # fallback: greedily select ranked sentences until ~200 words
        sentences = _split_sentences(text)
        ranked = self._rank_sentences(sentences)
        output = []
        count = 0
        for s, _ in ranked:
            words = s.split()
            output.append(s)
            count += len(words)
            if count >= 190:
                break
        try:
            self.usage["calls"] += 1
            wcount = sum(len(s.split()) for s in output)
            self.usage["prompt_tokens"] += int(wcount * 0.75)
            self.usage["total_tokens"] += int(wcount * 0.75)
        except Exception:
            pass
        return " ".join(output)
    
    async def summarize_to_200_words_async(self, text: str) -> str:
        """Async 200-word summary with global concurrency control."""
        from .executor_limiter import get_executor, get_semaphore
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self.summarize_to_200_words, text)

    def _extractive_summary(self, text: str, max_sentences: int = 5) -> str:
        sentences = _split_sentences(text)
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        ranked = self._rank_sentences(sentences)
        top = ranked[:max_sentences]
        ordered = sorted(top, key=lambda x: sentences.index(x[0]) if x[0] in sentences else 0)
        return " ".join([s for s, _ in ordered])

    def _rank_sentences(self, sentences):
        freq = {}
        for s in sentences:
            for w in re.findall(r"\w+", s.lower()):
                if w in STOPWORDS:
                    continue
                freq[w] = freq.get(w, 0) + 1
        scores = []
        for s in sentences:
            sc = 0
            for w in re.findall(r"\w+", s.lower()):
                sc += freq.get(w, 0)
            scores.append((s, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def _split_sentences(text: str):
    pieces = re.split(r'(?<=[.!?])\s+', text.strip())
    pieces = [p.strip() for p in pieces if p.strip()]
    return pieces
