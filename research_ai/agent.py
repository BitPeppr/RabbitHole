import asyncio
import uuid
import re
import os

from .llm import _split_sentences


class Agent:
    def __init__(self, topic: str, depth: int = 0, max_depth: int = 2, max_children: int = 2, parent_id: str = None, parent_context: str = None):
        self.id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.topic = topic
        self.depth = depth
        self.max_depth = max_depth
        self.max_children = max_children
        # parent_context carries the original/root prompt or upstream context so child agents stay on topic
        self.parent_context = parent_context
        # root_topic: the canonical original prompt for this branch; if not provided, default to this agent's topic
        self.root_topic = parent_context or topic

    async def run(self, orchestrator):
        # fetch documents; ensure minimum 5 sources per user request
        try:
            configured = max(1, int(os.environ.get("SOURCE_COUNT", "3")))
        except Exception:
            configured = 3
        # use more sources for root-level requests to increase coverage
        if self.depth == 0:
            n_sources = max(8, configured)
        else:
            n_sources = max(5, configured)
        docs = await orchestrator.connector.fetch(self.topic, n=n_sources)
        summaries = []
        if not docs:
            summaries.append(
                {
                    "title": "Retrieval warning",
                    "url": "about:blank",
                    "summary": (
                        "No online sources were retrieved for this topic. "
                        "Check CONNECTOR, network connectivity, and provider restrictions."
                    ),
                }
            )
        
        # OPTIMIZATION: Process all documents in parallel instead of sequentially
        # This runs all summarizations concurrently (respecting global semaphore limits)
        async def process_doc(doc):
            text = doc.get("text", "")
            # include parent/root context in summarization prompt when available
            combined_text = (self.parent_context or "") + "\n\n" + text if self.parent_context else text
            summary = await orchestrator.llm.summarize_async(combined_text, max_sentences=5)
            # persist raw text
            path = None
            try:
                path = orchestrator.datastore.save_raw_artifact(text, filename=None)
            except Exception:
                pass
            # compute and save embeddings if available
            try:
                if getattr(orchestrator, "embeddings", None):
                    emb = orchestrator.embeddings.embed(text)
                    emb_id = f"emb-{uuid.uuid4().hex}"
                    doc_id = doc.get("url") or emb_id
                    orchestrator.datastore.save_embedding(emb_id, orchestrator.job_id, doc_id, emb, metadata={"title": doc.get("title"), "path": path})
            except Exception:
                pass
            return {"title": doc.get("title"), "url": doc.get("url"), "summary": summary}
        
        if docs:
            # Run all doc processing in parallel
            summaries = await asyncio.gather(*[process_doc(doc) for doc in docs])
            summaries = list(summaries)  # Convert from tuple
        
        spawned = 0
        result = {
            "agent_id": self.id,
            "parent_id": self.parent_id,
            "topic": self.topic,
            "depth": self.depth,
            "doc_count": len(docs),
            "spawned_children": spawned,
            "summaries": summaries,
        }
        # Provide direct recommendations for root-level 'best' queries to ensure useful output
        if self.depth == 0 and any(k in (self.root_topic or "").lower() for k in ("best", "top", "recommend")) and "lens" in (self.root_topic or "").lower():
            try:
                recs = await orchestrator.llm.provide_recommendations_async(summaries, self.root_topic, n=5)
                # If no recommendations found, perform a targeted expanded search (Nikon-specific) and retry
                if not recs:
                    extra_query = f"{self.root_topic} Nikon Z 35mm lens reviews"
                    extra_docs = await orchestrator.connector.fetch(extra_query, n=8)
                    extra_summaries = []
                    for d in extra_docs:
                        txt = d.get('text','')
                        ssum = await orchestrator.llm.summarize_async(txt, max_sentences=4)
                        extra_summaries.append({"title": d.get('title'), "url": d.get('url'), "summary": ssum})
                    combined = extra_summaries + summaries
                    recs2 = await orchestrator.llm.provide_recommendations_async(combined, self.root_topic, n=5)
                    result['recommendations'] = recs2 or []
                else:
                    result['recommendations'] = recs
            except Exception:
                result['recommendations'] = []
        # spawn children (LLM-driven proposer with validator; fallback to heuristic)
        if self.depth < self.max_depth:
            proposed = []
            max_retries = 2
            valid_subtopics = []
            # Try LLM-driven proposals with a small retry loop. Validate each candidate and stop when we have enough.
            for attempt in range(max_retries):
                try:
                    proposed = await orchestrator.llm.propose_subtopics_async(
                        root_topic=self.root_topic,
                        current_topic=self.topic,
                        summaries=summaries,
                        depth=self.depth,
                        max_depth=self.max_depth,
                        max_children=self.max_children,
                    ) or []
                except Exception:
                    proposed = []
                
                # OPTIMIZATION: Run validations in parallel instead of sequentially
                # Filter by heuristic first (fast), then validate remaining in parallel
                # Only validate up to max_children * 1.5 to avoid wasting LLM calls
                heuristic_passed = [st for st in proposed if self._is_on_topic(st, self.root_topic)]
                heuristic_passed = heuristic_passed[:int(self.max_children * 1.5)]  # Don't validate more than needed
                
                if heuristic_passed:
                    # Create validation tasks for all heuristic-passed candidates
                    async def validate_one(st):
                        try:
                            if getattr(orchestrator.llm, 'use_openrouter', False) or getattr(orchestrator.llm, 'use_openai', False):
                                return st, await orchestrator.llm.validate_candidate_async(st, self.root_topic, summaries)
                            return st, True
                        except Exception:
                            return st, False
                    
                    # Run all validations concurrently
                    validation_results = await asyncio.gather(*[validate_one(st) for st in heuristic_passed])
                    
                    # Process results and spawn children
                    for st, llm_ok in validation_results:
                        if len(valid_subtopics) >= self.max_children:
                            break
                        if not llm_ok:
                            continue
                        # build a richer parent_context for children: include root topic and short parent summaries
                        parent_summaries_text = "\n".join([f"- {s.get('title')}: { (s.get('summary') or '')[:200] }" for s in summaries[:5]])
                        child_parent_context = (self.root_topic or "") + "\n\nParent summaries:\n" + parent_summaries_text
                        child = Agent(topic=st, depth=self.depth + 1, max_depth=self.max_depth, max_children=self.max_children, parent_id=self.id, parent_context=child_parent_context)
                        await orchestrator.enqueue(child)
                        valid_subtopics.append(st)
                
                if len(valid_subtopics) >= self.max_children:
                    break
                # if none valid and we can retry, continue to ask proposer again
            # if still no valid proposals, fallback to heuristics and filter them
            if not valid_subtopics:
                fallback = self._derive_subtopics(self.topic, summaries, self.max_children * 2)
                for st in fallback:
                    if len(valid_subtopics) >= self.max_children:
                        break
                    if self._is_on_topic(st, self.root_topic):
                        parent_summaries_text = "\n".join([f"- {s.get('title')}: { (s.get('summary') or '')[:200] }" for s in summaries[:5]])
                        child_parent_context = (self.root_topic or "") + "\n\nParent summaries:\n" + parent_summaries_text
                        child = Agent(topic=st, depth=self.depth + 1, max_depth=self.max_depth, max_children=self.max_children, parent_id=self.id, parent_context=child_parent_context)
                        await orchestrator.enqueue(child)
                        valid_subtopics.append(st)
            spawned = len(valid_subtopics)
            result["spawned_children"] = spawned
        return result

    def _derive_subtopics(self, parent_topic: str, summaries, max_children: int = 2):
        """Heuristic fallback for proposing subtopics when LLM-driven proposal is unavailable."""
        subtopics = []
        seen = set()
        parent_norm = self._norm(parent_topic)
        for s in summaries:
            candidates = []
            title = (s.get("title") or "").strip()
            if "—" in title:
                title = title.split("—", 1)[0].strip()
            if title:
                candidates.append(title)
            text = s.get("summary", "")
            sentences = _split_sentences(text)
            if sentences:
                candidates.append(sentences[0])
            for raw in candidates:
                candidate = self._clean_candidate(raw)
                candidate = candidate if len(candidate) < 150 else candidate[:147] + "..."
                c_norm = self._norm(candidate)
                if not c_norm:
                    continue
                if c_norm in seen or c_norm == parent_norm or parent_norm.startswith(c_norm):
                    continue
                if self._too_similar(parent_norm, c_norm):
                    continue
                # Avoid generic boilerplate child topics that cause repetitive recursion.
                if len(c_norm.split()) >= 4:
                    subtopics.append(candidate)
                    seen.add(c_norm)
                    break
            if len(subtopics) >= max_children:
                break
        return subtopics

    def _is_on_topic(self, candidate: str, root_topic: str) -> bool:
        """Validate whether a candidate subtopic is sufficiently on-topic relative to the root prompt.

        Uses a lightweight lexical overlap heuristic and a title containment check. Returns True if
        the candidate appears topically related; False otherwise.
        """
        if not candidate or not root_topic:
            return False
        a = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        b = set(re.findall(r"[a-z0-9]+", root_topic.lower()))
        if not a or not b:
            return False
        # Filter out generic tokens that are not topic-discriminative
        stop_tokens = {"mount", "lens", "lenses", "best", "for", "the", "and", "of", "in", "on", "using", "with", "review", "reviews"}
        am = {t for t in a if t not in stop_tokens}
        bm = {t for t in b if t not in stop_tokens}
        # If the root contains a brand-like token (nikon, canon, sony, pentax, sigma, tamron), require brand to match
        brands = {"nikon", "canon", "sony", "pentax", "sigma", "tamron", "zeiss", "samyang", "fujifilm", "panasonic"}
        root_brands = bm & brands
        if root_brands:
            # candidate must contain at least one of the root brands
            if root_brands & a:
                return True
            return False
        # If there are no meaningful root tokens to check, fall back to token overlap with original sets
        if not bm:
            overlap = len(a & b) / max(1, min(len(a), len(b)))
            return overlap >= 0.25
        # compute overlap on meaningful tokens
        meaningful_overlap = len(am & bm) / max(1, min(len(am) if am else len(a), len(bm)))
        return meaningful_overlap >= 0.2

    def _clean_candidate(self, text: str) -> str:
        t = (text or "").strip()
        t = re.sub(r"^(the text (repeatedly )?(discusses|examines|focuses on)\s+)", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^(it (suggests|highlights|emphasizes)\s+)", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^((notably|critically|historically|importantly)\s+this\s+aspect\s+suggests\s+)", "", t, flags=re.IGNORECASE)
        return t.strip(" .")

    def _norm(self, text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def _too_similar(self, a: str, b: str) -> bool:
        sa = set(a.split())
        sb = set(b.split())
        if not sa or not sb:
            return False
        overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
        return overlap >= 0.8
