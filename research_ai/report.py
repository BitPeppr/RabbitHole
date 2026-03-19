import os
import re


class ReportGenerator:
    def __init__(self, llm, results, root_topic: str = None):
        self.llm = llm
        self.results = results
        self.root_topic = root_topic or (results[0].get("topic") if results else "Research Topic")

    async def generate(self, output_path: str):
        """Generate a coherent, well-structured report synthesizing all sources."""
        sections = self._build_sections(self.results)
        
        # Collect all summaries for synthesis
        all_summaries = []
        all_sources = []
        for r in sections:
            for s in r.get("summaries", []):
                all_summaries.append(s.get("summary", ""))
                all_sources.append({"title": s.get("title"), "url": s.get("url")})
        
        combined_text = "\n\n".join(all_summaries)
        
        # Generate executive summary
        exec_summary = await self.llm.summarize_to_200_words_async(combined_text)
        
        # Generate synthesized report body with logical structure
        report_body = await self._synthesize_report(combined_text, all_sources)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {self.root_topic}\n\n")
            f.write("## Summary\n\n")
            f.write(exec_summary + "\n\n")
            f.write("---\n\n")
            f.write(report_body)
            f.write("\n\n---\n\n")
            f.write("## Sources\n\n")
            seen = set()
            for src in all_sources:
                key = (src.get("title", ""), src.get("url", ""))
                if key in seen or not src.get("url"):
                    continue
                seen.add(key)
                f.write(f"- [{src.get('title', 'Untitled')}]({src.get('url', '')})\n")

        return os.path.abspath(output_path)

    async def _synthesize_report(self, combined_text: str, sources: list) -> str:
        """Use LLM to synthesize a coherent, logically structured report."""
        if not combined_text.strip():
            return "No content available to synthesize.\n"
        
        # Read word count range from environment
        try:
            min_words = int(os.environ.get("REPORT_MIN_WORDS", "500"))
        except ValueError:
            min_words = 500
        try:
            max_words = int(os.environ.get("REPORT_MAX_WORDS", "1000"))
        except ValueError:
            max_words = 1000
        
        prompt = f"""Based on the following research summaries about "{self.root_topic}", write a well-organized report.

IMPORTANT GUIDELINES:
- Organize the content into logical sections with clear headings (use ## for main sections, ### for subsections)
- Structure information in a natural progression (e.g., for a recipe: overview, ingredients, preparation, cooking steps, tips; for a product: overview, key features, comparisons, recommendations)
- Synthesize and combine information from multiple sources into coherent paragraphs
- Do NOT just list source summaries one after another
- Do NOT include source citations inline - sources will be listed separately
- Focus on practical, actionable information
- Remove redundant or repetitive information
- If the topic is a "how-to", organize as clear steps
- If the topic is a comparison/review, organize by criteria or options

Research summaries:
{combined_text[:12000]}

Write the organized report (markdown format, {min_words}-{max_words} words):"""

        if self.llm.use_openrouter or self.llm.use_openai:
            try:
                if self.llm.use_openrouter:
                    messages = [{"role": "user", "content": prompt}]
                    content, _ = self.llm._call_openrouter(messages, temperature=0.3)
                    if content:
                        return content.strip()
                elif self.llm.use_openai:
                    resp = self.llm.openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
        
        # Fallback: basic structure from summaries
        return self._fallback_structure(combined_text)

    def _fallback_structure(self, text: str) -> str:
        """Fallback when LLM synthesis unavailable."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Group into paragraphs of ~5 sentences
        paragraphs = []
        for i in range(0, len(sentences), 5):
            chunk = " ".join(sentences[i:i+5])
            if chunk.strip():
                paragraphs.append(chunk)
        return "\n\n".join(paragraphs[:20])  # Limit to reasonable length

    def _build_sections(self, results):
        grouped = {}
        for r in sorted(results, key=lambda x: (x.get("depth", 999), x.get("topic", ""))):
            topic = (r.get("topic") or "").strip()
            if not topic:
                continue
            key = self._topic_key(topic)
            if key not in grouped:
                grouped[key] = {"topic": topic, "depth": r.get("depth", 0), "summaries": []}
            else:
                grouped[key]["depth"] = min(grouped[key]["depth"], r.get("depth", 0))
                if len(topic) < len(grouped[key]["topic"]):
                    grouped[key]["topic"] = topic
            grouped[key]["summaries"].extend(r.get("summaries", []))

        out = []
        for sec in grouped.values():
            sec["summaries"] = self._dedupe_summaries(sec.get("summaries", []))
            if sec["summaries"]:
                out.append(sec)
        out.sort(key=lambda x: (x.get("depth", 0), x.get("topic", "")))
        return out

    def _dedupe_summaries(self, summaries):
        seen = set()
        out = []
        for s in summaries:
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            summary = self._compact_text((s.get("summary") or "").strip())
            summary_key = " ".join(re.findall(r"[a-z0-9]+", summary.lower()))
            key = (url.lower(), title.lower(), summary_key[:240])
            if key in seen:
                continue
            seen.add(key)
            out.append({"title": title, "url": url, "summary": summary})
        return out

    def _topic_key(self, text: str) -> str:
        t = text.lower()
        t = re.sub(r"^(the text (repeatedly )?(discusses|examines|focuses on)\s+)", "", t)
        t = re.sub(r"[^a-z0-9]+", " ", t)
        return " ".join(t.split())

    def _compact_text(self, text: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        kept = []
        seen = set()
        for p in parts:
            key = " ".join(re.findall(r"[a-z0-9]+", p.lower()))
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(p)
        return " ".join(kept)
