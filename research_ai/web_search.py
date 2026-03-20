"""HTTP connector using requests + BeautifulSoup with a synchronous fetch wrapped for async use.

This connector performs Bing HTML search, then falls back to Wikipedia and arXiv
when search results are sparse. It returns real online sources whenever possible.
"""

import asyncio
import base64
import time
import urllib.parse
import xml.etree.ElementTree as ET
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from .executor_limiter import get_executor, get_semaphore
from .logger import log

# Retry configuration for web search
BING_MAX_RETRIES = 3
BING_RETRY_DELAY_SEC = 2

# Global persistent session with connection pooling
_global_session = None

def _get_session() -> requests.Session:
    """Get or create a global requests session with connection pooling."""
    global _global_session
    if _global_session is None:
        _global_session = requests.Session()
        # Configure connection pooling and retries
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=20,  # Connection pool size
            pool_maxsize=20,      # Max connections per host
            max_retries=retry_strategy
        )
        _global_session.mount("http://", adapter)
        _global_session.mount("https://", adapter)
        _global_session.headers.update({"User-Agent": "Mozilla/5.0 (ResearchAI/0.1)"})
    return _global_session


# Global URL content cache: maps URL -> (title, text) to avoid re-fetching same pages
_url_cache = {}
_url_cache_hits = 0


def _extract_real_url(bing_url: str) -> str:
    """Extract the actual destination URL from a Bing tracking redirect URL.
    
    Bing wraps search result URLs like:
    https://www.bing.com/ck/a?...&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbQ...
    
    The 'u' parameter contains a base64-encoded URL prefixed with 'a1'.
    """
    if not bing_url or 'bing.com/ck/a' not in bing_url:
        return bing_url
    try:
        parsed = urllib.parse.urlparse(bing_url)
        params = urllib.parse.parse_qs(parsed.query)
        u_param = params.get('u', [''])[0]
        if u_param.startswith('a1'):
            # Remove 'a1' prefix and decode base64
            encoded = u_param[2:]
            # Add padding if needed
            padding = 4 - len(encoded) % 4
            if padding != 4:
                encoded += '=' * padding
            decoded = base64.urlsafe_b64decode(encoded).decode('utf-8')
            return decoded
    except Exception:
        pass
    return bing_url


class WebSearchConnector:
    async def fetch(self, topic: str, n: int = 3):
        """Fetch sources using global executor and semaphore for strict concurrency control."""
        loop = asyncio.get_event_loop()
        executor = get_executor()
        semaphore = get_semaphore()
        async with semaphore:
            return await loop.run_in_executor(executor, self._sync_fetch, topic, n)

    def _bing_search_with_retry(self, session, query: str, headers: dict) -> list:
        """Perform Bing search with retry logic. Returns list of (title, url) tuples."""
        links = []
        last_error = None
        
        for attempt in range(BING_MAX_RETRIES):
            try:
                resp = session.get("https://www.bing.com/search", params={"q": query}, timeout=10, headers=headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for item in soup.select("li.b_algo"):
                    h2_a = item.select_one("h2 a")
                    if h2_a:
                        href = h2_a.get('href')
                        title = h2_a.get_text(strip=True)
                        if href and title:
                            real_url = _extract_real_url(href)
                            links.append((title, real_url))
                # Success - return the links
                return links
            except Exception as e:
                last_error = e
                if attempt < BING_MAX_RETRIES - 1:
                    log.warning(f"Bing search failed (attempt {attempt + 1}/{BING_MAX_RETRIES}), retrying in {BING_RETRY_DELAY_SEC}s: {e}")
                    time.sleep(BING_RETRY_DELAY_SEC)
                else:
                    log.warning(f"Bing search failed after {BING_MAX_RETRIES} attempts: {e}")
        
        # All retries failed - return empty, will fall back to Wikipedia/arXiv
        return []

    def _sync_fetch(self, topic: str, n: int = 3):
        results = []
        seen_urls = set()
        topic_terms = self._terms(topic)
        try:
            # OPTIMIZATION: Use global session with connection pooling
            s = _get_session()
            links = []
            # Generate contextual query variants that preserve the full meaning
            # Avoid single-word matches by keeping topic phrases intact
            query_variants = [
                f'"{topic}"',  # Exact phrase search first
                topic,
                f"{topic} guide",
                f"{topic} tutorial",
                f"how to {topic}" if not topic.lower().startswith("how") else topic,
            ]
            # Collect many candidate links across query variants (with retry)
            for q in query_variants:
                new_links = self._bing_search_with_retry(s, q, s.headers)
                links.extend(new_links)
                # stop collecting once we have a reasonable pool
                if len(links) >= max(20, n * 6):
                    break
            # Deduplicate and prioritize review/retailer domains and non-wikipedia domains
            preferred_domains = [
                'dpreview.com', 'dxomark.com', 'bloomberg.com', 'bhphotovideo.com', 'bhphotovideo', 'adorama.com', 'adorama',
                'photographyblog.com', 'thephoblographer.com', 'thephoblographer', 'petaPixel.com', 'petapixel.com', 'lensrentals.com',
                'fstoppers.com', 'kenrockwell.com', 'digitalcameraworld.com', 'dpreview.com', 'ephotozine.com', 'cameralabs.com'
            ]
            ordered = []
            seen_link_urls = set()
            # first: preferred domains
            for title, url in links:
                if url in seen_link_urls:
                    continue
                for d in preferred_domains:
                    if d in url.lower():
                        ordered.append((title, url))
                        seen_link_urls.add(url)
                        break
            # second: non-wikipedia domains
            for title, url in links:
                if url in seen_link_urls:
                    continue
                if 'wikipedia.org' not in url.lower():
                    ordered.append((title, url))
                    seen_link_urls.add(url)
            # finally: wikipedia domains
            for title, url in links:
                if url in seen_link_urls:
                    continue
                ordered.append((title, url))
                seen_link_urls.add(url)
            # Limit candidate list
            ordered = ordered[: max(n * 8, 40)]
            # Fetch pages in order until we have n good results
            global _url_cache, _url_cache_hits
            for title, url in ordered:
                if len(results) >= n:
                    break
                try:
                    # OPTIMIZATION: Check URL cache first
                    if url in _url_cache:
                        cached_title, text = _url_cache[url]
                        _url_cache_hits += 1
                        if _url_cache_hits % 20 == 0:
                            log.progress(f"[url_cache] hits={_url_cache_hits}")
                    else:
                        r = s.get(url, timeout=10)
                        psoup = BeautifulSoup(r.text, "html.parser")
                        paragraphs = psoup.find_all("p")
                        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs[:8])
                        if not text:
                            text = r.text[:5000]
                        # Cache the fetched content
                        _url_cache[url] = (title, text)
                    
                    if url and url not in seen_urls and text:
                        score = self._relevance_score(topic_terms, f"{title} {text[:1200]}", topic)
                        # accept slightly lower score to increase diversity but keep relevance
                        if score >= 1:
                            results.append({"title": title or url, "url": url, "text": text})
                            seen_urls.add(url)
                except Exception:
                    continue
        except Exception:
            pass
        # If we didn't reach n, fallback to Wikipedia but limit wikipedia dominance
        if len(results) < n:
            try:
                query = urllib.parse.quote_plus(topic)
                w_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={max(3, (n-len(results))*3)}"
                w_resp = s.get(w_url, timeout=10)
                data = w_resp.json()
                for item in data.get("query", {}).get("search", []):
                    if len(results) >= n:
                        break
                    title = item.get("title", "").strip()
                    if not title:
                        continue
                    p_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    p_resp = s.get(p_url, timeout=10)
                    p_json = p_resp.json() if p_resp.status_code == 200 else {}
                    text = (p_json.get("extract") or "").strip()
                    page_url = p_json.get("content_urls", {}).get("desktop", {}).get("page", "")
                    if not page_url:
                        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    if page_url and page_url not in seen_urls and text:
                        score = self._relevance_score(topic_terms, f"{title} {text[:1200]}", topic)
                        if score >= 1:
                            results.append({"title": title, "url": page_url, "text": text})
                            seen_urls.add(page_url)
            except Exception:
                pass
        # Fallback 2: arXiv abstracts for research-heavy topics
        if len(results) < n:
            try:
                q = urllib.parse.quote_plus(topic)
                a_url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max(3, (n-len(results))*2)}"
                a_resp = s.get(a_url, timeout=10)
                if a_resp.status_code == 200:
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    root = ET.fromstring(a_resp.text)
                    for entry in root.findall("atom:entry", ns):
                        title = (entry.find("atom:title", ns).text or "").strip() if entry.find("atom:title", ns) is not None else ""
                        summary = (entry.find("atom:summary", ns).text or "").strip() if entry.find("atom:summary", ns) is not None else ""
                        link = ""
                        for l in entry.findall("atom:link", ns):
                            if l.get("type") == "text/html":
                                link = l.get("href")
                                break
                        if link and link not in seen_urls and summary:
                            score = self._relevance_score(topic_terms, f"{title} {summary[:1200]}", topic)
                            if score >= 1:
                                results.append({"title": title or link, "url": link, "text": summary})
                                seen_urls.add(link)
                        if len(results) >= n:
                            break
            except Exception:
                pass
        # Rank and return up to n results
        scored = []
        for r in results:
            combined = f"{r.get('title', '')} {r.get('text', '')[:1200]}"
            score = self._relevance_score(topic_terms, combined, topic)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [r for score, r in scored if score > 0]
        return (filtered or [r for _, r in scored])[:n]

    def _terms(self, text: str):
        raw = re.findall(r"[a-z0-9]+", (text or "").lower())
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "when", "where", "best", "how"}
        return {t for t in raw if len(t) > 2 and t not in stop}

    def _relevance_score(self, topic_terms, text: str, original_topic: str = ""):
        if not topic_terms:
            return 0
        text_lower = (text or "").lower()
        words = set(re.findall(r"[a-z0-9]+", text_lower))
        
        # Check if the original topic phrase appears (strong relevance signal)
        topic_lower = (original_topic or "").lower()
        phrase_bonus = 3 if topic_lower and topic_lower in text_lower else 0
        
        # Penalize results that match common words but miss the core topic
        # e.g., "make" appearing without "ramen" context
        overlap = len(topic_terms & words)
        
        # If topic has multiple significant terms, require at least half to match
        significant_terms = {t for t in topic_terms if len(t) > 3}
        if len(significant_terms) >= 2:
            sig_overlap = len(significant_terms & words)
            if sig_overlap < len(significant_terms) // 2:
                return 0  # Too few significant terms match
        
        return overlap + phrase_bonus
