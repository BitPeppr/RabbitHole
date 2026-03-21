"""HTTP connector using requests + BeautifulSoup with a synchronous fetch wrapped for async use.

This connector supports multiple search backends:
- Bing HTML scraping (default, free) with Wikipedia/arXiv fallbacks
- Brave Search API (requires BRAVE_API_KEY)
- SerpAPI/Google (requires SERPAPI_API_KEY)
- Tavily (requires TAVILY_API_KEY)
- Exa (requires EXA_API_KEY)

Set SEARCH_PROVIDER env var to choose: bing, brave, serpapi, tavily, exa
"""

import asyncio
import base64
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
import re
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from .executor_limiter import get_executor, get_semaphore
from .logger import log

# Search provider configuration
# SEARCH_PROVIDER: Primary provider (bing, brave, serpapi, tavily, exa)
# SEARCH_FALLBACK_CHAIN: Comma-separated fallback order when primary fails or returns insufficient results
# Example: SEARCH_FALLBACK_CHAIN=brave,serpapi,bing,tavily,wikipedia,arxiv
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "bing").lower()
SEARCH_FALLBACK_CHAIN = os.environ.get("SEARCH_FALLBACK_CHAIN", "wikipedia,arxiv").lower()

# API keys for paid providers
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")

# Retry configuration for web search
BING_MAX_RETRIES = 3
BING_RETRY_DELAY_SEC = 2

# Proxy configuration (optional - helps bypass IP blocks)
# Set PROXY_URL to use a specific proxy, e.g., "http://user:pass@proxy.example.com:8080"
PROXY_URL = os.environ.get("PROXY_URL", "")

# arXiv rate limiting - they enforce strict limits and return 429 when exceeded
# We use a global lock and timestamp to throttle requests across all workers
import threading
_arxiv_lock = threading.Lock()
_arxiv_last_request = 0.0
_arxiv_backoff_until = 0.0  # If we hit 429, back off until this time
ARXIV_MIN_DELAY_SEC = 3.0  # arXiv recommends max 1 request per 3 seconds

# Web search stats tracking
_stats_lock = threading.Lock()
_web_stats = {"bing": 0, "fallback": 0, "failed": 0, "fetch_ok": 0, "fetch_fail": 0}

def _log_web_stats(source_type: str, fetch_ok: int = 0, fetch_fail: int = 0):
    """Track and periodically log web search statistics."""
    with _stats_lock:
        if source_type:
            _web_stats[source_type] = _web_stats.get(source_type, 0) + 1
        _web_stats["fetch_ok"] += fetch_ok
        _web_stats["fetch_fail"] += fetch_fail
        
        total = _web_stats["bing"] + _web_stats["fallback"] + _web_stats["failed"]
        # Log every 5 failures or every 20 total searches
        if _web_stats["failed"] > 0 and _web_stats["failed"] % 5 == 0:
            pct_bing = 100 * _web_stats["bing"] / total if total else 0
            pct_fall = 100 * _web_stats["fallback"] / total if total else 0
            pct_fail = 100 * _web_stats["failed"] / total if total else 0
            fetch_total = _web_stats["fetch_ok"] + _web_stats["fetch_fail"]
            fetch_pct = 100 * _web_stats["fetch_ok"] / fetch_total if fetch_total else 0
            log.progress(f"[web_stats] searches={total} bing={pct_bing:.0f}% fallback={pct_fall:.0f}% failed={pct_fail:.0f}% | fetches={fetch_total} ok={fetch_pct:.0f}%")

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
        # Simple User-Agent works better than "realistic" browser headers
        # Bing actually blocks requests with full Chrome headers but allows simple ones
        # Accept-Language ensures English results regardless of server location
        _global_session.headers.update({
            "User-Agent": "Mozilla/5.0 (ResearchAI/0.1)",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
        # Configure proxy if set
        if PROXY_URL:
            _global_session.proxies = {
                "http": PROXY_URL,
                "https": PROXY_URL,
            }
            log.progress(f"[web_search] Using proxy: {PROXY_URL.split('@')[-1] if '@' in PROXY_URL else PROXY_URL}")
        
        # Log which search provider is configured
        if SEARCH_PROVIDER != "bing":
            log.progress(f"[web_search] Using search provider: {SEARCH_PROVIDER}")
    
    return _global_session


def _arxiv_rate_limited_get(session, url: str, timeout: int = 10):
    """Make a rate-limited GET request to arXiv API.
    
    arXiv enforces strict rate limits (~1 request per 3 seconds).
    This function ensures we respect that limit across all workers.
    """
    global _arxiv_last_request, _arxiv_backoff_until
    
    with _arxiv_lock:
        now = time.time()
        
        # Check if we're in a backoff period from a recent 429
        if now < _arxiv_backoff_until:
            wait_time = _arxiv_backoff_until - now
            log.progress(f"[web_search] arXiv rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            now = time.time()
        
        # Enforce minimum delay between requests
        elapsed = now - _arxiv_last_request
        if elapsed < ARXIV_MIN_DELAY_SEC:
            time.sleep(ARXIV_MIN_DELAY_SEC - elapsed)
        
        _arxiv_last_request = time.time()
    
    # Make the request outside the lock
    resp = session.get(url, timeout=timeout)
    
    # If we get a 429, set a longer backoff period
    if resp.status_code == 429:
        with _arxiv_lock:
            _arxiv_backoff_until = time.time() + 60  # Back off for 60 seconds
        log.warning("[web_search] arXiv returned 429 - backing off for 60s")
    
    return resp


# Global URL content cache: maps URL -> (title, text) to avoid re-fetching same pages
_url_cache = {}
_url_cache_hits = 0

# Rotating User-Agents for content fetch retries
_CONTENT_USER_AGENTS = [
    "Mozilla/5.0 (ResearchAI/0.1)",  # Simple bot UA (works for many sites)
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",  # Googlebot
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",  # Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",  # Firefox
]


def _fetch_url_content(session, url: str, timeout: int = 12) -> str:
    """Fetch URL content with retry and rotating User-Agents.
    
    Tries multiple User-Agents on failure since some sites block specific UAs.
    Returns extracted text or empty string on failure.
    """
    for i, ua in enumerate(_CONTENT_USER_AGENTS):
        try:
            headers = {"User-Agent": ua}
            r = session.get(url, timeout=timeout, headers=headers, verify=True)
            r.raise_for_status()
            
            # Parse and extract text
            psoup = BeautifulSoup(r.text, "html.parser")
            
            # Remove script/style elements
            for tag in psoup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            paragraphs = psoup.find_all("p")
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs[:10])
            
            # Fallback to raw text if no paragraphs
            if not text or len(text) < 100:
                text = psoup.get_text(separator="\n", strip=True)[:6000]
            
            if text and len(text) > 50:
                return text
                
        except requests.exceptions.SSLError:
            # Try without SSL verification as last resort
            if i == len(_CONTENT_USER_AGENTS) - 1:
                try:
                    r = session.get(url, timeout=timeout, headers=headers, verify=False)
                    psoup = BeautifulSoup(r.text, "html.parser")
                    paragraphs = psoup.find_all("p")
                    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs[:10])
                    if text and len(text) > 50:
                        return text
                except Exception:
                    pass
        except requests.exceptions.Timeout:
            # Don't retry timeouts with different UA - site is just slow
            break
        except Exception:
            # Try next UA
            continue
    
    return ""


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
                # setlang=en ensures English results regardless of server geolocation
                # Note: mkt param breaks multi-word queries, so we only use setlang
                resp = session.get("https://www.bing.com/search", params={
                    "q": query,
                    "setlang": "en",
                }, timeout=10, headers=headers)
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
                # Log if Bing returned no results (helps debug)
                if not links:
                    log.warning(f"[web_search] Bing returned empty results for '{query[:50]}...'")
                # Return the links (even if empty)
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

    def _brave_search(self, session, query: str, n: int = 10) -> list:
        """Search using Brave Search API. Returns list of result dicts."""
        if not BRAVE_API_KEY:
            log.error("[web_search] BRAVE_API_KEY not set")
            return []
        
        results = []
        try:
            resp = session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": n},
                headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            
            for item in data.get("web", {}).get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                # Brave provides description and extra_snippets for context
                text = item.get("description", "")
                extra = item.get("extra_snippets", [])
                if extra:
                    text = text + " " + " ".join(extra)
                
                if url and text:
                    results.append({"title": title or url, "url": url, "text": text.strip()})
            
            log.progress(f"[web_search] Brave returned {len(results)} results for '{query[:50]}...'")
        except Exception as e:
            log.error(f"[web_search] Brave search failed: {e}")
        
        return results

    def _serpapi_search(self, session, query: str, n: int = 10) -> list:
        """Search using SerpAPI (Google). Returns list of result dicts."""
        if not SERPAPI_API_KEY:
            log.error("[web_search] SERPAPI_API_KEY not set")
            return []
        
        results = []
        try:
            resp = session.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": SERPAPI_API_KEY,
                    "num": n,
                    "hl": "en",
                    "gl": "us"
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract organic results
            for item in data.get("organic_results", []):
                title = item.get("title", "")
                url = item.get("link", "")
                text = item.get("snippet", "")
                
                if url and text:
                    results.append({"title": title or url, "url": url, "text": text.strip()})
            
            log.progress(f"[web_search] SerpAPI returned {len(results)} results for '{query[:50]}...'")
        except Exception as e:
            log.error(f"[web_search] SerpAPI search failed: {e}")
        
        return results

    def _tavily_search(self, session, query: str, n: int = 10) -> list:
        """Search using Tavily API. Returns list of result dicts."""
        if not TAVILY_API_KEY:
            log.error("[web_search] TAVILY_API_KEY not set")
            return []
        
        results = []
        try:
            resp = session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": n,
                    "include_answer": False,
                    "include_raw_content": False,
                    "search_depth": "advanced"
                },
                headers={"Content-Type": "application/json"},
                timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
            
            for item in data.get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                text = item.get("content", "")
                
                if url and text:
                    results.append({"title": title or url, "url": url, "text": text.strip()})
            
            log.progress(f"[web_search] Tavily returned {len(results)} results for '{query[:50]}...'")
        except Exception as e:
            log.error(f"[web_search] Tavily search failed: {e}")
        
        return results

    def _exa_search(self, session, query: str, n: int = 10) -> list:
        """Search using Exa API. Returns list of result dicts."""
        if not EXA_API_KEY:
            log.error("[web_search] EXA_API_KEY not set")
            return []
        
        results = []
        try:
            resp = session.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "numResults": n,
                    "type": "auto",
                    "contents": {"text": {"maxCharacters": 3000}}
                },
                headers={
                    "x-api-key": EXA_API_KEY,
                    "Content-Type": "application/json"
                },
                timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
            
            for item in data.get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                text = item.get("text", "")
                
                if url and text:
                    results.append({"title": title or url, "url": url, "text": text.strip()})
            
            log.progress(f"[web_search] Exa returned {len(results)} results for '{query[:50]}...'")
        except Exception as e:
            log.error(f"[web_search] Exa search failed: {e}")
        
        return results

    def _sync_fetch(self, topic: str, n: int = 3):
        """Fetch sources using the configured search provider with fallback chain."""
        results = []
        seen_urls = set()
        topic_terms = self._terms(topic)
        s = _get_session()
        
        # Build provider chain: primary + fallbacks
        primary = SEARCH_PROVIDER.strip()
        fallbacks = [f.strip() for f in SEARCH_FALLBACK_CHAIN.split(",") if f.strip()]
        
        # Remove primary from fallbacks if present to avoid duplicate
        providers = [primary] + [f for f in fallbacks if f != primary]
        
        log.progress(f"[web_search] Provider chain: {' → '.join(providers)}")
        
        # Try each provider in order until we have enough results
        providers_used = []
        for provider in providers:
            if len(results) >= n:
                break
            
            needed = n - len(results)
            provider_results = self._fetch_from_provider(s, provider, topic, needed * 2, seen_urls)
            
            if provider_results:
                providers_used.append(provider)
                for item in provider_results:
                    url = item.get("url", "")
                    if url and url not in seen_urls:
                        results.append(item)
                        seen_urls.add(url)
                        if len(results) >= n:
                            break
            
            if len(results) < n and provider_results:
                log.progress(f"[web_search] {provider} returned {len(provider_results)}, need {n - len(results)} more...")
        
        # Log results
        if not results:
            _log_web_stats("failed")
            log.warning(f"[web_search] No sources found for '{topic}' (all providers failed)")
        else:
            primary_stat = providers_used[0] if providers_used else "failed"
            _log_web_stats(primary_stat if len(providers_used) == 1 else "fallback")
            log.progress(f"[web_search] Found {len(results)} sources via {'+'.join(providers_used)} for '{topic[:50]}...'")
        
        # Score and return top n results
        scored = []
        for r in results:
            combined = f"{r.get('title', '')} {r.get('text', '')[:1200]}"
            score = self._relevance_score(topic_terms, combined, topic)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [r for score, r in scored if score > 0]
        return (filtered or [r for _, r in scored])[:n]

    def _fetch_from_provider(self, session, provider: str, topic: str, n: int, seen_urls: set) -> list:
        """Fetch from a specific provider. Returns list of result dicts."""
        provider = provider.lower().strip()
        
        if provider == "brave":
            return self._brave_search(session, topic, n)
        elif provider == "serpapi":
            return self._serpapi_search(session, topic, n)
        elif provider == "tavily":
            return self._tavily_search(session, topic, n)
        elif provider == "exa":
            return self._exa_search(session, topic, n)
        elif provider == "bing":
            return self._bing_fetch(session, topic, n, seen_urls)
        elif provider == "wikipedia":
            return self._wikipedia_search(session, topic, n, seen_urls)
        elif provider == "arxiv":
            return self._arxiv_search(session, topic, n, seen_urls)
        else:
            log.warning(f"[web_search] Unknown provider: {provider}")
            return []

    def _bing_fetch(self, session, topic: str, n: int, seen_urls: set) -> list:
        """Fetch from Bing with content extraction."""
        results = []
        links = []
        
        # Generate query variants
        query_variants = [
            topic,
            f"{topic} guide",
            f"{topic} tutorial",
            f"how to {topic}" if not topic.lower().startswith("how") else topic,
        ]
        
        for q in query_variants:
            new_links = self._bing_search_with_retry(session, q, session.headers)
            links.extend(new_links)
            if len(links) >= max(20, n * 6):
                break
        
        # Deduplicate and prioritize
        preferred_domains = [
            'dpreview.com', 'dxomark.com', 'bloomberg.com', 'bhphotovideo.com',
            'adorama.com', 'photographyblog.com', 'petapixel.com', 'lensrentals.com',
            'fstoppers.com', 'kenrockwell.com', 'digitalcameraworld.com'
        ]
        ordered = []
        seen_link_urls = set()
        
        # Preferred domains first
        for title, url in links:
            if url in seen_link_urls or url in seen_urls:
                continue
            for d in preferred_domains:
                if d in url.lower():
                    ordered.append((title, url))
                    seen_link_urls.add(url)
                    break
        
        # Non-wikipedia next
        for title, url in links:
            if url in seen_link_urls or url in seen_urls:
                continue
            if 'wikipedia.org' not in url.lower():
                ordered.append((title, url))
                seen_link_urls.add(url)
        
        # Wikipedia last
        for title, url in links:
            if url in seen_link_urls or url in seen_urls:
                continue
            ordered.append((title, url))
            seen_link_urls.add(url)
        
        ordered = ordered[:max(n * 8, 40)]
        
        # Fetch content
        global _url_cache, _url_cache_hits
        topic_terms = self._terms(topic)
        
        for title, url in ordered:
            if len(results) >= n:
                break
            
            if url in _url_cache:
                cached_title, text = _url_cache[url]
                _url_cache_hits += 1
            else:
                text = _fetch_url_content(session, url)
                if text:
                    _url_cache[url] = (title, text)
            
            if url and text:
                score = self._relevance_score(topic_terms, f"{title} {text[:1200]}", topic)
                if score >= 1:
                    results.append({"title": title or url, "url": url, "text": text})
        
        return results

    def _wikipedia_search(self, session, topic: str, n: int, seen_urls: set) -> list:
        """Fetch from Wikipedia API."""
        results = []
        try:
            search_query = self._extract_search_query(topic)
            query = urllib.parse.quote_plus(search_query)
            w_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={max(5, n*3)}"
            w_resp = session.get(w_url, timeout=10)
            data = w_resp.json()
            
            for item in data.get("query", {}).get("search", []):
                if len(results) >= n:
                    break
                title = item.get("title", "").strip()
                if not title:
                    continue
                p_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                p_resp = session.get(p_url, timeout=10)
                p_json = p_resp.json() if p_resp.status_code == 200 else {}
                text = (p_json.get("extract") or "").strip()
                page_url = p_json.get("content_urls", {}).get("desktop", {}).get("page", "")
                if not page_url:
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                if page_url and page_url not in seen_urls and text:
                    results.append({"title": title, "url": page_url, "text": text})
        except Exception as e:
            log.error(f"[web_search] Wikipedia search failed: {e}")
        
        return results

    def _arxiv_search(self, session, topic: str, n: int, seen_urls: set) -> list:
        """Fetch from arXiv API."""
        results = []
        try:
            search_query = self._extract_search_query(topic, max_words=5)
            q = urllib.parse.quote_plus(search_query)
            a_url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max(5, n*2)}"
            a_resp = _arxiv_rate_limited_get(session, a_url, timeout=10)
            
            if a_resp.status_code == 200:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(a_resp.text)
                for entry in root.findall("atom:entry", ns):
                    if len(results) >= n:
                        break
                    title = (entry.find("atom:title", ns).text or "").strip() if entry.find("atom:title", ns) is not None else ""
                    summary = (entry.find("atom:summary", ns).text or "").strip() if entry.find("atom:summary", ns) is not None else ""
                    link = ""
                    for l in entry.findall("atom:link", ns):
                        if l.get("type") == "text/html":
                            link = l.get("href")
                            break
                    if link and link not in seen_urls and summary:
                        results.append({"title": title or link, "url": link, "text": summary})
            elif a_resp.status_code == 429:
                log.warning(f"[web_search] arXiv rate limited - skipping")
        except Exception as e:
            log.error(f"[web_search] arXiv search failed: {e}")
        
        return results

    def _terms(self, text: str):
        raw = re.findall(r"[a-z0-9]+", (text or "").lower())
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "when", "where", "best", "how"}
        return {t for t in raw if len(t) > 2 and t not in stop}

    def _extract_search_query(self, topic: str, max_words: int = 8) -> str:
        """Extract key search terms from a long topic for API queries.
        
        Long natural language topics like 'What are the best uses for llms?...'
        need to be shortened to key terms like 'llm uses automation python'.
        """
        # Remove common question words and filler
        stop_words = {
            'what', 'are', 'the', 'best', 'uses', 'for', 'how', 'to', 'can', 'i',
            'with', 'and', 'or', 'is', 'in', 'of', 'a', 'an', 'that', 'this',
            'give', 'me', 'lots', 'ideas', 'preferably', 'please', 'would', 'like',
            'some', 'any', 'other', 'others', 'etc', 'e', 'g', 'such', 'as',
            'infinite', 'token', 'time'  # Too generic
        }
        
        # Extract words, keeping order initially
        words = re.findall(r"[a-z0-9]+", topic.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique_keywords.append(w)
        
        # Take first N keywords (preserves topic order/importance)
        selected = unique_keywords[:max_words]
        
        # If we got very few keywords, be more lenient
        if len(selected) < 3:
            # Include shorter words
            selected = [w for w in words if w not in stop_words and len(w) > 1][:max_words]
        
        return ' '.join(selected) if selected else topic[:100]

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
