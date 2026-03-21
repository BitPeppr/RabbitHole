"""HTTP connector using requests + BeautifulSoup with a synchronous fetch wrapped for async use.

This connector performs Bing HTML search, then falls back to Wikipedia and arXiv
when search results are sparse. It returns real online sources whenever possible.
"""

import asyncio
import base64
import os
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

    def _sync_fetch(self, topic: str, n: int = 3):
        results = []
        seen_urls = set()
        topic_terms = self._terms(topic)
        bing_results = 0  # Track how many results came from Bing vs fallback
        try:
            # OPTIMIZATION: Use global session with connection pooling
            s = _get_session()
            links = []
            # Generate contextual query variants that preserve the full meaning
            # Plain topic first (quoted queries can give poor results on some topics)
            query_variants = [
                topic,  # Plain topic first
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
            fetch_ok, fetch_fail = 0, 0
            for title, url in ordered:
                if len(results) >= n:
                    break
                # OPTIMIZATION: Check URL cache first
                if url in _url_cache:
                    cached_title, text = _url_cache[url]
                    _url_cache_hits += 1
                    if _url_cache_hits % 20 == 0:
                        log.progress(f"[url_cache] hits={_url_cache_hits}")
                    fetch_ok += 1
                else:
                    # Use robust fetch with retry and rotating UAs
                    text = _fetch_url_content(s, url)
                    if text:
                        _url_cache[url] = (title, text)
                        fetch_ok += 1
                    else:
                        fetch_fail += 1
                
                if url and url not in seen_urls and text:
                    score = self._relevance_score(topic_terms, f"{title} {text[:1200]}", topic)
                    # accept slightly lower score to increase diversity but keep relevance
                    if score >= 1:
                        results.append({"title": title or url, "url": url, "text": text})
                        seen_urls.add(url)
            bing_results = len(results)
            _log_web_stats("", fetch_ok=fetch_ok, fetch_fail=fetch_fail)
        except Exception as e:
            log.error(f"[web_search] Bing search failed for '{topic}': {e}")
        # If we didn't reach n, fallback to Wikipedia but limit wikipedia dominance
        if len(results) < n:
            try:
                # Extract key terms for better Wikipedia search (long topics fail otherwise)
                search_query = self._extract_search_query(topic)
                query = urllib.parse.quote_plus(search_query)
                w_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={max(5, (n-len(results))*3)}"
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
                        # Wikipedia is a fallback - accept results without strict relevance filtering
                        # The Wikipedia search API already returns relevant results
                        results.append({"title": title, "url": page_url, "text": text})
                        seen_urls.add(page_url)
            except Exception as e:
                log.error(f"[web_search] Wikipedia fallback failed for '{topic[:50]}': {e}")
        # Fallback 2: arXiv abstracts for research-heavy topics
        if len(results) < n:
            try:
                # Extract key terms for arXiv search
                search_query = self._extract_search_query(topic, max_words=5)
                q = urllib.parse.quote_plus(search_query)
                a_url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max(5, (n-len(results))*2)}"
                # Use rate-limited request to avoid 429 errors
                a_resp = _arxiv_rate_limited_get(s, a_url, timeout=10)
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
                            # arXiv is a fallback - accept results without strict relevance filtering
                            results.append({"title": title or link, "url": link, "text": summary})
                            seen_urls.add(link)
                        if len(results) >= n:
                            break
                elif a_resp.status_code == 429:
                    log.warning(f"[web_search] arXiv rate limited for '{topic[:50]}' - skipping")
            except Exception as e:
                log.error(f"[web_search] arXiv fallback failed for '{topic[:50]}': {e}")
        
        # Track search stats: bing success vs fallback vs failed
        if not results:
            _log_web_stats("failed")
            log.warning(f"[web_search] No sources found for '{topic}' (Bing + Wikipedia + arXiv all failed)")
        elif bing_results >= n:
            _log_web_stats("bing")
            log.progress(f"[web_search] Found {len(results)} sources for '{topic[:50]}...'")
        else:
            _log_web_stats("fallback")
            log.progress(f"[web_search] Found {len(results)} sources for '{topic[:50]}...' (with fallback)")
        
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
