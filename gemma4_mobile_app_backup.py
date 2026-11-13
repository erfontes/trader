#!/usr/bin/env python3
"""
Gemma 4 Mobile Web UI & Web Search Gateway
------------------------------------------
Zero-dependency, lightweight mobile web interface with live Web Search,
URL/Article Summarization, and Multimodal Image & PDF Import built
specifically for gemma4_server.py / gemma4_fast_server.py (port 11434).

Runs completely isolated from the WSL 'vel' environment and trader codebase.
"""

import os
import sys

# Support both console (python.exe) and headless background (pythonw.exe) execution
_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobile_ui.log")
if sys.stdout is None or sys.stderr is None:
    _log_file = open(_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _log_file
    sys.stderr = _log_file
else:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure Anaconda/Miniconda OpenSSL DLLs on Windows are discoverable before importing ssl/requests
_conda_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_bin):
    os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_conda_bin)
        except Exception:
            pass

import re
import json
import time
import socket
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import requests
from bs4 import BeautifulSoup

# ==============================================================================
# Configuration (Strictly non-intrusive to gemma4_server.py)
# ==============================================================================
GEMMA_API_BASE = os.getenv("GEMMA_API_BASE", "http://127.0.0.1:11434")
UI_HOST = os.getenv("MOBILE_UI_HOST", "0.0.0.0")
UI_PORT = int(os.getenv("MOBILE_UI_PORT", "8088"))

# Keep context generous (supports both 3072-token original server and 4096-token fast server)
MAX_SCRAPE_CHARS_PER_PAGE = 4500
MAX_TOTAL_CONTEXT_CHARS = 8500
MAX_HISTORY_CHARS = 8000

# Cache the most recent web/URL/PDF context so follow-up questions remember details
LAST_WEB_CONTEXT = {"context": "", "sources": [], "timestamp": 0.0}

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Mobile Safari/537.36"
)
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def get_lan_ip() -> str:
    """Detect primary LAN IPv4 address (e.g. 192.168.x.x) for phone access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ==============================================================================
# Web Search & URL Scraper / Summarizer Engine
# ==============================================================================
def unwrap_ddg_url(href: str) -> str:
    """Extract clean target URL if DuckDuckGo wraps it in a redirect."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc:
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return qs["uddg"][0]
    return href


RECENCY_QUERY_PATTERNS = re.compile(
    r"\b(today|tonight|yesterday|this week|latest|breaking|current|now|live|"
    r"score|scores|won|lose|lost|game|match|vs|news|headline|headlines|price|stock|weather)\b",
    re.IGNORECASE,
)


def search_google_live(query: str, max_results: int = 4, is_recent: bool = False) -> list:
    """
    Search Google's live RSS index (news.google.com/rss/search) — 100% free,
    no API key, zero CAPTCHAs, and returns real-time sports/news/finance results with timestamps.
    """
    results = []
    try:
        g_query = query.strip()
        if is_recent and "when:" not in g_query:
            cleaned = re.sub(r"\b(today|tonight|right now)\b", "", g_query, flags=re.I).strip()
            g_query = f"{cleaned or g_query} when:3d"

        encoded_q = urllib.parse.quote_plus(g_query)
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, headers={"User-Agent": DESKTOP_UA}, timeout=7)
        if resp.status_code == 200 and resp.text:
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                source_el = item.find("source")
                publisher = source_el.text.strip() if (source_el is not None and source_el.text) else "Google News"
                desc_html = item.findtext("description") or ""
                desc_text = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)

                if title and link:
                    snippet_parts = []
                    if pub_date:
                        snippet_parts.append(f"[Published: {pub_date} · {publisher}]")
                    if desc_text and desc_text != title:
                        snippet_parts.append(desc_text[:280])
                    results.append({
                        "title": title,
                        "url": link,
                        "snippet": " ".join(snippet_parts) or f"Live result via {publisher}",
                        "engine": "google",
                    })
                if len(results) >= max_results:
                    break
    except Exception as e:
        print(f"[Search Warning] Google live search error: {e}")
    return results


def search_duckduckgo(query: str, max_results: int = 4, is_recent: bool = False) -> list:
    """Search DuckDuckGo HTML endpoint for general web pages."""
    results = []
    try:
        ddg_q = query.strip()
        if is_recent and re.search(r"\b(today|tonight|yesterday)\b", ddg_q, re.I):
            month_year = time.strftime("%B %Y")
            if month_year.lower() not in ddg_q.lower():
                ddg_q = f"{ddg_q} {month_year}"

        post_data = {"q": ddg_q, "kl": "us-en"}
        if is_recent:
            post_data["df"] = "w"

        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data=post_data,
            headers={"User-Agent": DESKTOP_UA, "Referer": "https://html.duckduckgo.com/"},
            timeout=8,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for block in soup.select(".result"):
                a_tag = block.select_one("a.result__a")
                snip_tag = block.select_one(".result__snippet")
                if not a_tag:
                    continue
                title = a_tag.get_text(" ", strip=True)
                url = unwrap_ddg_url(a_tag.get("href", ""))
                snippet = snip_tag.get_text(" ", strip=True) if snip_tag else ""
                if url and url.startswith("http") and "duckduckgo.com/y.js" not in url:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "engine": "ddg",
                    })
                if len(results) >= max_results:
                    break
    except Exception as e:
        print(f"[Search Warning] DuckDuckGo search error: {e}")
    return results


def search_web(query: str, max_results: int = 6) -> list:
    """
    Hybrid Search Engine: Combines Google Live Search (news.google.com/rss) +
    DuckDuckGo Web Search with smart date/recency prioritization.
    """
    is_recent = bool(RECENCY_QUERY_PATTERNS.search(query))
    google_hits = search_google_live(query, max_results=4 if is_recent else 2, is_recent=is_recent)
    ddg_hits = search_duckduckgo(query, max_results=4, is_recent=is_recent)

    if is_recent and not ddg_hits:
        ddg_hits = search_duckduckgo(query, max_results=4, is_recent=False)

    if is_recent:
        combined = google_hits + ddg_hits
    else:
        combined = ddg_hits + google_hits

    return combined[:max_results]


def fetch_page_text(url: str, max_chars: int = MAX_SCRAPE_CHARS_PER_PAGE, use_jina_fallback: bool = True) -> dict:
    """Fetch and clean readable article text from a URL, with Jina Reader fallback for JS/SPA pages."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    title = url
    cleaned = ""
    err_msg = ""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": DESKTOP_UA, "Accept": "text/html,application/xhtml+xml"},
            timeout=8,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        if "text/html" in content_type or "text/plain" in content_type or "xml" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.get_text(" ", strip=True)

            for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form", "iframe"]):
                tag.decompose()

            main_node = soup.find("article") or soup.find("main") or soup.body or soup
            paragraphs = []
            seen = set()
            for el in main_node.find_all(["h1", "h2", "h3", "p", "li"]):
                txt = el.get_text(" ", strip=True)
                if len(txt) >= 25 and txt not in seen:
                    seen.add(txt)
                    paragraphs.append(txt)
            cleaned = "\n".join(paragraphs)
            if len(cleaned) < 200:
                cleaned = main_node.get_text(" ", strip=True)
    except Exception as e:
        err_msg = str(e)

    if use_jina_fallback and len(cleaned.strip()) < 220:
        try:
            jina_url = f"https://r.jina.ai/{url}"
            jr = requests.get(jina_url, headers={"User-Agent": DESKTOP_UA}, timeout=10)
            if jr.status_code == 200 and len(jr.text.strip()) > len(cleaned):
                cleaned = jr.text.strip()
                err_msg = ""
        except Exception:
            pass

    cleaned = re.sub(r"\s{3,}", "\n\n", cleaned).strip()
    if cleaned:
        return {"url": url, "title": title[:140], "text": cleaned[:max_chars], "error": ""}
    return {"url": url, "title": title[:140], "text": "", "error": err_msg or "No readable text found"}


def extract_urls(text: str) -> list:
    """Find any http(s) URLs in user prompt."""
    return re.findall(r"https?://[^\s<>\"')]+", text or "")


# ==============================================================================
# HTML / PWA Frontend (Single-File Mobile App)
# ==============================================================================
MOBILE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover, interactive-widget=resizes-content">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0b0f19">
<link rel="icon" type="image/jpeg" href="/crest.jpg?v=2">
<link rel="apple-touch-icon" href="/crest.jpg?v=2">
<link rel="manifest" href="/manifest.json">
<title>Mega FF AI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<style>
  :root {
    --bg: #0b0f19;
    --panel: #131a2a;
    --panel-2: #1c2539;
    --border: rgba(255,255,255,0.09);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-2: #6366f1;
    --green: #22c55e;
    --amber: #f59e0b;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html {
    margin: 0; padding: 0;
    width: 100%; height: 100%;
    background: var(--bg);
    overflow: hidden;
    overscroll-behavior: none;
  }
  body {
    margin: 0; padding: 0;
    position: fixed;
    top: 0; left: 0; right: 0;
    width: 100%;
    height: 100dvh;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    overflow: hidden;
    overscroll-behavior: none;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  header {
    padding: calc(env(safe-area-inset-top, 0px) + 10px) 14px 10px;
    background: rgba(19, 26, 42, 0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    z-index: 20;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }
  .logo {
    width: 38px; height: 38px;
    border-radius: 10px;
    object-fit: cover;
    border: 1.5px solid rgba(218, 178, 78, 0.6);
    flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(218, 178, 78, 0.25);
    background: #0c181c;
  }
  .hero-crest {
    width: 136px;
    height: auto;
    border-radius: 16px;
    border: 1.5px solid rgba(218, 178, 78, 0.45);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.55), 0 0 18px rgba(218, 178, 78, 0.18);
    margin-bottom: 10px;
  }
  .brand-text h1 {
    margin: 0; font-size: 15px; font-weight: 700;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .status-line {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--muted);
    margin-top: 2px;
  }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--amber);
    flex-shrink: 0;
  }
  .dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot.offline { background: var(--danger); }

  .header-actions {
    display: flex; align-items: center; gap: 6px;
  }
  .icon-btn {
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: flex; align-items: center; gap: 5px;
  }
  .icon-btn:active { transform: scale(0.96); }

  /* Chat Area */
  #chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 14px 12px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    scroll-behavior: smooth;
  }

  .empty-state {
    margin: auto 0;
    text-align: center;
    padding: 16px 10px;
  }
  .empty-state h2 {
    font-size: 20px; margin: 0 0 6px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .empty-state p {
    font-size: 13px; color: var(--muted);
    margin: 0 auto 18px; max-width: 340px; line-height: 1.45;
  }
  .quick-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    max-width: 440px;
    margin: 0 auto;
  }
  .quick-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 11px 12px;
    text-align: left;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .quick-card:active { background: var(--panel-2); border-color: var(--accent); }
  .quick-card .q-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 3px; }
  .quick-card .q-sub { font-size: 11px; color: var(--muted); }

  /* Messages */
  .msg-row {
    display: flex;
    flex-direction: column;
    max-width: 92%;
    animation: fadeIn 0.18s ease-out;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  .msg-row.user { align-self: flex-end; align-items: flex-end; }
  .msg-row.assistant { align-self: flex-start; align-items: flex-start; max-width: 96%; }

  .bubble {
    padding: 11px 14px;
    border-radius: 16px;
    font-size: 14.5px;
    line-height: 1.52;
    word-wrap: break-word;
    overflow-wrap: anywhere;
  }
  .msg-row.user .bubble {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg-row.assistant .bubble {
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-left-radius: 4px;
    width: 100%;
  }

  /* Formatted Rich Markdown Inside Assistant Bubble */
  .bubble p { margin: 0 0 10px; }
  .bubble p:last-child { margin-bottom: 0; }
  .bubble strong { color: #f8fafc; font-weight: 700; }
  .bubble em { color: #cbd5e1; }
  .bubble h1, .bubble h2, .bubble h3, .bubble h4 {
    margin: 14px 0 8px;
    line-height: 1.3;
    color: #f8fafc;
    font-weight: 700;
  }
  .bubble h1:first-child, .bubble h2:first-child, .bubble h3:first-child { margin-top: 2px; }
  .bubble h1 { font-size: 17.5px; border-bottom: 1px solid var(--border); padding-bottom: 5px; }
  .bubble h2 { font-size: 16px; color: #7dd3fc; border-bottom: 1px solid rgba(56,189,248,0.18); padding-bottom: 4px; }
  .bubble h3 { font-size: 15px; color: #fcd34d; }
  .bubble h4 { font-size: 14px; color: #a5b4fc; }

  .bubble ul, .bubble ol {
    margin: 6px 0 12px;
    padding-left: 20px;
  }
  .bubble li {
    margin-bottom: 6px;
    line-height: 1.5;
  }
  .bubble ul li::marker { color: var(--accent); }
  .bubble ol li::marker { color: #818cf8; font-weight: 700; }

  .bubble hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);
    margin: 14px 0;
  }

  .bubble blockquote {
    margin: 10px 0;
    padding: 9px 12px;
    background: rgba(56, 189, 248, 0.08);
    border-left: 3px solid var(--accent);
    border-radius: 4px 10px 10px 4px;
    color: #e2e8f0;
  }

  .bubble code {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 2px 6px;
    border-radius: 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    color: #38bdf8;
  }

  .code-card {
    margin: 10px 0;
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .code-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    background: rgba(255,255,255,0.04);
    border-bottom: 1px solid var(--border);
    font-size: 11.5px;
    color: var(--muted);
    font-family: ui-monospace, monospace;
  }
  .copy-code-btn {
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
  }
  .code-card pre {
    margin: 0;
    padding: 10px 12px;
    overflow-x: auto;
    font-size: 12.5px;
    line-height: 1.45;
  }
  .code-card pre code {
    background: none;
    border: none;
    padding: 0;
    color: #e2e8f0;
  }

  /* Responsive Markdown Tables */
  .table-wrap {
    width: 100%;
    overflow-x: auto;
    margin: 10px 0;
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .bubble table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .bubble th, .bubble td {
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .bubble th {
    background: rgba(56, 189, 248, 0.1);
    color: #7dd3fc;
    font-weight: 600;
  }
  .bubble tr:last-child td { border-bottom: none; }

  .bubble a { color: var(--accent); text-decoration: underline; }

  /* Source Citations */
  .sources-wrap {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    width: 100%;
    padding-bottom: 6px;
    scrollbar-width: none;
  }
  .sources-wrap::-webkit-scrollbar { display: none; }
  .source-chip {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 6px 10px;
    font-size: 11.5px;
    color: var(--text);
    text-decoration: none;
    flex-shrink: 0;
    max-width: 210px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .source-chip .s-title {
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .source-chip .s-domain {
    font-size: 10px;
    color: var(--accent);
  }

  .msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 5px;
    font-size: 11px;
    color: var(--muted);
    padding-left: 4px;
  }
  .meta-btn {
    background: none;
    border: none;
    color: var(--muted);
    font-size: 11px;
    cursor: pointer;
    padding: 2px 4px;
  }
  .meta-btn:hover { color: var(--text); }

  /* Composer Area */
  .composer-area {
    background: rgba(19, 26, 42, 0.96);
    backdrop-filter: blur(12px);
    border-top: 1px solid var(--border);
    padding: 8px 10px calc(env(safe-area-inset-bottom, 0px) + 10px);
    z-index: 20;
  }
  body.kb-open .composer-area {
    padding-bottom: 8px;
  }
  .toolbar {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 7px;
    scrollbar-width: none;
  }
  .toolbar::-webkit-scrollbar { display: none; }
  .pill {
    background: var(--panel-2);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
    cursor: pointer;
    flex-shrink: 0;
  }
  .pill.active {
    background: rgba(56, 189, 248, 0.16);
    border-color: var(--accent);
    color: var(--accent);
  }
  .pill.deep.active {
    background: rgba(99, 102, 241, 0.2);
    border-color: #818cf8;
    color: #a5b4fc;
  }
  .pill.think-pill.active {
    background: rgba(245, 158, 11, 0.18);
    border-color: #f59e0b;
    color: #fbbf24;
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
  }
  .think-box {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 3px solid #f59e0b;
    border-radius: 9px;
    padding: 7px 11px;
    margin-bottom: 10px;
    font-size: 13px;
    color: #cbd5e1;
  }
  .think-box summary {
    cursor: pointer;
    font-weight: 600;
    color: #fbbf24;
    font-size: 12.5px;
    user-select: none;
    outline: none;
  }
  .think-content {
    margin-top: 7px;
    padding-top: 7px;
    border-top: 1px solid rgba(255,255,255,0.07);
    font-size: 12.5px;
    line-height: 1.45;
    color: #94a3b8;
    white-space: pre-wrap;
  }

  .input-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 8px 6px 8px;
  }
  .input-row:focus-within { border-color: var(--accent); }
  .attach-btn {
    width: 34px; height: 34px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--panel-2);
    color: var(--text);
    font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    flex-shrink: 0;
    transition: border-color 0.15s, background 0.15s;
  }
  .attach-btn:hover, .attach-btn:active {
    border-color: var(--accent);
    background: rgba(56, 189, 248, 0.15);
  }
  .attach-preview-bar {
    display: none;
    gap: 8px;
    overflow-x: auto;
    padding: 4px 2px 8px;
    scrollbar-width: none;
  }
  .attach-preview-bar.has-items { display: flex; }
  .attach-preview-bar::-webkit-scrollbar { display: none; }
  .attach-chip {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--panel-2);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 10px;
    padding: 5px 8px;
    font-size: 12px;
    color: var(--text);
    flex-shrink: 0;
    max-width: 220px;
  }
  .attach-chip img {
    width: 32px; height: 32px;
    border-radius: 6px;
    object-fit: cover;
    border: 1px solid var(--border);
  }
  .attach-chip .a-info {
    display: flex; flex-direction: column;
    min-width: 0;
  }
  .attach-chip .a-name {
    font-weight: 600;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .attach-chip .a-meta {
    font-size: 10.5px; color: var(--muted);
  }
  .attach-chip .a-remove {
    background: rgba(239, 68, 68, 0.18);
    color: #fca5a5;
    border: none;
    border-radius: 50%;
    width: 20px; height: 20px;
    font-size: 11px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .msg-attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 6px;
    justify-content: flex-end;
  }
  .msg-img-thumb {
    max-width: 180px;
    max-height: 150px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.2);
    object-fit: cover;
    display: block;
  }
  .msg-doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 8px;
    padding: 5px 9px;
    font-size: 12px;
    color: #f8fafc;
  }
  textarea {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-size: 16px;
    line-height: 1.4;
    max-height: 110px;
    resize: none;
    outline: none;
    padding: 4px 0;
    font-family: inherit;
  }
  textarea::placeholder { color: #64748b; }

  .send-btn {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff;
    font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    flex-shrink: 0;
  }
  .send-btn:disabled { opacity: 0.4; cursor: default; }

  /* Loading Indicator */
  .StatusStep {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--accent);
  }
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid rgba(56,189,248,0.25);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Modal for Phone QR / Connection Info */
  .modal-backdrop {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.7);
    backdrop-filter: blur(4px);
    display: none; align-items: center; justify-content: center;
    z-index: 100; padding: 16px;
  }
  .modal-backdrop.open { display: flex; }
  .modal {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    max-width: 360px; width: 100%;
    text-align: center;
  }
  .modal h3 { margin: 0 0 8px; font-size: 16px; }
  .modal p { font-size: 12.5px; color: var(--muted); margin: 6px 0 12px; line-height: 1.45; }
  .url-box {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 9px 12px;
    border-radius: 8px;
    font-family: monospace;
    font-size: 13px;
    color: var(--accent);
    margin-bottom: 12px;
    word-break: break-all;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <img src="/crest.jpg?v=2" alt="Fontes Family Crest" class="logo">
    <div class="brand-text">
      <h1>Mega FF AI</h1>
      <div class="status-line">
        <span id="status-dot" class="dot"></span>
        <span id="status-text">Checking server...</span>
      </div>
    </div>
  </div>
  <div class="header-actions">
    <button class="icon-btn" onclick="toggleModal(true)" title="Phone Link">📱 Link</button>
    <button class="icon-btn" onclick="clearChat()" title="New Chat">＋ New</button>
  </div>
</header>

<main id="chat-container">
  <div id="empty-state" class="empty-state">
    <img src="/crest.jpg?v=2" alt="Fontes Family Crest" class="hero-crest">
    <h2>Ask, Search & Summarize</h2>
    <p>Connected directly to your desktop <b>Mega FF AI</b> server with live web search, Vision, and PDF reading.</p>
    <div class="quick-grid">
      <div class="quick-card" onclick="usePreset('search', 'What are the top technology & AI news headlines today?')">
        <div class="q-title">🌐 Search Live Web</div>
        <div class="q-sub">Today's top tech & AI news</div>
      </div>
      <div class="quick-card" onclick="promptUrlSummarize()">
        <div class="q-title">📄 Summarize a URL</div>
        <div class="q-sub">Paste any article or webpage link</div>
      </div>
      <div class="quick-card" onclick="usePreset('deep', 'Research and compare the pros and cons of current local LLM quantization methods')">
        <div class="q-title">🔎 Deep Web Research</div>
        <div class="q-sub">Reads full top web pages</div>
      </div>
      <div class="quick-card" onclick="triggerFilePicker()">
        <div class="q-title">📎 Upload Image or PDF</div>
        <div class="q-sub">Analyze photos, charts, or PDFs</div>
      </div>
    </div>
  </div>
</main>

<div class="composer-area">
  <input type="file" id="file-input" accept="image/*,.pdf,.txt,.csv,.md,.json,.py,.js,.html,.log" multiple style="display:none" onchange="handleFileSelect(event)">
  <div class="toolbar">
    <button id="pill-think" class="pill think-pill" onclick="toggleThinkMode()">🧠 Think: OFF</button>
    <button class="pill" onclick="triggerFilePicker()">📎 Image / PDF</button>
    <button id="pill-auto" class="pill active" onclick="setSearchMode('auto')">✨ Auto Web</button>
    <button id="pill-code" class="pill" onclick="setSearchMode('code')">💻 Full Code</button>
    <button id="pill-search" class="pill" onclick="setSearchMode('search')">🌐 Always Search</button>
    <button id="pill-deep" class="pill deep" onclick="setSearchMode('deep')">🔎 Deep Read</button>
    <button id="pill-off" class="pill" onclick="setSearchMode('off')">⚡ LLM Only</button>
    <button class="pill" onclick="promptUrlSummarize()">🔗 Paste URL</button>
  </div>
  <div id="attachment-bar" class="attach-preview-bar"></div>
  <div class="input-row">
    <button class="attach-btn" onclick="triggerFilePicker()" title="Upload Image, Photo, or PDF">📎</button>
    <textarea id="prompt-input" rows="1" placeholder="Ask anything, attach 📎 image/PDF, or paste URL..." oninput="autoResize(this)"></textarea>
    <button id="send-btn" class="send-btn" onclick="sendMessage()">➤</button>
  </div>
</div>

<!-- Phone Connection Modal -->
<div id="phone-modal" class="modal-backdrop" onclick="if(event.target===this)toggleModal(false)">
  <div class="modal">
    <h3>📱 Open on Your Phone</h3>
    <p>On your home Wi-Fi, open your phone's browser (Safari / Chrome) and visit:</p>
    <div id="lan-url" class="url-box">http://...</div>
    <img id="qr-img" src="" alt="QR Code" style="width:170px;height:170px;border-radius:10px;background:#fff;padding:8px;margin-bottom:10px;">
    <p style="font-size:11.5px;">💡 Tip: Tap <b>Share → Add to Home Screen</b> on your phone for a full-screen app icon!</p>
    <button class="icon-btn" style="width:100%;justify-content:center;padding:10px;" onclick="toggleModal(false)">Done</button>
  </div>
</div>

<script>
  let messages = JSON.parse(localStorage.getItem("gemma4_mobile_chat") || "[]");
  let searchMode = localStorage.getItem("gemma4_search_mode") || "auto";
  let thinkMode = localStorage.getItem("gemma4_think_mode") === "true";
  let isBusy = false;
  let pendingAttachments = [];

  const chatEl = document.getElementById("chat-container");
  const emptyEl = document.getElementById("empty-state");
  const inputEl = document.getElementById("prompt-input");
  const sendBtn = document.getElementById("send-btn");
  const fileInputEl = document.getElementById("file-input");
  const attachBarEl = document.getElementById("attachment-bar");

  if (window.pdfjsLib) {
    window.pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }

  function triggerFilePicker() {
    if (isBusy) return;
    fileInputEl.value = "";
    fileInputEl.click();
  }

  function renderPendingAttachments() {
    if (!pendingAttachments.length) {
      attachBarEl.classList.remove("has-items");
      attachBarEl.innerHTML = "";
      return;
    }
    attachBarEl.classList.add("has-items");
    attachBarEl.innerHTML = pendingAttachments.map((att, idx) => {
      const preview = att.thumb
        ? `<img src="${att.thumb}" alt="preview">`
        : `<span style="font-size:18px">${att.kind === "pdf" ? "📄" : "📝"}</span>`;
      return `
        <div class="attach-chip">
          ${preview}
          <div class="a-info">
            <span class="a-name">${escapeHtml(att.name)}</span>
            <span class="a-meta">${escapeHtml(att.meta || "")}</span>
          </div>
          <button class="a-remove" onclick="removePendingAttachment(${idx})" title="Remove">✕</button>
        </div>
      `;
    }).join("");
  }

  function removePendingAttachment(idx) {
    pendingAttachments.splice(idx, 1);
    renderPendingAttachments();
  }

  function resizeDataUrl(dataUrl, maxDim, quality = 0.86) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          if (w >= h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
        }
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(img, 0, 0, w, h);
        const outUrl = canvas.toDataURL("image/jpeg", quality);
        resolve({ dataUrl: outUrl, width: w, height: h });
      };
      img.onerror = reject;
      img.src = dataUrl;
    });
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsDataURL(file);
    });
  }

  function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsArrayBuffer(file);
    });
  }

  function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsText(file);
    });
  }

  async function processImageFile(file) {
    const rawUrl = await readFileAsDataUrl(file);
    const full = await resizeDataUrl(rawUrl, 1280, 0.86);
    const thumb = await resizeDataUrl(rawUrl, 180, 0.72);
    const b64 = full.dataUrl.split(",")[1];
    pendingAttachments.push({
      kind: "image",
      name: file.name || "image.jpg",
      meta: `${full.width}×${full.height} · Vision`,
      b64: b64,
      thumb: thumb.dataUrl
    });
    renderPendingAttachments();
  }

  async function processPdfFile(file) {
    if (!window.pdfjsLib) {
      alert("PDF reader library is still loading. Please check your connection and try again.");
      return;
    }
    const buf = await readFileAsArrayBuffer(file);
    const pdf = await window.pdfjsLib.getDocument({ data: buf }).promise;
    const totalPages = pdf.numPages;
    const maxPages = Math.min(totalPages, 25);
    const pageTexts = [];
    const pageImages = [];
    let thumbUrl = "";
    let totalChars = 0;

    for (let p = 1; p <= maxPages; p++) {
      const page = await pdf.getPage(p);
      const tc = await page.getTextContent();
      const strings = tc.items.map(item => item.str);
      const pText = strings.join(" ").replace(/\s+/g, " ").trim();
      if (pText) {
        pageTexts.push(`[Page ${p}/${totalPages}]\n${pText}`);
        totalChars += pText.length;
      }

      if (p === 1 || (p <= 2 && totalChars < 200)) {
        const vp1 = page.getViewport({ scale: 1.0 });
        const scale = Math.min(1.6, 1280 / Math.max(vp1.width, vp1.height, 1));
        const vp = page.getViewport({ scale });
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(vp.width);
        canvas.height = Math.round(vp.height);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        await page.render({ canvasContext: ctx, viewport: vp }).promise;
        const pageDataUrl = canvas.toDataURL("image/jpeg", 0.85);
        pageImages.push(pageDataUrl.split(",")[1]);
        if (p === 1) {
          const th = await resizeDataUrl(pageDataUrl, 160, 0.7);
          thumbUrl = th.dataUrl;
        }
      }
    }

    const joinedText = pageTexts.join("\n\n");
    const isScanned = totalChars < 120;
    pendingAttachments.push({
      kind: "pdf",
      name: file.name || "document.pdf",
      meta: isScanned ? `${totalPages}p · Scanned Vision` : `${totalPages} page${totalPages > 1 ? "s" : ""} · PDF`,
      text: joinedText,
      pagesImages: pageImages,
      thumb: thumbUrl
    });
    renderPendingAttachments();
  }

  async function processTextFile(file) {
    const txt = await readFileAsText(file);
    pendingAttachments.push({
      kind: "doc",
      name: file.name || "file.txt",
      meta: `${Math.max(1, Math.round(txt.length / 1024))} KB · Text`,
      text: txt.slice(0, 12000)
    });
    renderPendingAttachments();
  }

  async function handleFilesList(fileList) {
    for (const file of Array.from(fileList || [])) {
      try {
        const nameLower = (file.name || "").toLowerCase();
        if (file.type.startsWith("image/") || /\.(jpg|jpeg|png|webp|gif|bmp)$/i.test(nameLower)) {
          await processImageFile(file);
        } else if (file.type === "application/pdf" || nameLower.endsWith(".pdf")) {
          await processPdfFile(file);
        } else {
          await processTextFile(file);
        }
      } catch (err) {
        alert(`Could not import ${file.name}: ${err.message || err}`);
      }
    }
    inputEl.focus();
  }

  async function handleFileSelect(event) {
    if (event.target.files && event.target.files.length) {
      await handleFilesList(event.target.files);
    }
  }

  window.addEventListener("paste", async (e) => {
    const items = (e.clipboardData || {}).items || [];
    const files = [];
    for (const item of items) {
      if (item.kind === "file") {
        const f = item.getAsFile();
        if (f) files.push(f);
      }
    }
    if (files.length) {
      e.preventDefault();
      await handleFilesList(files);
    }
  });

  window.addEventListener("dragover", (e) => { e.preventDefault(); });
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      await handleFilesList(e.dataTransfer.files);
    }
  });

  function syncThinkUI() {
    const btn = document.getElementById("pill-think");
    if (!btn) return;
    btn.classList.toggle("active", thinkMode);
    btn.textContent = thinkMode ? "🧠 Think: ON" : "🧠 Think: OFF";
  }

  function toggleThinkMode() {
    thinkMode = !thinkMode;
    localStorage.setItem("gemma4_think_mode", String(thinkMode));
    syncThinkUI();
  }

  function setSearchMode(mode) {
    searchMode = mode;
    localStorage.setItem("gemma4_search_mode", mode);
    ["auto", "code", "search", "deep", "off"].forEach(m => {
      const el = document.getElementById("pill-" + m);
      if (el) el.classList.toggle("active", m === mode);
    });
  }

  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 700) {
      e.preventDefault();
      sendMessage();
    }
  });

  function usePreset(mode, text) {
    setSearchMode(mode);
    inputEl.value = text;
    autoResize(inputEl);
    inputEl.focus();
  }

  function promptUrlSummarize() {
    let url = prompt("Paste the webpage URL to read & summarize:");
    if (url && url.trim()) {
      url = url.trim();
      if (!/^https?:\/\//i.test(url)) {
        url = "https://" + url;
      }
      inputEl.value = `Summarize the key points of ${url}`;
      autoResize(inputEl);
      sendMessage();
    }
  }

  function escapeHtml(str) {
    return (str || "").replace(/[&<>"']/g, m => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[m]));
  }

  function cleanLatex(str) {
    if (!str) return "";
    let s = str;
    const map = [
      [/\$?\\(?:longrightarrow|rightarrow|to)\$?/g, " → "],
      [/\$?\\(?:longleftarrow|leftarrow|gets)\$?/g, " ← "],
      [/\$?\\(?:leftrightarrow|iff)\$?/g, " ↔ "],
      [/\$?\\(?:Longrightarrow|Rightarrow|implies)\$?/g, " ⇒ "],
      [/\$?\\(?:Leftarrow)\$?/g, " ⇐ "],
      [/\$?\\(?:uparrow)\$?/g, " ↑ "],
      [/\$?\\(?:downarrow)\$?/g, " ↓ "],
      [/\$?\\times\$?/g, " × "],
      [/\$?\\div\$?/g, " ÷ "],
      [/\$?\\cdot\$?/g, " · "],
      [/\$?\\pm\$?/g, " ± "],
      [/\$?\\mp\$?/g, " ∓ "],
      [/\$?\\(?:approx|simeq|sim)\$?/g, " ≈ "],
      [/\$?\\(?:neq|ne)\$?/g, " ≠ "],
      [/\$?\\(?:leq|le)\$?/g, " ≤ "],
      [/\$?\\(?:geq|ge)\$?/g, " ≥ "],
      [/\$?\\infty\$?/g, "∞"],
      [/\$?\\(?:circ|degree)\$?/g, "°"],
      [/\^\{\\circ\}/g, "°"],
      [/\$?\\alpha\$?/g, "α"],
      [/\$?\\beta\$?/g, "β"],
      [/\$?\\gamma\$?/g, "γ"],
      [/\$?\\delta\$?/g, "δ"],
      [/\$?\\Delta\$?/g, "Δ"],
      [/\$?\\theta\$?/g, "θ"],
      [/\$?\\lambda\$?/g, "λ"],
      [/\$?\\mu\$?/g, "μ"],
      [/\$?\\pi\$?/g, "π"],
      [/\$?\\sigma\$?/g, "σ"],
      [/\$?\\Sigma\$?/g, "Σ"],
      [/\$?\\omega\$?/g, "ω"],
      [/\$?\\Omega\$?/g, "Ω"],
      [/\\(?:text|mathrm|mathbf|textit|textbf)\{([^}]*)\}/g, "$1"],
      [/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1 / $2)"],
      [/\\sqrt\{([^}]*)\}/g, "√($1)"],
      [/\\%/g, "%"],
    ];
    for (const [re, rep] of map) {
      s = s.replace(re, rep);
    }
    s = s.replace(/\$([^$\n]{1,60})\$/g, (m, inner) => {
      if (/^\d/.test(inner.trim()) && !/[\\^_=<>±×÷→←≈≤≥]/.test(inner)) return m;
      return inner;
    });
    return s.replace(/ {2,}/g, " ");
  }

  function formatInline(str) {
    let s = escapeHtml(cleanLatex(str));
    s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }

  function universalCopy(text, btn) {
    const showFeedback = (ok) => {
      if (!btn) return;
      const old = btn.dataset.origText || btn.textContent;
      btn.dataset.origText = old;
      btn.textContent = ok ? "✓ Copied!" : "⚠️ Copy failed";
      setTimeout(() => { btn.textContent = old; }, 1400);
    };

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(() => showFeedback(true)).catch(() => fallbackCopy(text, showFeedback));
    } else {
      fallbackCopy(text, showFeedback);
    }
  }

  function fallbackCopy(text, cb) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.top = "-9999px";
      ta.style.left = "-9999px";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      ta.setSelectionRange(0, text.length);
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      cb(ok);
    } catch (e) {
      cb(false);
    }
  }

  function copyRawCode(btn) {
    const pre = btn.closest(".code-card").querySelector("code");
    if (pre) {
      universalCopy(pre.innerText, btn);
    }
  }

  function renderMarkdown(md) {
    if (!md) return "";
    let norm = md.replace(/\r\n/g, "\n");
    let thinkHtml = "";
    norm = norm.replace(/<think>([\s\S]*?)(?:<\/think>|$)/i, (_, thought) => {
      const cleanThought = thought.trim();
      if (cleanThought) {
        thinkHtml = `<details class="think-box"><summary>🧠 Thought Process (tap to view)</summary><div class="think-content">${escapeHtml(cleanThought)}</div></details>`;
      }
      return "";
    }).trim();

    const fenceCount = (norm.match(/^[ \t]*```/gm) || []).length;
    if (fenceCount % 2 === 1) {
      norm += "\n```";
    }
    const codeBlocks = [];
    let text = norm.replace(/^[ \t]*```([^\n`]*)\n([\s\S]*?)^[ \t]*```/gm, (_, lang, code) => {
      const idx = codeBlocks.length;
      const label = (lang.trim() || "code").toUpperCase();
      codeBlocks.push(
        `<div class="code-card"><div class="code-header"><span>${escapeHtml(label)}</span><button class="copy-code-btn" onclick="copyRawCode(this)">📋 Copy</button></div><pre><code>${escapeHtml(code.trim())}</code></pre></div>`
      );
      return `\n@@CODEBLOCK_${idx}@@\n`;
    });

    const lines = text.split("\n");
    const out = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();

      if (!trimmed) {
        i++;
        continue;
      }

      const cbMatch = trimmed.match(/^@@CODEBLOCK_(\d+)@@$/);
      if (cbMatch) {
        out.push(codeBlocks[Number(cbMatch[1])]);
        i++;
        continue;
      }

      if (/^([-*_])\1{2,}$/.test(trimmed)) {
        out.push("<hr>");
        i++;
        continue;
      }

      const hMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (hMatch) {
        const level = hMatch[1].length;
        out.push(`<h${level}>${formatInline(hMatch[2])}</h${level}>`);
        i++;
        continue;
      }

      if (
        trimmed.includes("|") &&
        i + 1 < lines.length &&
        /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(lines[i + 1])
      ) {
        const parseRow = (r) =>
          r.trim().replace(/^\||\|$/g, "").split("|").map(c => formatInline(c.trim()));
        const headers = parseRow(line);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().includes("|") && !/^\s*$/.test(lines[i])) {
          if (/^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(lines[i])) {
            i++;
            continue;
          }
          rows.push(parseRow(lines[i]));
          i++;
        }
        const thead = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>`;
        const tbody = `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
        out.push(`<div class="table-wrap"><table>${thead}${tbody}</table></div>`);
        continue;
      }

      if (trimmed.startsWith(">")) {
        const qLines = [];
        while (i < lines.length && lines[i].trim().startsWith(">")) {
          qLines.push(formatInline(lines[i].trim().replace(/^>\s?/, "")));
          i++;
        }
        out.push(`<blockquote>${qLines.join("<br>")}</blockquote>`);
        continue;
      }

      if (/^\s*[-*•]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
          items.push(`<li>${formatInline(lines[i].replace(/^\s*[-*•]\s+/, ""))}</li>`);
          i++;
        }
        out.push(`<ul>${items.join("")}</ul>`);
        continue;
      }

      if (/^\s*\d+[.)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          items.push(`<li>${formatInline(lines[i].replace(/^\s*\d+[.)]\s+/, ""))}</li>`);
          i++;
        }
        out.push(`<ol>${items.join("")}</ol>`);
        continue;
      }

      const pLines = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !lines[i].trim().match(/^@@CODEBLOCK_(\d+)@@$/) &&
        !lines[i].trim().match(/^(#{1,4})\s+/) &&
        !/^([-*_])\1{2,}$/.test(lines[i].trim()) &&
        !/^\s*[-*•]\s+/.test(lines[i]) &&
        !/^\s*\d+[.)]\s+/.test(lines[i]) &&
        !lines[i].trim().startsWith(">") &&
        !(lines[i].trim().includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(lines[i + 1]))
      ) {
        pLines.push(formatInline(lines[i]));
        i++;
      }
      if (pLines.length) {
        out.push(`<p>${pLines.join("<br>")}</p>`);
      } else {
        i++;
      }
    }

    return thinkHtml + out.join("");
  }

  function getDomain(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
  }

  function renderAllMessages() {
    Array.from(chatEl.querySelectorAll(".msg-row")).forEach(n => n.remove());
    emptyEl.style.display = messages.length ? "none" : "block";
    messages.forEach((m, idx) => appendMessageDOM(m, idx));
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function appendMessageDOM(m, idx) {
    emptyEl.style.display = "none";
    const row = document.createElement("div");
    row.className = "msg-row " + m.role;

    if (m.role === "assistant" && m.sources && m.sources.length) {
      const sWrap = document.createElement("div");
      sWrap.className = "sources-wrap";
      m.sources.forEach(s => {
        const a = document.createElement("a");
        a.className = "source-chip";
        a.href = s.url;
        a.target = "_blank";
        a.rel = "noopener";
        a.innerHTML = `<span class="s-title">${escapeHtml(s.title || s.url)}</span><span class="s-domain">🌐 ${escapeHtml(getDomain(s.url))}</span>`;
        sWrap.appendChild(a);
      });
      row.appendChild(sWrap);
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (m.role === "assistant") {
      bubble.innerHTML = renderMarkdown(m.content);
    } else {
      let attHtml = "";
      if (m.attachments && m.attachments.length) {
        const itemsHtml = m.attachments.map(a => {
          if (a.kind === "image" && a.thumb) {
            return `<img src="${a.thumb}" class="msg-img-thumb" alt="${escapeHtml(a.name)}">`;
          }
          const icon = a.kind === "pdf" ? "📄" : "📝";
          const thumbImg = a.thumb ? `<img src="${a.thumb}" style="width:26px;height:32px;object-fit:cover;border-radius:4px;">` : `<span>${icon}</span>`;
          return `<div class="msg-doc-badge">${thumbImg}<span><b>${escapeHtml(a.name)}</b> <span style="opacity:0.75;font-size:11px">(${escapeHtml(a.meta || "")})</span></span></div>`;
        }).join("");
        attHtml = `<div class="msg-attachments">${itemsHtml}</div>`;
      }
      bubble.innerHTML = attHtml + escapeHtml(m.content).replace(/\n/g, "<br>");
    }
    row.appendChild(bubble);

    if (m.role === "assistant") {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      const stats = m.stats ? `⚡ ${m.stats}` : "Mega FF AI";
      meta.innerHTML = `
        <span>${escapeHtml(stats)}</span>
        <button class="meta-btn" onclick="copyMsg(${idx}, this)">📋 Copy</button>
        <button class="meta-btn" onclick="speakMsg(${idx})">🔊 Listen</button>
      `;
      row.appendChild(meta);
    }

    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function stripThinkTags(str) {
    return (str || "").replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "").trim();
  }

  function copyMsg(idx, btn) {
    if (messages[idx]) {
      universalCopy(stripThinkTags(messages[idx].content), btn);
    }
  }

  function speakMsg(idx) {
    if (!("speechSynthesis" in window) || !messages[idx]) return;
    window.speechSynthesis.cancel();
    const clean = stripThinkTags(messages[idx].content).replace(/[#*`_~]/g, "");
    const utter = new SpeechSynthesisUtterance(clean);
    window.speechSynthesis.speak(utter);
  }

  function clearChat() {
    if (isBusy) return;
    messages = [];
    localStorage.removeItem("gemma4_mobile_chat");
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    renderAllMessages();
  }

  function saveMessagesToStorage() {
    try {
      localStorage.setItem("gemma4_mobile_chat", JSON.stringify(messages.slice(-20)));
    } catch (e) {
      try {
        const slim = messages.slice(-12).map(m => ({ role: m.role, content: m.content, sources: m.sources, stats: m.stats }));
        localStorage.setItem("gemma4_mobile_chat", JSON.stringify(slim));
      } catch {}
    }
  }

  async function sendMessage() {
    let text = inputEl.value.trim();
    if ((!text && !pendingAttachments.length) || isBusy) return;

    const currentAttachments = pendingAttachments.slice();
    pendingAttachments = [];
    renderPendingAttachments();

    if (!text && currentAttachments.length) {
      const hasPdfOrDoc = currentAttachments.some(a => a.kind === "pdf" || a.kind === "doc");
      text = hasPdfOrDoc
        ? "Summarize and explain the key points of this attached document."
        : "Analyze and describe this image in detail.";
    }

    isBusy = true;
    sendBtn.disabled = true;
    inputEl.value = "";
    autoResize(inputEl);

    const uiAttachments = currentAttachments.map(a => ({
      kind: a.kind,
      name: a.name,
      meta: a.meta,
      thumb: a.thumb || ""
    }));

    const userMsg = { role: "user", content: text, attachments: uiAttachments };
    messages.push(userMsg);
    saveMessagesToStorage();
    appendMessageDOM(userMsg, messages.length - 1);

    const payloadImages = [];
    const payloadDocs = [];
    for (const a of currentAttachments) {
      if (a.kind === "image" && a.b64) {
        payloadImages.push(a.b64);
      } else if (a.kind === "pdf") {
        if (a.text) {
          payloadDocs.push({ name: a.name, text: a.text, meta: a.meta });
        }
        if (a.pagesImages && a.pagesImages.length) {
          payloadImages.push(...a.pagesImages.slice(0, 2));
        }
      } else if (a.kind === "doc" && a.text) {
        payloadDocs.push({ name: a.name, text: a.text, meta: a.meta });
      }
    }

    const loadingRow = document.createElement("div");
    loadingRow.className = "msg-row assistant";
    const stepText = currentAttachments.length
      ? "Analyzing attached file(s) on RTX 4090..."
      : (thinkMode ? "Thinking deeply & working..." : "Working (searching / queuing on GPU)...");
    loadingRow.innerHTML = `<div class="bubble"><div class="StatusStep"><div class="spinner"></div><span id="step-label">${stepText}</span></div></div>`;
    chatEl.appendChild(loadingRow);
    chatEl.scrollTop = chatEl.scrollHeight;

    try {
      const res = await fetch("/api/mobile_chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages.slice(-14).map(m => ({ role: m.role, content: m.content })),
          search_mode: searchMode,
          think_mode: thinkMode,
          images: payloadImages,
          documents: payloadDocs
        })
      });
      const data = await res.json();
      loadingRow.remove();

      if (!res.ok || data.error) {
        throw new Error(data.error || "Server error");
      }

      const assistantMsg = {
        role: "assistant",
        content: data.reply,
        sources: data.sources || [],
        stats: data.stats || ""
      };
      messages.push(assistantMsg);
      saveMessagesToStorage();
      appendMessageDOM(assistantMsg, messages.length - 1);
    } catch (err) {
      loadingRow.remove();
      const errMsg = {
        role: "assistant",
        content: `⚠️ **Error:** ${err.message}`,
        sources: [],
        stats: "Failed"
      };
      messages.push(errMsg);
      appendMessageDOM(errMsg, messages.length - 1);
    } finally {
      isBusy = false;
      sendBtn.disabled = false;
    }
  }

  async function checkStatus() {
    const dot = document.getElementById("status-dot");
    const txt = document.getElementById("status-text");
    try {
      const r = await fetch("/api/status");
      const d = await r.json();
      if (d.status === "online") {
        dot.className = "dot online";
        const vram = d.gpu ? ` · ${d.gpu.allocated_gb}/${d.gpu.total_gb} GB VRAM` : "";
        txt.textContent = `Online (RTX 4090${vram})`;
      } else {
        dot.className = "dot";
        txt.textContent = d.status || "Connecting...";
      }
      if (d.lan_url) {
        document.getElementById("lan-url").textContent = d.lan_url;
        document.getElementById("qr-img").src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(d.lan_url)}`;
      }
    } catch {
      dot.className = "dot offline";
      txt.textContent = "Server unreachable";
    }
  }

  function toggleModal(open) {
    document.getElementById("phone-modal").classList.toggle("open", open);
  }

  function syncMobileViewport() {
    const vv = window.visualViewport;
    if (!vv) return;
    document.body.style.height = vv.height + "px";
    document.body.style.top = vv.offsetTop + "px";
    const kbOpen = vv.height < window.innerHeight * 0.82;
    document.body.classList.toggle("kb-open", kbOpen);
    if (kbOpen) {
      window.scrollTo(0, 0);
      setTimeout(() => { chatEl.scrollTop = chatEl.scrollHeight; }, 30);
    }
  }

  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", syncMobileViewport);
    window.visualViewport.addEventListener("scroll", syncMobileViewport);
  }
  window.addEventListener("resize", syncMobileViewport);
  inputEl.addEventListener("focus", () => setTimeout(syncMobileViewport, 80));
  inputEl.addEventListener("blur", () => setTimeout(syncMobileViewport, 80));

  syncMobileViewport();
  syncThinkUI();
  setSearchMode(searchMode);
  renderAllMessages();
  checkStatus();
  setInterval(checkStatus, 15000);
</script>
</body>
</html>
"""


# ==============================================================================
# Smart Search Trigger & Prompt Augmentation
# ==============================================================================
SEARCH_HINT_PATTERNS = re.compile(
    r"\b(search|google|look up|lookup|latest|today|current|news|weather|price|stock|"
    r"score|who is|what happened|2025|2026|recent|update|release|website|url|link)\b",
    re.IGNORECASE,
)


FOLLOWUP_PATTERNS = re.compile(
    r"^\s*(and\b|what about|how about|why did|can you|elaborate|explain more|tell me more|"
    r"which one|the first|the second|the third|summarize it|in spanish|in portuguese|shorter|longer)\b"
    r"|\b(this article|that page|those deals|that link|above|previous|earlier)\b",
    re.IGNORECASE,
)


def is_followup_question(user_text: str, has_history: bool) -> bool:
    if not has_history:
        return False
    if FOLLOWUP_PATTERNS.search(user_text):
        return True
    if len(user_text.split()) <= 9 and re.search(r"\b(it|that|this|they|them|those|these|he|she)\b", user_text, re.I):
        return True
    return False


CODING_PROMPT_PATTERNS = re.compile(
    r"\b(code|coding|script|function|class|method|python|javascript|typescript|html|css|sql|"
    r"bash|shell|powershell|react|fastapi|flask|pandas|numpy|pytorch|regex|json|api|"
    r"implement|refactor|debug|fix this|syntax|compile|algorithm|program|snippet)\b"
    r"|```|\bdef\s+\w+|\bimport\s+\w+",
    re.IGNORECASE,
)


def is_coding_request(user_text: str, search_mode: str = "auto") -> bool:
    if search_mode == "code":
        return True
    return bool(CODING_PROMPT_PATTERNS.search(user_text))


def should_auto_search(user_text: str, has_history: bool = False) -> bool:
    if is_followup_question(user_text, has_history):
        return False
    if is_coding_request(user_text):
        return False
    if SEARCH_HINT_PATTERNS.search(user_text):
        return True
    if user_text.strip().endswith("?") and len(user_text) < 220:
        if not re.search(r"\b(write a|rewrite|translate|fix this code|python|function)\b", user_text, re.I):
            return True
    return False


def build_augmented_messages(
    raw_messages: list,
    search_mode: str,
    think_mode: bool = False,
    images: list = None,
    documents: list = None,
):
    if not raw_messages:
        return [], []

    images = images or []
    documents = documents or []
    has_attachments = bool(images or documents)

    last_user_msg = raw_messages[-1]["content"]
    urls = extract_urls(last_user_msg)
    sources = []
    context_blocks = []

    # 0. Uploaded PDF / Text Documents
    for doc in documents[:3]:
        d_name = doc.get("name") or "Attached Document"
        d_text = (doc.get("text") or "").strip()
        if d_text:
            context_blocks.append(
                f"[Uploaded Document: {d_name}]\n{d_text[:MAX_TOTAL_CONTEXT_CHARS]}"
            )

    # 1. Direct URL Scraping / Summarization
    for u in urls[:2]:
        page = fetch_page_text(u, max_chars=MAX_SCRAPE_CHARS_PER_PAGE)
        if page.get("text"):
            sources.append({"title": page["title"], "url": u})
            context_blocks.append(
                f"[Fetched Webpage Content for {u} — Title: {page['title']}]\n{page['text']}"
            )
        elif page.get("error"):
            context_blocks.append(
                f"[Webpage {u} direct fetch note: {page['error']}]"
            )
        parsed_u = urllib.parse.urlparse(u)
        path_words = re.sub(r"[-_/]+", " ", parsed_u.path).strip()
        companion_q = f"{parsed_u.netloc} {path_words}".strip()
        for h in search_web(companion_q, max_results=3):
            if h["url"] != u:
                sources.append({"title": h["title"], "url": h["url"]})
            context_blocks.append(
                f"[Related Search Result: {h['title']} ({h['url']})]\nSnippet: {h['snippet']}"
            )

    # 2. Live Web Search
    has_history = len(raw_messages) > 1
    do_search = not urls and (
        search_mode in ("search", "deep")
        or (search_mode == "auto" and not has_attachments and should_auto_search(last_user_msg, has_history=has_history))
    )

    if do_search:
        raw_q = last_user_msg.strip()
        if has_history and len(raw_q.split()) <= 6:
            prev_user = next((m["content"] for m in reversed(raw_messages[:-1]) if m.get("role") == "user"), "")
            if prev_user:
                raw_q = f"{prev_user[:80]} {raw_q}"
        query = re.sub(r"^(search( the web)?( for)?|look up|tell me about)\s+", "", raw_q, flags=re.I)
        query = query[:160]
        hits = search_web(query, max_results=6)
        scraped_count = 0
        max_scrapes = 2 if search_mode == "deep" else 1
        for idx, h in enumerate(hits, 1):
            sources.append({"title": h["title"], "url": h["url"]})
            block = f"[{idx}] {h['title']}\nURL: {h['url']}\nSnippet: {h['snippet']}"
            if scraped_count < max_scrapes and "news.google.com" not in h["url"]:
                page = fetch_page_text(h["url"], max_chars=1800, use_jina_fallback=False)
                if page.get("text"):
                    block += f"\nExtracted Page Content: {page['text']}"
                    scraped_count += 1
            context_blocks.append(block)

    trimmed_history = []
    char_budget = MAX_HISTORY_CHARS
    for m in reversed(raw_messages[:-1]):
        raw_c = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", m.get("content") or "", flags=re.I).strip()
        c = raw_c[:3000]
        if char_budget - len(c) < 0:
            break
        char_budget -= len(c)
        trimmed_history.insert(0, {"role": m["role"], "content": c})

    today_str = time.strftime("%B %d, %Y")
    if context_blocks:
        joined_context = "\n\n".join(context_blocks)[:MAX_TOTAL_CONTEXT_CHARS]
        LAST_WEB_CONTEXT["context"] = joined_context
        LAST_WEB_CONTEXT["sources"] = sources
        LAST_WEB_CONTEXT["timestamp"] = time.time()

        cleaned_user_req = last_user_msg
        for u in urls:
            cleaned_user_req = cleaned_user_req.replace(u, f"the fetched webpage ({u}) provided above")

        augmented_user_prompt = (
            f"Current Date: {today_str}\n"
            f"SYSTEM INSTRUCTION: Our web gateway has ALREADY fetched the live webpage/document text and search results for you below. "
            f"Do NOT say that you lack internet access or cannot open links/files. "
            f"Answer the user's request directly using the provided CONTEXT below.\n\n"
            f"=== LIVE CONTEXT ===\n{joined_context}\n====================\n\n"
            f"User Request: {cleaned_user_req}"
        )
    elif has_history and LAST_WEB_CONTEXT["context"] and (time.time() - LAST_WEB_CONTEXT["timestamp"] < 1800):
        reused_ctx = LAST_WEB_CONTEXT["context"][:4500]
        augmented_user_prompt = (
            f"Current Date: {today_str}\n"
            f"Reference Context from earlier in this conversation (if relevant to the follow-up):\n"
            f"=== PRIOR CONTEXT ===\n{reused_ctx}\n=====================\n\n"
            f"User Follow-up Request: {last_user_msg}"
        )
    else:
        augmented_user_prompt = last_user_msg

    if is_coding_request(last_user_msg, search_mode):
        augmented_user_prompt += (
            "\n\n[CODING OUTPUT FORMAT REQUIREMENT: Provide the COMPLETE, self-contained, runnable code "
            "inside a single fenced Markdown code block (```language ... ```) with all imports included. "
            "Do NOT scatter or embed code fragments inside plain text paragraphs, and NEVER omit sections "
            "with placeholders like '# ... rest of code'. Put any brief notes after the code block.]"
        )

    if think_mode:
        augmented_user_prompt += (
            "\n\n[DEEP THINKING MODE ENABLED: First, think through the problem carefully step-by-step "
            "(analyzing edge cases, logic, math, or architecture) inside <think>...</think> tags. "
            "Keep your thinking concise (under 250 words), close with </think>, and then write your "
            "complete, well-formatted final response.]"
        )

    augmented_user_prompt += (
        "\n\n[FORMATTING NOTE: Use clean Unicode symbols (such as →, ←, ×, ÷, ≈, ≠, ≤, ≥, ±, °) "
        "instead of raw LaTeX math syntax like $\\rightarrow$ or \\times.]"
    )

    user_msg_dict = {"role": "user", "content": augmented_user_prompt}
    if images:
        user_msg_dict["images"] = images

    final_messages = trimmed_history + [user_msg_dict]
    return final_messages, sources


# ==============================================================================
# Threaded HTTP Server
# ==============================================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class MobileUIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[MobileUI] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            body = MOBILE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/crest.jpg"):
            crest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crest.jpg")
            if os.path.exists(crest_path):
                with open(crest_path, "rb") as f:
                    img_bytes = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(img_bytes)
                return

        if self.path == "/manifest.json":
            manifest = {
                "name": "Mega FF AI",
                "short_name": "Mega FF AI",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0b0f19",
                "theme_color": "#0b0f19",
                "icons": [
                    {
                        "src": "/crest.jpg?v=2",
                        "sizes": "512x512",
                        "type": "image/jpeg",
                    }
                ],
            }
            self._send_json(manifest)
            return

        if self.path == "/api/status":
            lan_ip = get_lan_ip()
            lan_url = f"http://{lan_ip}:{UI_PORT}"
            try:
                r = requests.get(f"{GEMMA_API_BASE}/health", timeout=3)
                data = r.json()
                data["lan_url"] = lan_url
                self._send_json(data)
            except Exception as e:
                self._send_json({"status": "offline", "error": str(e), "lan_url": lan_url}, status=200)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/mobile_chat":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                req_data = json.loads(raw_body)

                raw_messages = req_data.get("messages", [])
                search_mode = req_data.get("search_mode", "auto")
                think_mode = bool(req_data.get("think_mode", False))
                images = req_data.get("images") or []
                documents = req_data.get("documents") or []

                t0 = time.time()
                final_messages, sources = build_augmented_messages(
                    raw_messages,
                    search_mode,
                    think_mode=think_mode,
                    images=images,
                    documents=documents,
                )
                search_elapsed = time.time() - t0

                t_llm = time.time()
                resp = requests.post(
                    f"{GEMMA_API_BASE}/v1/chat/completions",
                    json={
                        "model": "gemma4:31b",
                        "messages": final_messages,
                        "temperature": 0.0,
                        "max_tokens": 2048,
                        "stream": False,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                llm_data = resp.json()
                llm_elapsed = time.time() - t_llm

                reply = (
                    llm_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                usage = llm_data.get("usage", {})
                comp_tokens = usage.get("completion_tokens", 0)
                tok_s = (comp_tokens / llm_elapsed) if llm_elapsed > 0 else 0

                stats_parts = []
                if sources:
                    stats_parts.append(f"{len(sources)} web sources ({search_elapsed:.1f}s)")
                if comp_tokens:
                    stats_parts.append(f"{comp_tokens} tok in {llm_elapsed:.1f}s ({tok_s:.1f} tok/s)")

                self._send_json({
                    "reply": reply,
                    "sources": sources,
                    "stats": " · ".join(stats_parts) if stats_parts else f"{llm_elapsed:.1f}s",
                })
            except Exception as e:
                print(f"[MobileUI Error] {e}")
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_error(404, "Not Found")


if __name__ == "__main__":
    lan_ip = get_lan_ip()
    print("=" * 68)
    print("📱 Mega FF AI Mobile Web UI & Search Gateway")
    print(f"🔗 Connected to Mega FF AI backend : {GEMMA_API_BASE}")
    print(f"💻 Desktop URL                     : http://localhost:{UI_PORT}")
    print(f"📲 Phone Wi-Fi URL                 : http://{lan_ip}:{UI_PORT}")
    print("=" * 68)
    server = ThreadedHTTPServer((UI_HOST, UI_PORT), MobileUIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Mobile UI...")
        server.server_close()
