#!/usr/bin/env python3
"""
================================================================================
🔍 GEMMA 4 AGENTIC WEB SEARCH & EMAIL BRIEFING ENGINE
================================================================================
Uses local Gemma 4 31B (with MTP Speculative Decoding on port 11434) to:
1. Auto-start gemma4_server.py via /home/erfontes/miniconda3/envs/vel/bin/python
   if not already running (and cleanly release VRAM afterwards if started by us).
2. Break a research topic down into targeted web & news search queries.
3. Fetch live web & news results via DuckDuckGo (ddgs) and scrape article text.
4. Synthesize an executive HTML briefing with clickable source citations.
5. Dispatch the report via SMTP using credentials in /home/erfontes/research/trader/.env.
================================================================================
"""

import os
import sys
import re
import json
import time
import signal
import smtplib
import argparse
import subprocess
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# ==============================================================================
# 1. Configuration & Environment Loading
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env_file():
    """Load key-value pairs from .env or daily_pipeline.env in BASE_DIR."""
    for env_name in [".env", "daily_pipeline.env"]:
        env_path = os.path.join(BASE_DIR, env_name)
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(
                            k.strip(), v.strip().strip('"').strip("'")
                        )


load_env_file()

GEMMA4_URL = os.getenv("GEMMA4_SERVER_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("GEMMA4_MODEL_NAME", "gemma4:31b")

DEFAULT_TOPICS = os.getenv(
    "SEARCH_AGENT_TOPICS",
    "Latest market-moving catalysts, AI hardware & LLM breakthroughs, and macro news for GOOGL, NVDA, COST, TSLA, AMD",
)


# ==============================================================================
# 2. Gemma 4 Server Connection Check (Uses Existing Running Server Only)
# ==============================================================================
def ensure_gemma_server_online(url: str = GEMMA4_URL, timeout: float = 5.0) -> None:
    """Checks that the existing Gemma 4 server is online without ever spawning a second process."""
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        if r.status_code == 200 and r.json().get("status") == "online":
            gpu = r.json().get("gpu", {})
            print(
                f"🚀 Connected to running Gemma 4 MTP server at {url} "
                f"(VRAM: {gpu.get('allocated_gb', '?')}/{gpu.get('total_gb', '?')} GB)"
            )
            return
    except Exception as e:
        raise RuntimeError(
            f"Gemma 4 server is not reachable at {url}/health ({e}). "
            f"Please ensure gemma4_server.py is running."
        )
    raise RuntimeError(f"Gemma 4 server at {url} is not ready.")


# ==============================================================================
# 3. Gemma 4 Inference Helper
# ==============================================================================
def ask_gemma(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Calls the /v1/chat/completions endpoint on gemma4_server.py."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{GEMMA4_URL}/v1/chat/completions", json=payload, timeout=300
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ==============================================================================
# 4. Web & News Search + Page Excerpt Extraction
# ==============================================================================
def fetch_page_excerpt(url: str, max_chars: int = 600) -> str:
    """Fetches a brief clean-text excerpt from a web URL to enrich search snippets."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if len(p) > 40])
        return text[:max_chars]
    except Exception:
        return ""


def execute_searches(
    queries: List[str],
    max_results_per_query: int = 4,
    timelimit: str = None,
    deep_scrape: bool = True,
) -> List[Dict[str, str]]:
    """Runs both News and Web searches via DuckDuckGo and deduplicates URLs."""
    ddgs = DDGS()
    collected: List[Dict[str, str]] = []
    seen_urls = set()
    tl = None if timelimit in (None, "all", "") else timelimit

    for q in queries:
        print(f"  🔎 Searching: '{q}'...")
        # 1. General web search (prioritized for product/store pages)
        try:
            web_hits = ddgs.text(
                q, max_results=max_results_per_query, timelimit=tl
            )
            for item in web_hits or []:
                url = item.get("href") or item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(
                    {
                        "query": q,
                        "title": item.get("title", "Untitled"),
                        "url": url,
                        "source": "Web",
                        "date": "",
                        "snippet": item.get("body", ""),
                    }
                )
        except Exception as e:
            print(f"    ⚠️ Web search warning for '{q}': {e}")

        # 2. Recent news search
        try:
            news_hits = ddgs.news(
                q, max_results=max(2, max_results_per_query // 2), timelimit=tl
            )
            for item in news_hits or []:
                url = item.get("url") or item.get("href", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(
                    {
                        "query": q,
                        "title": item.get("title", "Untitled"),
                        "url": url,
                        "source": item.get("source", "News"),
                        "date": item.get("date", ""),
                        "snippet": item.get("body", ""),
                    }
                )
        except Exception as e:
            print(f"    ⚠️ News search warning for '{q}': {e}")

    # Enrich top results with scraped article body text
    if deep_scrape and collected:
        print(f"  🌐 Enriching top {min(8, len(collected))} results with page excerpts...")
        for item in collected[:8]:
            excerpt = fetch_page_excerpt(item["url"])
            if excerpt:
                item["excerpt"] = excerpt

    return collected


# ==============================================================================
# 5. Agentic Workflow: Query Planning -> Search -> Synthesis
# ==============================================================================
def plan_search_queries(topic: str, num_queries: int = 4) -> List[str]:
    """Asks Gemma 4 to act as a research planner and generate optimal search queries."""
    today_str = date.today().strftime("%B %d, %Y")
    prompt = (
        f"Current Date: {today_str}\n"
        f"Research Goal: {topic}\n\n"
        f"Generate {num_queries} distinct, specific search engine queries to find the most "
        f"important, up-to-date facts, news, and insights about this topic.\n"
        f"Respond ONLY with a valid JSON array of strings, for example:\n"
        f'["query 1", "query 2", "query 3"]'
    )
    print("🧠 Step 1: Asking Gemma 4 to plan targeted search queries...")
    raw = ask_gemma([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=256)

    # Extract JSON array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            queries = json.loads(match.group(0))
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [q.strip() for q in queries if q.strip()][:num_queries]
        except Exception:
            pass

    # Fallback if JSON parsing fails
    lines = [
        re.sub(r"^[\d\-\*\.\s\"']+|[\"']+$", "", line.strip())
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith(("[", "]"))
    ]
    return lines[:num_queries] if lines else [topic]


def synthesize_briefing(topic: str, results: List[Dict[str, str]]) -> str:
    """Passes gathered search evidence to Gemma 4 to write a friendly, casual deal summary."""
    today_str = date.today().strftime("%B %d, %Y")
    context_blocks = []
    for idx, r in enumerate(results[:10], 1):
        block = (
            f"[{idx}] Title: {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    Snippet: {r['snippet']}"
        )
        if r.get("excerpt"):
            block += f"\n    Page Text: {r['excerpt']}"
        context_blocks.append(block)

    search_context = "\n\n".join(context_blocks) if context_blocks else "No results found."

    prompt = f"""You are helping find the best prices and store links for: {topic} (Date: {today_str}).

Here are the search results found today:
{search_context}

Write a short, casual, easy-to-scan HTML summary of the prices and websites found so we can quickly check the links and decide whether to buy.
Keep it simple and informal (no corporate/analyst jargon):
1. **Quick Price Check**: 2-3 short bullets on the lowest prices found and whether it's in stock.
2. **Where to Buy (Stores & Links)**: A simple bulleted list of the stores/websites found, showing the price (if mentioned), a brief note on the listing, and a direct clickable link (`<a href="URL">Store / Listing Name</a>`).

Use only simple HTML tags (`<h3>`, `<p>`, `<ul>`, `<li>`, `<strong>`, `<a href="...">`). Do NOT wrap in `<html>` or `<body>` tags, and do NOT use ```html code fences.
"""
    print("🧠 Step 3: Summarizing prices and links with Gemma 4...")
    raw_html = ask_gemma(
        [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=768
    )
    # Strip markdown code fences if the model included them
    raw_html = re.sub(r"^```(?:html)?\s*", "", raw_html.strip(), flags=re.IGNORECASE)
    raw_html = re.sub(r"\s*```$", "", raw_html.strip())
    return raw_html


def wrap_in_email_template(topic: str, queries: List[str], body_html: str, recipient: str = None) -> str:
    """Wraps the summary in a clean, simple, easy-to-read email layout."""
    now_str = datetime.now().strftime("%b %d, %Y")
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color: #1e293b; background-color: #f8fafc; margin: 0; padding: 16px; }}
    h3 {{ color: #0f172a; margin-top: 18px; margin-bottom: 8px; font-size: 16px; }}
    p, li {{ color: #334155; line-height: 1.5; font-size: 14px; }}
    ul {{ padding-left: 20px; margin-top: 6px; }}
    li {{ margin-bottom: 8px; }}
    a {{ color: #2563eb; text-decoration: underline; font-weight: 500; }}
  </style>
</head>
<body>
  <div style="max-width: 640px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; padding: 20px; border: 1px solid #e2e8f0;">
    <h2 style="margin: 0 0 4px 0; font-size: 18px; color: #0f172a;">🏷️ Deal & Price Update ({now_str})</h2>
    <p style="margin: 0 0 16px 0; font-size: 13px; color: #64748b;">Search: {topic}</p>
    <hr style="border: none; border-top: 1px solid #e2e8f0; margin-bottom: 16px;">
    {body_html}
  </div>
</body>
</html>"""


# ==============================================================================
# 6. Email Dispatcher (Reuses /home/erfontes/research/trader/.env)
# ==============================================================================
def send_email_report(subject: str, html_content: str, recipient: str = None) -> bool:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    recipient = recipient or os.getenv("ALERT_EMAIL_TO", "erfontes@gmail.com")

    out_file = os.path.join(BASE_DIR, "latest_search_briefing.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"📄 Saved HTML research report to: {out_file}")

    if not smtp_host or not smtp_user or not smtp_pass or not recipient:
        print("⚠️ SMTP credentials missing in .env; skipped email send.")
        return False

    print(f"📧 Sending email to {recipient} via {smtp_host}:{smtp_port}...")
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.attach(MIMEText(html_content, "html"))

        use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        with smtplib.SMTP(smtp_host, smtp_port, timeout=25) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [recipient], msg.as_string())

        print(f"✅ Email successfully sent to {recipient}!")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


# ==============================================================================
# 7. CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 31B Agentic Web Search & Email Briefing"
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default=DEFAULT_TOPICS,
        help="Research topic or question for Gemma 4 to investigate.",
    )
    parser.add_argument(
        "--queries",
        "-q",
        type=int,
        default=4,
        help="Number of search queries Gemma 4 should generate (default: 4).",
    )
    parser.add_argument(
        "--results-per-query",
        "-r",
        type=int,
        default=4,
        help="Max search results per query (default: 4).",
    )
    parser.add_argument(
        "--timelimit",
        type=str,
        default=None,
        choices=["d", "w", "m", "y", "all"],
        help="Search recency filter: d (day), w (week), m (month), y (year), or all (no limit).",
    )
    parser.add_argument(
        "--to",
        type=str,
        default=None,
        help="Override recipient email address (defaults to ALERT_EMAIL_TO in .env).",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Run search and save HTML locally without sending email.",
    )
    args = parser.parse_args()

    print("=" * 76)
    print(f"🚀 GEMMA 4 SEARCH AGENT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📌 Topic: {args.topic}")
    print("=" * 76)

    ensure_gemma_server_online()
    queries = plan_search_queries(args.topic, num_queries=args.queries)
    print(f"📋 Planned Queries ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")

    print("\n🌐 Step 2: Executing live web & news searches...")
    results = execute_searches(
        queries,
        max_results_per_query=args.results_per_query,
        timelimit=args.timelimit,
        deep_scrape=True,
    )
    print(f"✅ Collected {len(results)} unique sources.")

    body_html = synthesize_briefing(args.topic, results)
    recipient_display = args.to or os.getenv("ALERT_EMAIL_TO", "erfontes@gmail.com")
    full_html = wrap_in_email_template(args.topic, queries, body_html, recipient_display)

    if args.no_email:
        out_file = os.path.join(BASE_DIR, "latest_search_briefing.html")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"📄 Dry-run complete. Report saved to: {out_file}")
    else:
        short_topic = args.topic[:55] + ("..." if len(args.topic) > 55 else "")
        subject = f"🏷️ Price & Link Check: {short_topic} ({date.today().strftime('%b %d')})"
        send_email_report(subject, full_html, recipient=args.to)


if __name__ == "__main__":
    main()
