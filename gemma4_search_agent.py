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
import sqlite3
import imaplib
import email
from email.header import decode_header
import argparse
import subprocess
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any, Tuple

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
# 4B. Direct Lowes Foods Weekly Ad + Gmail Receipt Habit Analyzer
# ==============================================================================
LOWES_FLIPP_TOKEN = "07da9af5b839c91b8ebca4cee1c162e4"
GROCERY_DB_PATH = os.path.join(BASE_DIR, "grocery_habits.db")

STOPWORDS = {
    "LOWES", "FOODS", "LFS", "ORGANIC", "FRESH", "ALL", "NATURAL", "SELECTED",
    "THE", "AND", "OR", "IN", "WITH", "OF", "OZ", "LB", "PK", "CT", "EA", "BOTT",
    "CAN", "CANS", "PACK", "2PK", "4PK", "6PK", "12PK", "4P", "1LB", "100%", "2%",
    "0.0%", "LITER", "LI", "MUL", "C", "H", "S", "FP", "HH", "SW", "W/BUN"
}


def init_grocery_db() -> sqlite3.Connection:
    conn = sqlite3.connect(GROCERY_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            msg_id TEXT PRIMARY KEY,
            receipt_date TEXT,
            store_number TEXT,
            subtotal REAL,
            savings REAL,
            total REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS receipt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_id TEXT,
            receipt_date TEXT,
            product_id TEXT,
            item_name TEXT,
            qty REAL,
            unit_price REAL,
            line_total REAL,
            UNIQUE(msg_id, product_id, item_name)
        )
    """)
    conn.commit()
    return conn


def parse_lowes_receipt_text(text: str) -> Dict[str, Any]:
    """Parses both Concord (modern multi-line) and Waxhaw (compact) Lowes Foods email receipts."""
    store_num = "0282"
    receipt_date = date.today().isoformat()

    # Format A: 11/21/2026 13:55 PM Store: 0282
    m_date = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+\d{1,2}:\d{2}(?:\s*[AP]M)?\s+Store:\s*(\d+)", text)
    if m_date:
        raw_dt, store_num = m_date.group(1), m_date.group(2)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                receipt_date = datetime.strptime(raw_dt, fmt).date().isoformat()
                break
            except ValueError:
                pass
    else:
        # Format B: 11/22/26 13:04 0291 ... Store #291
        m_date_b = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+\d{1,2}:\d{2}\s+(\d{4})", text)
        if m_date_b:
            raw_dt, store_num = m_date_b.group(1), m_date_b.group(2)
            for fmt in ("%m/%d/%y", "%m/%d/%Y"):
                try:
                    receipt_date = datetime.strptime(raw_dt, fmt).date().isoformat()
                    break
                except ValueError:
                    pass

    items = []
    # Match Format A item blocks:
    # ITEM NAME 21.98
    # Product ID: 7289000621
    # 2 @ 10.99
    pattern_a = re.compile(
        r"------------------------------\s*\r?\n"
        r"([A-Z0-9\'\-%\.\/&\s]+?)\s+(\d+\.\d{2})\s*\r?\n"
        r"Product ID:\s*(\d+)\s*\r?\n"
        r"([\d\.]+)\s*@\s*(\d+\.\d{2})",
        re.MULTILINE,
    )
    for m in pattern_a.finditer(text):
        name = m.group(1).strip()
        line_total = float(m.group(2))
        pid = m.group(3).strip()
        qty = float(m.group(4))
        unit_price = float(m.group(5))
        items.append({
            "product_id": pid,
            "item_name": name,
            "qty": qty,
            "unit_price": unit_price,
            "line_total": line_total,
        })

    # Match Format B compact lines (e.g., CARD #XXXXXXX8698   5" ORCHID CERAMI   19.99 T)
    if not items:
        pattern_b = re.compile(r"CARD\s+#[X\d]+\s+([A-Z0-9\"\'\-\s]+?)\s+(\d+\.\d{2})\s+[TF]", re.IGNORECASE)
        for m in pattern_b.finditer(text):
            name = m.group(1).strip()
            price = float(m.group(2))
            items.append({
                "product_id": "UNKNOWN",
                "item_name": name,
                "qty": 1.0,
                "unit_price": price,
                "line_total": price,
            })

    def extract_money(label: str) -> float:
        m = re.search(rf"(?:^|\n)\s*{label}\s+(-?\d+\.\d{{2}})", text, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    subtotal = extract_money("Subtotal")
    savings = abs(extract_money("Savings"))
    total = extract_money("Total") or extract_money("BALANCE")

    return {
        "receipt_date": receipt_date,
        "store_number": store_num,
        "subtotal": subtotal,
        "savings": savings,
        "total": total,
        "items": items,
    }


def sync_receipts_from_gmail() -> Dict[str, Any]:
    """Connects to Gmail IMAP, parses any new Lowes Foods receipt emails, and updates grocery_habits.db."""
    conn = init_grocery_db()
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    if not smtp_user or not smtp_pass:
        print("⚠️ Gmail credentials not found in .env; skipping receipt sync.")
        return get_household_habits(conn)

    new_receipts = 0
    new_items = 0
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(smtp_user, smtp_pass)
        mail.select('"[Gmail]/All Mail"')

        status, data = mail.search(None, '(TEXT "Lowes Foods")')
        msg_ids = data[0].split() if data and data[0] else []

        for raw_id in msg_ids:
            # Fetch header + Message-ID first to avoid re-downloading processed messages
            status, hdr_data = mail.fetch(raw_id, "(BODY.PEEK[HEADER.FIELDS (SUBJECT MESSAGE-ID DATE)])")
            if not hdr_data or not hdr_data[0]:
                continue
            hdr_msg = email.message_from_bytes(hdr_data[0][1])
            subj = str(email.header.make_header(decode_header(hdr_msg.get("Subject", ""))))
            if "receipt" not in subj.lower() or "lowes" not in subj.lower():
                continue

            unique_msg_id = (hdr_msg.get("Message-ID") or raw_id.decode()).strip()
            exists = conn.execute("SELECT 1 FROM receipts WHERE msg_id = ?", (unique_msg_id,)).fetchone()
            if exists:
                continue

            status, msg_data = mail.fetch(raw_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            plain_body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        plain_body = payload.decode(errors="ignore")
                        break
            if not plain_body:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            soup = BeautifulSoup(payload.decode(errors="ignore"), "html.parser")
                            plain_body = soup.get_text("\n", strip=True)
                            break

            parsed = parse_lowes_receipt_text(plain_body)
            if not parsed["items"]:
                continue

            conn.execute(
                "INSERT OR IGNORE INTO receipts (msg_id, receipt_date, store_number, subtotal, savings, total) VALUES (?, ?, ?, ?, ?, ?)",
                (unique_msg_id, parsed["receipt_date"], parsed["store_number"], parsed["subtotal"], parsed["savings"], parsed["total"]),
            )
            new_receipts += 1
            for it in parsed["items"]:
                conn.execute(
                    """INSERT OR IGNORE INTO receipt_items
                       (msg_id, receipt_date, product_id, item_name, qty, unit_price, line_total)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (unique_msg_id, parsed["receipt_date"], it["product_id"], it["item_name"], it["qty"], it["unit_price"], it["line_total"]),
                )
                new_items += 1

        conn.commit()
        mail.logout()
    except Exception as e:
        print(f"⚠️ Gmail receipt sync warning: {e}")

    habits = get_household_habits(conn)
    habits["new_receipts_synced"] = new_receipts
    habits["new_items_synced"] = new_items
    conn.close()
    return habits


def get_household_habits(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Aggregates receipt history into ranked household staples and recent purchases."""
    rows = conn.execute("""
        SELECT
            item_name,
            product_id,
            COUNT(DISTINCT msg_id) AS trip_count,
            SUM(qty) AS total_qty,
            ROUND(AVG(unit_price), 2) AS avg_unit_price,
            ROUND(AVG(line_total / CASE WHEN qty > 0 THEN qty ELSE 1 END), 2) AS avg_paid_price,
            MAX(receipt_date) AS last_bought
        FROM receipt_items
        GROUP BY item_name
        ORDER BY trip_count DESC, total_qty DESC, last_bought DESC
    """).fetchall()

    receipt_summary = conn.execute("""
        SELECT COUNT(*), COALESCE(SUM(total), 0), COALESCE(SUM(savings), 0), MAX(receipt_date)
        FROM receipts
    """).fetchone()

    items = []
    for r in rows:
        items.append({
            "item_name": r[0],
            "product_id": r[1],
            "trip_count": r[2],
            "total_qty": r[3],
            "avg_unit_price": r[4],
            "avg_paid_price": r[5],
            "last_bought": r[6],
        })

    return {
        "total_receipts": receipt_summary[0] if receipt_summary else 0,
        "total_spent": round(receipt_summary[1], 2) if receipt_summary else 0.0,
        "total_saved": round(receipt_summary[2], 2) if receipt_summary else 0.0,
        "latest_receipt_date": receipt_summary[3] if receipt_summary else "N/A",
        "items": items,
    }


def extract_keywords(text: str) -> set:
    tokens = re.findall(r"[A-Z]{3,}", text.upper())
    return {t for t in tokens if t not in STOPWORDS}


def match_habits_to_circular(habits: Dict[str, Any], products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-references household receipt items against the current week's Lowes Foods sale products."""
    matches = []
    seen_pairs = set()

    for h in habits.get("items", []):
        h_name = h["item_name"]
        h_kw = extract_keywords(h_name)
        if not h_kw:
            continue

        for p in products:
            p_name = (p.get("name") or "").strip()
            p_desc = (p.get("description") or "").replace("\n", " ").strip()
            p_kw = extract_keywords(p_name + " " + p_desc)
            overlap = h_kw.intersection(p_kw)

            # Match if at least one distinctive brand/product keyword overlaps
            # (or 2 keywords if generic like POTATO / CHIPS / BREAD / BEEF)
            generic_words = {"POTATO", "CHIPS", "BREAD", "BEEF", "CHICKEN", "MILK", "CHEESE", "SALAD", "CAKE", "BEER"}
            strong_overlap = overlap - generic_words
            is_match = len(strong_overlap) >= 1 or len(overlap) >= 2

            if is_match:
                pair_key = (h_name, p_name)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                price = (p.get("price_text") or "").strip()
                unit = (p.get("post_price_text") or "").strip()
                story = (p.get("sale_story") or "").strip()
                price_str = f"${price}" if price and not price.startswith("$") else price
                if unit and price_str:
                    price_str = f"{price_str} ({unit})"

                matches.append({
                    "habit_item": h_name,
                    "last_bought": h["last_bought"],
                    "usual_price": f"${h['avg_paid_price']:.2f} (reg ${h['avg_unit_price']:.2f})",
                    "sale_item": p_name,
                    "sale_price": price_str,
                    "sale_story": story,
                    "sale_desc": p_desc,
                })
    return matches


def fetch_lowes_foods_circular(postal_code: str = "28027", store_code: str = "282") -> Dict[str, Any]:
    """
    Pulls the actual live product items (Meat, Chicken, Seafood, BOGOs, Digital Coupons)
    directly from the official Lowes Foods Weekly Ad API for Lowes Foods of Concord (#282, 28027).
    """
    pub_url = (
        f"https://dam.flippenterprise.net/flyerkit/publications/fflowesfoods"
        f"?locale=en&access_token={LOWES_FLIPP_TOKEN}&show_storefronts=true"
        f"&postal_code={postal_code}&store_code={store_code}"
    )
    pubs = requests.get(pub_url, timeout=15).json()
    if not pubs:
        raise RuntimeError("No active Lowes Foods weekly circular found.")

    pub = pubs[0]
    pub_id = pub["id"]
    valid_from = pub.get("valid_from", "")
    valid_to = pub.get("valid_to", "")

    prod_url = (
        f"https://dam.flippenterprise.net/flyerkit/publication/{pub_id}/products"
        f"?display_type=all&locale=en&access_token={LOWES_FLIPP_TOKEN}"
    )
    products = requests.get(prod_url, timeout=15).json()

    def format_item(p: Dict[str, Any]) -> str:
        name = (p.get("name") or "").strip()
        price = (p.get("price_text") or "").strip()
        unit = (p.get("post_price_text") or "").strip()
        story = (p.get("sale_story") or "").strip()
        desc = (p.get("description") or "").replace("\n", " ").strip()

        price_str = f"${price}" if price and not price.startswith("$") else price
        if unit and price_str:
            price_str = f"{price_str} ({unit})"
        parts = [f"{name}"]
        if price_str:
            parts.append(f"Price: {price_str}")
        if story:
            parts.append(f"Deal: {story}")
        if desc:
            parts.append(f"({desc})")
        return " — ".join(parts)

    meat_seafood_items = []
    heavy_discount_items = []

    for p in products:
        cats = p.get("categories") or []
        story = (p.get("sale_story") or "").upper()
        name_upper = (p.get("name") or "").upper()

        is_meat_seafood = (
            "Meat & Seafood" in cats
            or any(
                kw in name_upper
                for kw in [
                    "CHICKEN", "BEEF", "STEAK", "ROAST", "PORK", "BACON", "SAUSAGE",
                    "SHRIMP", "SALMON", "CRAB", "LOBSTER", "TUNA", "COD", "CATFISH",
                    "HALIBUT", "MAHI", "MEATBALL", "TURKEY", "FRYER"
                ]
            )
        )
        if is_meat_seafood:
            meat_seafood_items.append(format_item(p))
        elif any(kw in story for kw in ["BUY", "FREE", "OFF", "HALF", "BOGO", "%"]):
            heavy_discount_items.append(format_item(p))

    return {
        "valid_from": valid_from,
        "valid_to": valid_to,
        "total_items": len(products),
        "raw_products": products,
        "meat_seafood": meat_seafood_items,
        "heavy_discounts": heavy_discount_items,
    }


def synthesize_lowes_foods_briefing(circular_data: Dict[str, Any], habits: Dict[str, Any]) -> str:
    """Uses Gemma 4 31B to create a personalized weekly shopping list combining receipt habits + weekly ad discounts."""
    valid_range = f"{circular_data['valid_from'][:10]} to {circular_data['valid_to'][:10]}"
    meat_list = "\n".join(f"- {item}" for item in circular_data["meat_seafood"][:22])
    discount_list = "\n".join(f"- {item}" for item in circular_data["heavy_discounts"][:18])

    # Build receipt habit summary & direct matches
    matches = match_habits_to_circular(habits, circular_data.get("raw_products", []))
    match_lines = []
    for m in matches[:10]:
        deal_part = " | ".join(x for x in [m["sale_price"], m["sale_story"], m["sale_desc"]] if x)
        match_lines.append(
            f"- ON SALE NOW: {m['sale_item']} ({deal_part}) <-- Matches receipt item '{m['habit_item']}' "
            f"(last bought {m['last_bought']} at {m['usual_price']})"
        )
    matches_block = "\n".join(match_lines) if match_lines else "- No exact brand matches this week, check category staples below."

    habit_lines = []
    for h in habits.get("items", [])[:18]:
        habit_lines.append(
            f"- {h['item_name']} (Bought {h['trip_count']}x, qty {h['total_qty']:g}, "
            f"paid ${h['avg_paid_price']:.2f}, last {h['last_bought']})"
        )
    habits_block = "\n".join(habit_lines) if habit_lines else "- No receipt items recorded yet."

    prompt = (
        f"You are building the personalized Smart Shopping weekly grocery list for Fabricia and Erfontes "
        f"at Lowes Foods of Concord (Ad cycle: {valid_range}).\n\n"
        f"1. HOUSEHOLD BUYING HABITS (From their Lowes Foods email receipts — "
        f"{habits.get('total_receipts', 0)} receipts, ${habits.get('total_saved', 0):.2f} saved):\n"
        f"{habits_block}\n\n"
        f"2. RECEIPT ITEMS ON SALE THIS WEEK:\n"
        f"{matches_block}\n\n"
        f"3. MEAT, CHICKEN & SEAFOOD ON SALE THIS WEEK:\n"
        f"{meat_list}\n\n"
        f"4. TOP BOGO & HEAVY DISCOUNTS THIS WEEK:\n"
        f"{discount_list}\n\n"
        f"Create a clean, casual, actionable HTML weekly shopping guide & list using ONLY "
        f"<h3>, <p>, <ul>, <li>, <strong>:\n"
        f"1. <h3>🎯 Your Receipt Staples ON SALE This Week</h3>: Highlight items from Section 2 "
        f"or close substitutes from their habits (steaks, chips, bread, milk, sodas, beer, produce) "
        f"that are on sale right now, comparing the sale deal vs. what they paid on their receipt.\n"
        f"2. <h3>📋 Suggested Weekly Restock Checklist (Based on Your Habits)</h3>: Recommend 6-8 staple items "
        f"from their purchase history to restock soon (noting last bought date and matching sale/BOGO deals).\n"
        f"3. <h3>🍗🥩🦐 Meat, Chicken & Seafood Deals This Week</h3>: Group Chicken/Poultry, "
        f"Beef/Pork/Bacon, and Seafood deals with exact prices and BOGOs.\n"
        f"4. <h3>🔥 Other Can't-Miss BOGOs & Heavy Discounts</h3>: Top 8-10 BOGO / Digital Coupon deals.\n\n"
        f"Do NOT wrap in <html> or <body> tags, and do NOT use markdown code fences. "
        f"Keep all prices and receipt details 100% accurate!"
    )
    print("🧠 Step 3: Generating personalized Smart Shopping list with Gemma 4 31B (Receipts + Weekly Ad)...")
    raw_html = ask_gemma(
        [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=1536
    )
    raw_html = re.sub(r"^```(?:html)?\s*", "", raw_html.strip(), flags=re.IGNORECASE)
    raw_html = re.sub(r"\s*```$", "", raw_html.strip())

    receipt_badge = (
        f'<div style="background-color:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:10px 14px; '
        f'margin-bottom:16px; font-size:13px; color:#166534;">'
        f'🧾 <strong>Receipt Intelligence Active:</strong> Synced <strong>{habits.get("total_receipts", 0)} Lowes Foods receipts</strong> '
        f'({len(habits.get("items", []))} unique items tracked • Latest receipt: {habits.get("latest_receipt_date", "N/A")} • '
        f'Total receipt savings tracked: <strong>${habits.get("total_saved", 0):.2f}</strong>).'
        f'</div>'
    )
    links_footer = (
        '<h3>🔗 Quick Links</h3>'
        '<ul>'
        '<li><a href="https://weeklyad.lowesfoods.com/">Lowes Foods of Concord Weekly Ad Circular</a></li>'
        '<li><a href="https://shop.lowesfoods.com/shop/coupons">Clip Lowes Foods Digital Coupons</a></li>'
        '<li><a href="https://shop.lowesfoods.com/">Lowes Foods To Go (Online Shop)</a></li>'
        '</ul>'
    )
    return receipt_badge + "\n" + raw_html + "\n" + links_footer


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


def wrap_in_email_template(
    topic: str, queries: List[str], body_html: str, recipient: str = None, title: str = "🏷️ Deal & Price Update"
) -> str:
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
    <h2 style="margin: 0 0 4px 0; font-size: 18px; color: #0f172a;">{title} ({now_str})</h2>
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
    raw_recipients = recipient or os.getenv("ALERT_EMAIL_TO", "erfontes@gmail.com")
    recipient_list = [r.strip() for r in raw_recipients.split(",") if r.strip()]

    out_file = os.path.join(BASE_DIR, "latest_search_briefing.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"📄 Saved HTML research report to: {out_file}")

    if not smtp_host or not smtp_user or not smtp_pass or not recipient_list:
        print("⚠️ SMTP credentials missing in .env; skipped email send.")
        return False

    recipients_display = ", ".join(recipient_list)
    print(f"📧 Sending email to {recipients_display} via {smtp_host}:{smtp_port}...")
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipients_display
        msg.attach(MIMEText(html_content, "html"))

        use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        with smtplib.SMTP(smtp_host, smtp_port, timeout=25) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, recipient_list, msg.as_string())

        print(f"✅ Email successfully sent to {recipients_display}!")
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
        help="Override recipient email address(es), comma-separated (defaults to ALERT_EMAIL_TO in .env).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title/subject prefix (e.g., 'Smart Shopping').",
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
    recipient_display = args.to or os.getenv("ALERT_EMAIL_TO", "erfontes@gmail.com")
    header_title = f"🛒 {args.title}" if args.title else "🏷️ Deal & Price Update"

    if "lowesfoods" in args.topic.lower() or "lowes foods" in args.topic.lower():
        print("🧾 Step 1: Syncing Lowes Foods email receipts from Gmail IMAP into grocery_habits.db...")
        habits = sync_receipts_from_gmail()
        print(
            f"✅ Synced {habits.get('new_receipts_synced', 0)} new receipts "
            f"(Total tracked: {habits.get('total_receipts', 0)} receipts, "
            f"{len(habits.get('items', []))} unique items, ${habits.get('total_saved', 0):.2f} saved)."
        )

        print("🛒 Step 2: Fetching live Weekly Ad products directly from Lowes Foods of Concord (#282)...")
        circular = fetch_lowes_foods_circular()
        print(
            f"✅ Pulled {circular['total_items']} total ad items "
            f"({len(circular['meat_seafood'])} Meat/Chicken/Seafood, "
            f"{len(circular['heavy_discounts'])} BOGO/Heavy Discounts) "
            f"valid {circular['valid_from'][:10]} to {circular['valid_to'][:10]}."
        )
        body_html = synthesize_lowes_foods_briefing(circular, habits)
        topic_label = f"Lowes Foods of Concord ({circular['valid_from'][:10]} to {circular['valid_to'][:10]}) — Personalized by Your Receipts & Weekly Ad"
        full_html = wrap_in_email_template(
            topic_label, [], body_html, recipient_display, title=header_title
        )
    else:
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
        full_html = wrap_in_email_template(
            args.topic, queries, body_html, recipient_display, title=header_title
        )

    if args.no_email:
        out_file = os.path.join(BASE_DIR, "latest_search_briefing.html")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"📄 Dry-run complete. Report saved to: {out_file}")
    else:
        if args.title:
            subject = f"🛒 {args.title}: Lowes Foods Deals ({date.today().strftime('%b %d')})" if "lowes" in args.topic.lower() else f"🛒 {args.title} ({date.today().strftime('%b %d')})"
        else:
            short_topic = args.topic[:55] + ("..." if len(args.topic) > 55 else "")
            subject = f"🏷️ Price & Link Check: {short_topic} ({date.today().strftime('%b %d')})"
        send_email_report(subject, full_html, recipient=args.to)


if __name__ == "__main__":
    main()
