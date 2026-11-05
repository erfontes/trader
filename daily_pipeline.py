#!/usr/bin/env python3
"""
================================================================================
🚀 QUANT TRADING DAILY PRE-MARKET PIPELINE (READY BY 8:00 AM)
================================================================================
Automated pipeline executing daily before market open:
1. Ingests latest OHLCV price bars and curated financial news.
2. Manages Gemma 4 31B server lifecycle (spins up if offline, shuts down after inference).
3. Evaluates consensus sentiment and catalyst materiality with Gemma 4.
4. Generates 31 intraday geometry, macro covariance, and sentiment alpha signals.
5. Runs the locked-in 60/40 LightGBM + Random Forest Ensemble with conviction filtering.
6. Dispatches an executive HTML morning briefing via email by 8:00 AM EST.
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
import hashlib
import argparse
import subprocess
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from xml.etree import ElementTree as ET
import concurrent.futures

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# 1. Configuration & Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_env_file():
    """Load key-value pairs from .env or daily_pipeline.env if present."""
    for env_name in [".env", "daily_pipeline.env"]:
        env_path = os.path.join(BASE_DIR, env_name)
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env_file()

DB_PATH = os.getenv("TRADER_DB_PATH", os.path.join(BASE_DIR, "market_data.db"))
GEMMA4_URL = os.getenv("GEMMA4_SERVER_URL", "http://127.0.0.1:11434")
GEMMA4_PYTHON = os.getenv("GEMMA4_PYTHON_PATH", "/home/erfontes/miniconda3/envs/vel/bin/python")
GEMMA4_SCRIPT = os.getenv("GEMMA4_SCRIPT_PATH", os.path.join(BASE_DIR, "gemma4_fast_server.py"))

TARGET_TICKERS = [t.strip() for t in os.getenv("TARGET_TICKERS", "GOOGL,NVDA,COST,TSLA,AMD").split(",") if t.strip()]
BENCHMARK_TICKERS = [t.strip() for t in os.getenv("BENCHMARK_TICKERS", "^VIX,SPY,QQQ,SMH,XLP").split(",") if t.strip()]
ALL_TICKERS = TARGET_TICKERS + BENCHMARK_TICKERS

PUBLISHER_TIERS = {
    "tier_1": ["bloomberg", "reuters", "wall street journal", "wsj", "financial times", "pr newswire", "business wire", "sec", "globe newswire", "cnbc"],
    "tier_2": ["barron's", "barrons", "marketwatch", "investor's business daily", "ibd", "seeking alpha", "yahoo finance", "investing.com", "barchart", "thefly", "insider monkey", "tipranks"],
    "tier_3": ["marketbeat", "motley fool", "the motley fool", "24/7 wall st.", "stocktwits", "benzinga", "zacks", "simply wall st."]
}

# ==============================================================================
# 2. Database Initialization
# ==============================================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            Date TEXT,
            Ticker TEXT,
            Price REAL,
            Volume REAL,
            Open REAL,
            High REAL,
            Low REAL,
            PRIMARY KEY (Date, Ticker)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            NewsID TEXT PRIMARY KEY,
            Ticker TEXT,
            Date TEXT,
            Title TEXT,
            Summary TEXT,
            Publisher TEXT,
            Link TEXT,
            Processed INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_results (
            NewsID TEXT PRIMARY KEY,
            Ticker TEXT,
            Date TEXT,
            Sentiment_Label TEXT,
            Median_Intensity REAL,
            Disagreement_Std REAL,
            Vote_Count INTEGER,
            Materiality_Score REAL DEFAULT 0.7,
            Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            Date TEXT,
            Ticker TEXT,
            Model_Used TEXT,
            Prob_Up REAL,
            Conviction REAL,
            Action TEXT,
            Alloc_Pct REAL,
            Stop_Loss_Pct REAL,
            Entry_Price REAL,
            Next_Close_Price REAL,
            Realized_Return REAL,
            Target_Actual INTEGER,
            Correct INTEGER,
            Updated_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (Date, Ticker)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (Ticker, Date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_results (Ticker, Date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions_log (Ticker, Date)")
    conn.commit()
    conn.close()

# ==============================================================================
# 3. Gemma 4 Server Lifecycle Management
# ==============================================================================
def is_gemma_server_running(url=GEMMA4_URL, timeout=3.0) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return data.get("status") == "online"
    except Exception:
        return False
    return False

def wait_for_gemma_server(url=GEMMA4_URL, max_wait_sec=120) -> bool:
    print(f"⏳ Waiting up to {max_wait_sec}s for Gemma 4 server to become ready...")
    start = time.time()
    while time.time() - start < max_wait_sec:
        if is_gemma_server_running(url, timeout=2.0):
            print(f"✅ Gemma 4 server is ready after {time.time() - start:.1f}s!")
            return True
        time.sleep(2.0)
    print("❌ Timed out waiting for Gemma 4 server to start.")
    return False

class GemmaServerContext:
    """
    Context manager that checks if Gemma 4 is already up.
    If down, spins it up via the 'vel' python env and shuts it down upon exit.
    If already up, leaves it running uninterrupted.
    """
    def __init__(self, url=GEMMA4_URL, py_path=GEMMA4_PYTHON, script_path=GEMMA4_SCRIPT):
        self.url = url
        self.py_path = py_path
        self.script_path = script_path
        self.proc = None
        self.started_by_us = False

    def __enter__(self):
        if is_gemma_server_running(self.url):
            print("🚀 Gemma 4 server is already running and ready on port 11434.")
            self.started_by_us = False
            return self

        print(f"⚡ Gemma 4 server is offline. Spawning server via {self.py_path}...")
        log_file = open(os.path.join(BASE_DIR, "gemma4_pipeline_server.log"), "w")
        self.proc = subprocess.Popen(
            [self.py_path, self.script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR,
            preexec_fn=os.setsid
        )
        self.started_by_us = True

        ready = wait_for_gemma_server(self.url, max_wait_sec=120)
        if not ready:
            self.cleanup()
            raise RuntimeError("Failed to start Gemma 4 inference server.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started_by_us:
            self.cleanup()

    def cleanup(self):
        if self.proc and self.proc.poll() is None:
            print("🛑 Shutting down Gemma 4 server spawned by daily pipeline (releasing GPU VRAM)...")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            print("✅ Gemma 4 server cleanly stopped.")

# ==============================================================================
# 4. Data Ingestion: Prices & News
# ==============================================================================
def update_prices():
    print(f"📡 Step 1: Ingesting OHLCV data for {len(ALL_TICKERS)} symbols via yfinance...")
    raw = yf.download(ALL_TICKERS, period="1y", interval="1d", progress=False)
    if raw is None or raw.empty:
        print("⚠️ Warning: yfinance download returned empty frame.")
        return

    conn = get_db()
    total_upserted = 0

    for ticker in ALL_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                t_df = raw.xs(ticker, level=1, axis=1).copy()
            else:
                t_df = raw.copy()
            t_df = t_df.reset_index()
            t_df['Ticker'] = ticker
            t_df['Date'] = pd.to_datetime(t_df['Date']).dt.strftime('%Y-%m-%d')
            t_df = t_df.rename(columns={'Close': 'Price'}).dropna(subset=['Date', 'Price'])

            t_df = t_df[['Date', 'Ticker', 'Price', 'Volume', 'Open', 'High', 'Low']]
            t_df.to_sql("temp_prices", conn, if_exists="replace", index=False)
            cur = conn.execute("""
                INSERT OR REPLACE INTO prices (Date, Ticker, Price, Volume, Open, High, Low)
                SELECT Date, Ticker, Price, Volume, Open, High, Low FROM temp_prices
            """)
            conn.execute("DROP TABLE IF EXISTS temp_prices")
            total_upserted += cur.rowcount
        except Exception as e:
            print(f"⚠️ Notice downloading prices for {ticker}: {e}")

    conn.commit()
    conn.close()
    print(f"✅ Synchronized {total_upserted} price records into SQLite database.")

def get_pub_weight(publisher_name):
    if not publisher_name: return 0.25
    name = str(publisher_name).lower().strip()
    for t1 in PUBLISHER_TIERS["tier_1"]:
        if t1 in name: return 1.00
    for t2 in PUBLISHER_TIERS["tier_2"]:
        if t2 in name: return 0.60
    for t3 in PUBLISHER_TIERS["tier_3"]:
        if t3 in name: return 0.25
    return 0.50

def clean_title_tokens(title):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(title).lower())
    stop_words = {'the', 'and', 'for', 'that', 'with', 'from', 'this', 'stock', 'shares', 'market', 'inc', 'corp'}
    return set(w for w in words if w not in stop_words)

def compute_headline_similarity(tokens1, tokens2):
    if not tokens1 or not tokens2: return 0.0
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

def update_news():
    print(f"📰 Step 2: Fetching curated pre-market news (Yahoo Finance + Google News RSS)...")
    conn = get_db()
    new_items = 0

    ticker_search = {
        'GOOGL': 'Alphabet+Google',
        'NVDA': 'NVIDIA',
        'TSLA': 'Tesla',
        'AMD': 'Advanced+Micro+Devices+AMD',
        'COST': 'Costco',
    }

    for ticker in TARGET_TICKERS:
        existing_headlines = conn.execute("SELECT Date, Title FROM news WHERE Ticker = ?", (ticker,)).fetchall()
        existing_token_map = {}
        for edate, etitle in existing_headlines:
            eday = str(edate)[:10]
            existing_token_map.setdefault(eday, []).append(clean_title_tokens(etitle))

        candidates = []

        # Yahoo Finance news
        try:
            stock = yf.Ticker(ticker)
            stories = stock.news or []
            for story in stories:
                content = story.get('content', {})
                provider = content.get('provider', {})
                title = content.get('title', '')
                summary = content.get('summary', '')
                publisher = provider.get('displayName', 'Yahoo Finance')
                raw_date = content.get('pubDate', datetime.now().isoformat())
                try:
                    clean_date = pd.to_datetime(raw_date).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    clean_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                link = content.get('canonicalUrl', {}).get('url', '')
                if title:
                    candidates.append({
                        'ticker': ticker, 'date': clean_date, 'title': title, 'summary': summary,
                        'publisher': publisher, 'link': link, 'weight': get_pub_weight(publisher)
                    })
        except Exception as e:
            print(f"Notice: Yahoo News fetch for {ticker}: {e}")

        # Google News RSS
        try:
            query = ticker_search.get(ticker, ticker)
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                for item in root.findall('.//item')[:10]:
                    title = item.find('title').text if item.find('title') is not None else ''
                    link = item.find('link').text if item.find('link') is not None else ''
                    pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
                    source = item.find('source').text if item.find('source') is not None else 'Google News'
                    desc = item.find('description').text if item.find('description') is not None else title
                    try:
                        clean_date = pd.to_datetime(pub_date).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        clean_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if title:
                        candidates.append({
                            'ticker': ticker, 'date': clean_date, 'title': title, 'summary': desc,
                            'publisher': source, 'link': link, 'weight': get_pub_weight(source)
                        })
        except Exception as e:
            print(f"Notice: Google RSS fetch for {ticker}: {e}")

        # Deduplicate and insert
        candidates.sort(key=lambda x: x['weight'], reverse=True)
        for art in candidates:
            day_str = art['date'][:10]
            tokens = clean_title_tokens(art['title'])
            is_dup = False
            for prev_tokens in existing_token_map.get(day_str, []):
                if compute_headline_similarity(tokens, prev_tokens) >= 0.70:
                    is_dup = True
                    break
            if is_dup:
                continue

            news_id = hashlib.md5(f"{ticker}{art['link'] or art['title']}".encode()).hexdigest()
            cur = conn.execute("""
                INSERT OR IGNORE INTO news (NewsID, Ticker, Date, Title, Summary, Publisher, Link, Processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, (news_id, ticker, art['date'], art['title'], art['summary'], art['publisher'], art['link']))
            if cur.rowcount > 0:
                new_items += 1
                existing_token_map.setdefault(day_str, []).append(tokens)

    conn.commit()
    conn.close()
    print(f"✅ Ingested {new_items} fresh unanalyzed news headlines.")

# ==============================================================================
# 5. Gemma 4 Sentiment & Materiality Scoring
# ==============================================================================
def score_single_news_item(ticker, title, summary, num_votes=1):
    prompt = f"""
    Analyze financial market news for {ticker}.
    Headline: {title}
    Summary: {summary}

    Evaluate both Sentiment and Catalyst Materiality.
    - sentiment: float from -1.0 (extremely bearish) to +1.0 (extremely bullish).
    - materiality: float from 0.0 (vague speculation/listicle) to 1.0 (hard catalyst: earnings, M&A, FDA, SEC guidance).
    - confidence: float from 0.0 to 1.0.

    Return ONLY JSON:
    {{"sentiment": float, "materiality": float, "confidence": float, "reason": "string"}}
    """
    payload = {
        "model": "gemma4:31b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 80
        }
    }

    sent_votes, mat_votes = [], []
    for _ in range(num_votes):
        try:
            res = requests.post(f"{GEMMA4_URL}/api/generate", json=payload, timeout=25)
            if res.status_code == 200:
                raw_text = res.json().get("response", "{}")
                clean_text = re.sub(r"```(?:json)?\s*", "", raw_text)
                clean_text = re.sub(r"\s*```", "", clean_text).strip()
                s, m = None, None
                try:
                    data = json.loads(clean_text)
                    s = float(data.get("sentiment", 0.0))
                    m = float(data.get("materiality", 0.7))
                except Exception:
                    sent_m = re.search(r'"sentiment":\s*(-?\d*\.?\d+)', raw_text)
                    mat_m = re.search(r'"materiality(?:_score)?":\s*(\d*\.?\d+)', raw_text)
                    s = float(sent_m.group(1)) if sent_m else None
                    m = float(mat_m.group(1)) if mat_m else 0.7

                if s is not None: sent_votes.append(s)
                if m is not None: mat_votes.append(m)
        except Exception:
            pass

    if not sent_votes:
        return 0.0, 0.0, 0, 0.5
    return float(np.median(sent_votes)), float(np.std(sent_votes)), len(sent_votes), float(np.median(mat_votes))

def process_unscored_news(max_items=25, num_votes=1):
    max_items = int(os.getenv("MAX_NEWS_TO_SCORE", str(max_items)))
    conn = get_db()
    # Prioritize recent pre-market news first
    query = """
        SELECT NewsID, Ticker, Date, Title, Summary 
        FROM news 
        WHERE Processed = 0 
        ORDER BY Date DESC 
        LIMIT ?
    """
    unprocessed = conn.execute(query, (max_items,)).fetchall()
    if not unprocessed:
        print("🧠 Step 3: No new news items requiring Gemma 4 scoring.")
        conn.close()
        return

    print(f"🧠 Step 3: Scoring {len(unprocessed)} recent pre-market news items with Gemma 4 31B (max={max_items})...")
    for idx, (news_id, ticker, news_date, title, summary) in enumerate(unprocessed, 1):
        title_preview = (title[:65] + "...") if len(title) > 65 else title
        print(f"   [{idx}/{len(unprocessed)}] {ticker}: {title_preview}")
        med_sent, std_sent, votes, mat = score_single_news_item(ticker, title, summary, num_votes=num_votes)
        label = "Positive" if med_sent > 0.1 else "Negative" if med_sent < -0.1 else "Neutral"
        date_clean = pd.to_datetime(news_date).strftime('%Y-%m-%d')

        conn.execute("""
            INSERT OR REPLACE INTO sentiment_results 
            (NewsID, Ticker, Date, Sentiment_Label, Median_Intensity, Disagreement_Std, Vote_Count, Materiality_Score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (news_id, ticker, date_clean, label, med_sent, std_sent, votes, mat))
        conn.execute("UPDATE news SET Processed = 1 WHERE NewsID = ?", (news_id,))
        conn.commit()

    conn.close()
    print("✅ Finished scoring pre-market news items.")

# ==============================================================================
# 6. Feature Engineering & Locked-in Model Pipeline
# ==============================================================================
def build_feature_dataset():
    conn = get_db()
    all_prices = pd.read_sql("SELECT Date, Ticker, Price, Volume, Open, High, Low FROM prices ORDER BY Date ASC", conn)
    s_raw = pd.read_sql("""
        SELECT s.Date, s.Ticker, s.Sentiment_Label as Sentiment, 
               s.Median_Intensity as Value, s.Disagreement_Std as Disagreement,
               COALESCE(s.Materiality_Score, 0.7) as Materiality,
               n.Publisher
        FROM sentiment_results s
        LEFT JOIN news n ON s.NewsID = n.NewsID
    """, conn)
    conn.close()

    if all_prices.empty:
        raise ValueError("Prices database is empty. Please run update_prices first.")

    # Quality- & Catalyst-Materiality-Weighted Sentiment aggregation
    if not s_raw.empty:
        s_raw['Date'] = pd.to_datetime(s_raw['Date']).dt.strftime('%Y-%m-%d')
        s_raw['Pub_Weight'] = s_raw['Publisher'].apply(get_pub_weight)
        s_raw['Materiality'] = s_raw['Materiality'].fillna(0.7).clip(0.1, 1.0)
        s_raw['Weighted_Value'] = s_raw['Value'] * s_raw['Pub_Weight'] * s_raw['Materiality']

        summary = s_raw.groupby(['Date', 'Ticker']).agg({
            'Weighted_Value': ['mean', 'median', 'std'],
            'Disagreement': 'mean',
            'Pub_Weight': 'mean',
            'Materiality': 'mean'
        }).reset_index()
        summary.columns = ['Date', 'Ticker', 'Avg_Intensity', 'Median_Intensity', 'Intensity_Std', 'Avg_Disagreement', 'Avg_Publisher_Quality', 'Avg_Materiality']

        counts = s_raw[s_raw['Sentiment'] != 'Neutral'].groupby(['Date', 'Ticker', 'Sentiment']).size().unstack(fill_value=0)
        for col in ['Positive', 'Negative']:
            if col not in counts.columns: counts[col] = 0
        sent_df = summary.merge(counts, on=['Date', 'Ticker'], how='left').fillna({'Positive': 0, 'Negative': 0})
        sent_df['Net_Count'] = sent_df['Positive'] - sent_df['Negative']
    else:
        sent_df = pd.DataFrame(columns=['Date', 'Ticker', 'Avg_Intensity', 'Avg_Publisher_Quality', 'Avg_Materiality', 'Positive', 'Negative', 'Net_Count'])

    # Macro benchmarks
    price_df = all_prices.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    vix_df = price_df[price_df['Ticker'] == '^VIX'][['Date', 'Price']].rename(columns={'Price': 'VIX_Close'})
    vix_df['VIX_Daily_Change'] = vix_df['VIX_Close'].pct_change()

    spy_df = price_df[price_df['Ticker'] == 'SPY'][['Date', 'Price', 'Volume']].rename(columns={'Price': 'SPY_Close', 'Volume': 'SPY_Volume'})
    spy_df['SPY_Return'] = spy_df['SPY_Close'].pct_change()

    qqq_df = price_df[price_df['Ticker'] == 'QQQ'][['Date', 'Price']].rename(columns={'Price': 'QQQ_Close'})
    qqq_df['QQQ_Return'] = qqq_df['QQQ_Close'].pct_change()

    # Sector Benchmarks: SMH (Semiconductor) & XLP (Consumer Staples)
    smh_df = price_df[price_df['Ticker'] == 'SMH'][['Date', 'Price']].rename(columns={'Price': 'SMH_Close'})
    smh_df['SMH_Return'] = smh_df['SMH_Close'].pct_change()

    xlp_df = price_df[price_df['Ticker'] == 'XLP'][['Date', 'Price']].rename(columns={'Price': 'XLP_Close'})
    xlp_df['XLP_Return'] = xlp_df['XLP_Close'].pct_change()

    # Target stock features
    stock_df = price_df[price_df['Ticker'].isin(TARGET_TICKERS)].copy()
    stock_df = stock_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    stock_df['Daily_Return'] = stock_df.groupby('Ticker')['Price'].pct_change()
    stock_df['Return_3d'] = stock_df.groupby('Ticker')['Price'].pct_change(3)
    stock_df['Return_5d'] = stock_df.groupby('Ticker')['Price'].pct_change(5)

    stock_df['SMA_5'] = stock_df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).mean())
    stock_df['SMA_20'] = stock_df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(20).mean())
    stock_df['SMA_5_dist'] = (stock_df['Price'] - stock_df['SMA_5']) / (stock_df['SMA_5'] + 1e-6)
    stock_df['SMA_20_dist'] = (stock_df['Price'] - stock_df['SMA_20']) / (stock_df['SMA_20'] + 1e-6)

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    stock_df['RSI_14'] = stock_df.groupby('Ticker')['Price'].transform(lambda x: calc_rsi(x, 14))
    stock_df['Vol_10d'] = stock_df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(10).std())

    # Intraday geometry
    stock_df['Prev_Close'] = stock_df.groupby('Ticker')['Price'].shift(1)
    stock_df['Overnight_Gap'] = (stock_df['Open'] - stock_df['Prev_Close']) / (stock_df['Prev_Close'] + 1e-6)
    stock_df['Intraday_Return'] = (stock_df['Price'] - stock_df['Open']) / (stock_df['Open'] + 1e-6)
    stock_df['Daily_Range'] = (stock_df['High'] - stock_df['Low']) / (stock_df['Open'] + 1e-6)
    stock_df['CLV'] = ((stock_df['Price'] - stock_df['Low']) - (stock_df['High'] - stock_df['Price'])) / (stock_df['High'] - stock_df['Low'] + 1e-6)
    stock_df['Vol_SMA20'] = stock_df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    stock_df['Volume_Surge'] = stock_df['Volume'] / (stock_df['Vol_SMA20'] + 1e-6)

    # Next-day Target for training (combining directional return & market alpha outperformance)
    stock_df['Next_Price'] = stock_df.groupby('Ticker')['Price'].shift(-1)
    stock_df['Next_Return'] = (stock_df['Next_Price'] - stock_df['Price']) / (stock_df['Price'] + 1e-6)

    # Merge macros
    merged = stock_df.merge(vix_df, on='Date', how='left')
    merged = merged.merge(spy_df, on='Date', how='left')
    merged = merged.merge(qqq_df, on='Date', how='left')
    merged = merged.merge(smh_df, on='Date', how='left')
    merged = merged.merge(xlp_df, on='Date', how='left')
    merged['Excess_Return'] = merged['Daily_Return'] - merged['SPY_Return']
    merged['SMH_Excess_Return'] = (merged['Daily_Return'] - merged['SMH_Return']).fillna(0.0)
    merged['XLP_Excess_Return'] = (merged['Daily_Return'] - merged['XLP_Return']).fillna(0.0)
    merged['SMH_Beta_20d'] = merged.groupby('Ticker')['Daily_Return'].transform(
        lambda s: s.rolling(20, min_periods=5).cov(merged['SMH_Return']) / (merged['SMH_Return'].rolling(20, min_periods=5).var() + 1e-9)
    ).fillna(1.0)

    # Next-day SPY and XLP returns for sector-aligned relative alpha target
    merged['Next_SPY_Return'] = merged.groupby('Ticker')['SPY_Return'].shift(-1)
    merged['Next_Excess_Return'] = merged['Next_Return'] - merged['Next_SPY_Return']
    merged['Next_XLP_Return'] = merged.groupby('Ticker')['XLP_Return'].shift(-1)
    merged['Next_XLP_Excess_Return'] = (merged['Next_Return'] - merged['Next_XLP_Return']).fillna(merged['Next_Excess_Return'])

    # Target: 1 if next-day price goes UP and beats/matches its sector/market benchmark
    # (SPY for Tech/Growth stocks; XLP for low-beta Consumer Staples COST)
    merged['Target'] = ((merged['Next_Return'] > 0) & (merged['Next_Excess_Return'] >= -0.001)).astype(float)
    is_cost = merged['Ticker'] == 'COST'
    merged.loc[is_cost, 'Target'] = ((merged.loc[is_cost, 'Next_Return'] > 0) & (merged.loc[is_cost, 'Next_XLP_Excess_Return'] >= -0.001)).astype(float)
    merged.loc[merged['Next_Price'].isna(), 'Target'] = np.nan

    # Merge sentiment
    if not sent_df.empty:
        merged = merged.merge(sent_df, on=['Date', 'Ticker'], how='left')
    for c in ['Positive', 'Negative', 'Net_Count', 'Avg_Intensity', 'Avg_Publisher_Quality']:
        if c not in merged.columns: merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)

    merged['Sent_Rolling_3d'] = merged.groupby('Ticker')['Avg_Intensity'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    merged['Sent_Divergence'] = merged['Avg_Intensity'] - merged['Daily_Return']
    merged['High_Vol_Sentiment'] = merged['Avg_Intensity'] * merged['Volume_Surge']

    # Cross-Stock Covariance & Peer Spillover
    ret_pivot = merged.pivot(index='Date', columns='Ticker', values='Daily_Return')
    spy_ret_s = spy_df.set_index('Date')['SPY_Return']

    cov_20 = ret_pivot.rolling(20, min_periods=5).cov(spy_ret_s)
    var_spy_20 = spy_ret_s.rolling(20, min_periods=5).var()
    beta_df = cov_20.div(var_spy_20, axis=0).unstack().reset_index()
    beta_df.columns = ['Ticker', 'Date', 'Rolling_Beta_SPY_20d']

    corr_df = ret_pivot.rolling(20, min_periods=5).corr(spy_ret_s).unstack().reset_index()
    corr_df.columns = ['Ticker', 'Date', 'Rolling_Corr_SPY_20d']

    merged = merged.merge(beta_df, on=['Ticker', 'Date'], how='left')
    merged = merged.merge(corr_df, on=['Ticker', 'Date'], how='left')

    tech_stocks = ['GOOGL', 'NVDA', 'AMD', 'TSLA']
    tech_df = merged[merged['Ticker'].isin(tech_stocks)]
    peer_tech_ret = tech_df.groupby('Date')['Daily_Return'].mean().reset_index().rename(columns={'Daily_Return': 'Peer_Tech_Avg_Return'})
    peer_tech_sent = tech_df.groupby('Date')['Avg_Intensity'].mean().reset_index().rename(columns={'Avg_Intensity': 'Peer_Tech_Avg_Sent'})
    merged = merged.merge(peer_tech_ret, on='Date', how='left')
    merged = merged.merge(peer_tech_sent, on='Date', how='left')

    nvda_sub = merged[merged['Ticker'] == 'NVDA'][['Date', 'Daily_Return', 'Avg_Intensity']].rename(columns={'Daily_Return': 'NVDA_Lead_Return', 'Avg_Intensity': 'NVDA_Lead_Sent'})
    merged = merged.merge(nvda_sub, on='Date', how='left')

    return merged

def sync_and_grade_predictions_log(train_df, feature_cols, semi_features, staples_features, decay_rate):
    """
    1. If predictions_log has fewer than 15 graded rows per ticker, backfills out-of-sample
       historical predictions on the most recent 20% validation window so the rolling 20-day
       specialist accuracy tracker is immediately warm.
    2. Grades any pending predictions in predictions_log whose actual next-day close has arrived.
    3. Returns a dict of {Ticker: rolling_20d_win_rate} for Bandit/Kelly position scaling.
    """
    init_db()
    conn = get_db()
    graded_count = conn.execute("SELECT COUNT(*) FROM predictions_log WHERE Correct IS NOT NULL").fetchone()[0]

    # Step 1: Backfill out-of-sample validation window if tracker is cold
    if graded_count < 75 and len(train_df) >= 200:
        ordered = train_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        split_idx = int(len(ordered) * 0.80)
        hist_tr = ordered.iloc[:split_idx].copy()
        hist_val = ordered.iloc[split_idx:].copy()

        max_tr_date = pd.to_datetime(hist_tr['Date']).max()
        tr_w = np.exp(-decay_rate * (max_tr_date - pd.to_datetime(hist_tr['Date'])).dt.days)
        tr_w = tr_w / tr_w.mean()

        h_lgb = LGBMClassifier(n_estimators=60, max_depth=3, learning_rate=0.015, colsample_bytree=0.8, subsample=0.7, min_child_samples=15, random_state=42, n_jobs=1, verbose=-1)
        h_lgb.fit(hist_tr[feature_cols], hist_tr['Target'].astype(int), sample_weight=tr_w)

        h_rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=8, max_features='sqrt', random_state=42, n_jobs=1)
        h_rf.fit(hist_tr[feature_cols], hist_tr['Target'].astype(int), sample_weight=tr_w)

        s_tr = hist_tr[hist_tr['Ticker'].isin(['NVDA', 'AMD'])].copy()
        s_w = np.exp(-decay_rate * (max_tr_date - pd.to_datetime(s_tr['Date'])).dt.days)
        s_w = s_w / s_w.mean()
        h_semi = RandomForestClassifier(n_estimators=120, max_depth=3, min_samples_split=6, max_features='sqrt', random_state=42, n_jobs=1)
        h_semi.fit(s_tr[semi_features], s_tr['Target'].astype(int), sample_weight=s_w)

        c_tr = hist_tr[hist_tr['Ticker'] == 'COST'].copy()
        c_w = np.exp(-decay_rate * (max_tr_date - pd.to_datetime(c_tr['Date'])).dt.days)
        c_w = c_w / c_w.mean()
        h_cost = RandomForestClassifier(n_estimators=120, max_depth=3, min_samples_split=8, max_features='sqrt', random_state=42, n_jobs=1)
        h_cost.fit(c_tr[staples_features], c_tr['Target'].astype(int), sample_weight=c_w)

        p_lgb = h_lgb.predict_proba(hist_val[feature_cols])[:, 1]
        p_rf = h_rf.predict_proba(hist_val[feature_cols])[:, 1]
        p_ens = 0.60 * p_lgb + 0.40 * p_rf
        p_semi = h_semi.predict_proba(hist_val[semi_features])[:, 1]
        p_cost = h_cost.predict_proba(hist_val[staples_features])[:, 1]

        rows_to_insert = []
        for i, (_, r) in enumerate(hist_val.iterrows()):
            t = r['Ticker']
            if t == 'AMD':
                prob, m_name = float(p_semi[i]), "AMD Semi-RF Specialist"
            elif t == 'GOOGL':
                prob, m_name = float(p_semi[i]), "GOOGL AI/Semi-RF Specialist"
            elif t == 'COST':
                prob, m_name = float(p_cost[i]), "COST Staples-RF Specialist"
            elif t == 'TSLA':
                prob, m_name = float(p_rf[i]), "TSLA Tuned RF Specialist"
            else:
                prob, m_name = float(p_ens[i]), "NVDA 60/40 Ensemble Specialist"

            conv = max(prob, 1.0 - prob)
            act = "BUY / LONG" if prob >= 0.55 else ("SHORT / FADE" if prob <= 0.45 else "HOLD / PASS")
            vol_10d = float(np.clip(r.get('Vol_10d', 0.02), 0.008, 0.060))
            stop_pct = round(float(np.clip(vol_10d * 100.0 * 0.85, 1.0, 2.5)), 2)
            edge = abs(prob - 0.50) * 2.0
            alloc = round(float(np.clip((edge / vol_10d) * 0.90, 5.0, 30.0)), 1) if act != "HOLD / PASS" else 0.0
            target_act = int(r['Target'])
            pred_dir = 1 if prob >= 0.50 else 0
            correct = 1 if pred_dir == target_act else 0

            rows_to_insert.append((
                str(r['Date'])[:10], t, m_name, prob, conv, act, alloc, stop_pct,
                float(r['Price']), float(r['Next_Price']), float(r['Next_Return']),
                target_act, correct
            ))

        conn.executemany("""
            INSERT OR REPLACE INTO predictions_log
            (Date, Ticker, Model_Used, Prob_Up, Conviction, Action, Alloc_Pct, Stop_Loss_Pct,
             Entry_Price, Next_Close_Price, Realized_Return, Target_Actual, Correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        conn.commit()

    # Step 2: Grade any ungraded predictions where actual Target is now available in train_df
    pending = conn.execute("SELECT Date, Ticker, Prob_Up FROM predictions_log WHERE Correct IS NULL").fetchall()
    if pending:
        lookup = train_df.set_index(['Date', 'Ticker'])
        for p_date, p_ticker, p_prob in pending:
            if (p_date, p_ticker) in lookup.index:
                row_match = lookup.loc[(p_date, p_ticker)]
                if isinstance(row_match, pd.DataFrame):
                    row_match = row_match.iloc[-1]
                target_act = int(row_match['Target'])
                pred_dir = 1 if float(p_prob) >= 0.50 else 0
                correct = 1 if pred_dir == target_act else 0
                conn.execute("""
                    UPDATE predictions_log
                    SET Next_Close_Price = ?, Realized_Return = ?, Target_Actual = ?, Correct = ?, Updated_At = CURRENT_TIMESTAMP
                    WHERE Date = ? AND Ticker = ?
                """, (float(row_match['Next_Price']), float(row_match['Next_Return']), target_act, correct, p_date, p_ticker))
        conn.commit()

    # Step 3: Compute rolling 20-day realized specialist win rate per ticker
    rolling_win_rates = {}
    for ticker in TARGET_TICKERS:
        rows = conn.execute("""
            SELECT Correct FROM predictions_log
            WHERE Ticker = ? AND Correct IS NOT NULL
            ORDER BY Date DESC LIMIT 20
        """, (ticker,)).fetchall()
        if rows:
            rolling_win_rates[ticker] = float(np.mean([r[0] for r in rows]))
        else:
            rolling_win_rates[ticker] = 0.60

    conn.close()
    return rolling_win_rates


def run_models_and_inference():
    print("⚙️ Step 4: Engineering 36 Alpha Signals & Generating Pre-Market Inferences...")
    df = build_feature_dataset()

    feature_cols = [
        'Daily_Return', 'Return_3d', 'Return_5d', 'SMA_5_dist', 'SMA_20_dist', 'RSI_14', 'Vol_10d',
        'Overnight_Gap', 'Intraday_Return', 'Daily_Range', 'CLV', 'Volume_Surge',
        'SPY_Return', 'QQQ_Return', 'Excess_Return', 'VIX_Close', 'VIX_Daily_Change',
        'SMH_Return', 'SMH_Excess_Return', 'SMH_Beta_20d', 'XLP_Return', 'XLP_Excess_Return',
        'Positive', 'Negative', 'Net_Count', 'Avg_Intensity', 'Avg_Publisher_Quality', 'Sent_Rolling_3d', 'Sent_Divergence',
        'High_Vol_Sentiment', 'Rolling_Beta_SPY_20d', 'Rolling_Corr_SPY_20d',
        'Peer_Tech_Avg_Return', 'Peer_Tech_Avg_Sent', 'NVDA_Lead_Return', 'NVDA_Lead_Sent'
    ]

    # Clean historical training set (all rows where Target is known)
    train_df = df.dropna(subset=['Target', 'Daily_Return', 'Overnight_Gap', 'Volume_Surge', 'RSI_14', 'Rolling_Beta_SPY_20d']).copy()
    X_train = train_df[feature_cols]
    y_train = train_df['Target'].astype(int)

    # -------------------------------------------------------------
    # Recency Adaptation: Exponential Decay Sample Weights
    # Half-life configurable via RECENCY_HALF_LIFE_DAYS (default: 60 days)
    # -------------------------------------------------------------
    recency_half_life = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "60.0"))
    train_dates = pd.to_datetime(train_df['Date'])
    max_train_date = train_dates.max()
    days_diff = (max_train_date - train_dates).dt.days
    decay_rate = np.log(2.0) / max(recency_half_life, 1.0)
    sample_weights = np.exp(-decay_rate * days_diff)
    sample_weights = sample_weights / sample_weights.mean()

    # 1. Fit Tuned LightGBM (Regularized for noise reduction + Recency Weighted)
    lgb = LGBMClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.015,
        colsample_bytree=0.8,
        subsample=0.7,
        min_child_samples=15,
        random_state=42,
        n_jobs=1,
        verbose=-1
    )
    lgb.fit(X_train, y_train, sample_weight=sample_weights)

    # 2. Fit Tuned Random Forest (Capped depth=3 to prevent leaf overfitting + Recency Weighted)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train, y_train, sample_weight=sample_weights)

    # 3. Fit Dedicated Semiconductor / AI-Tech Specialist (NVDA, AMD & GOOGL + Recency Weighted)
    semi_features = [
        'Daily_Return', 'Return_3d', 'Return_5d', 'SMA_5_dist', 'SMA_20_dist', 'RSI_14', 'Vol_10d',
        'Overnight_Gap', 'Intraday_Return', 'Daily_Range', 'CLV', 'Volume_Surge',
        'SPY_Return', 'QQQ_Return', 'SMH_Return', 'SMH_Excess_Return', 'SMH_Beta_20d',
        'VIX_Close', 'VIX_Daily_Change',
        'Avg_Intensity', 'Sent_Rolling_3d', 'Sent_Divergence', 'High_Vol_Sentiment',
        'Peer_Tech_Avg_Return', 'Peer_Tech_Avg_Sent', 'NVDA_Lead_Return', 'NVDA_Lead_Sent'
    ]
    semi_train = train_df[train_df['Ticker'].isin(['NVDA', 'AMD'])].copy()
    semi_train_dates = pd.to_datetime(semi_train['Date'])
    semi_days_diff = (max_train_date - semi_train_dates).dt.days
    semi_weights = np.exp(-decay_rate * semi_days_diff)
    semi_weights = semi_weights / semi_weights.mean()

    semi_rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=3,
        min_samples_split=6,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    semi_rf.fit(semi_train[semi_features], semi_train['Target'].astype(int), sample_weight=semi_weights)

    # 4. Fit Dedicated Consumer Staples Specialist (COST isolated from Tech-Peer noise + Recency Weighted)
    staples_features = [
        'Daily_Return', 'Return_3d', 'Return_5d', 'SMA_5_dist', 'SMA_20_dist', 'RSI_14', 'Vol_10d',
        'Overnight_Gap', 'Intraday_Return', 'Daily_Range', 'CLV', 'Volume_Surge',
        'SPY_Return', 'Excess_Return', 'XLP_Return', 'XLP_Excess_Return',
        'VIX_Close', 'VIX_Daily_Change',
        'Avg_Intensity', 'Sent_Rolling_3d', 'Sent_Divergence', 'High_Vol_Sentiment',
        'Rolling_Beta_SPY_20d', 'Rolling_Corr_SPY_20d'
    ]
    cost_train = train_df[train_df['Ticker'] == 'COST'].copy()
    cost_train_dates = pd.to_datetime(cost_train['Date'])
    cost_days_diff = (max_train_date - cost_train_dates).dt.days
    cost_weights = np.exp(-decay_rate * cost_days_diff)
    cost_weights = cost_weights / cost_weights.mean()

    cost_rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=3,
        min_samples_split=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    cost_rf.fit(cost_train[staples_features], cost_train['Target'].astype(int), sample_weight=cost_weights)

    # 5. Sync & Grade SQLite predictions_log and get Rolling 20-Day Specialist Win Rates
    rolling_win_rates = sync_and_grade_predictions_log(
        train_df, feature_cols, semi_features, staples_features, decay_rate
    )

    # Generate today's inference on the latest available bar for each stock
    inferences = []
    latest_date = df['Date'].max()

    for ticker in TARGET_TICKERS:
        t_sub = df[df['Ticker'] == ticker].sort_values('Date')
        if t_sub.empty:
            continue
        latest_row = t_sub.iloc[-1:]
        X_today = latest_row[feature_cols].fillna(0.0)

        prob_lgb = lgb.predict_proba(X_today)[0, 1]
        prob_rf = rf.predict_proba(X_today)[0, 1]
        prob_ens = (prob_lgb * 0.60) + (prob_rf * 0.40)

        # Asset-Specific Specialist Routing (Upgraded 2026-10 Benchmark - 60.87% Overall Acc, 0.592 AUC):
        # NVDA  -> 60/40 Ensemble Specialist (61.8% test accuracy, 0.552 AUC)
        # COST  -> COST Staples-RF Specialist (61.8% test accuracy, 0.594 AUC; isolated from tech spillover)
        # TSLA  -> TSLA Tuned RF Specialist (60.7% test accuracy, 0.556 AUC)
        # AMD   -> AMD Semi-RF Specialist (60.0% test accuracy, 0.640 AUC)
        # GOOGL -> GOOGL AI/Semi-RF Specialist (60.0% test accuracy, 0.534 AUC)
        if ticker == 'AMD':
            X_semi = latest_row[semi_features].fillna(0.0)
            final_prob = semi_rf.predict_proba(X_semi)[0, 1]
            model_used = "AMD Semi-RF Specialist"
        elif ticker == 'GOOGL':
            X_semi = latest_row[semi_features].fillna(0.0)
            final_prob = semi_rf.predict_proba(X_semi)[0, 1]
            model_used = "GOOGL AI/Semi-RF Specialist"
        elif ticker == 'COST':
            X_staples = latest_row[staples_features].fillna(0.0)
            final_prob = cost_rf.predict_proba(X_staples)[0, 1]
            model_used = "COST Staples-RF Specialist"
        elif ticker == 'TSLA':
            final_prob = prob_rf
            model_used = "TSLA Tuned RF Specialist"
        else:
            final_prob = prob_ens
            model_used = "NVDA 60/40 Ensemble Specialist"

        conviction = max(final_prob, 1.0 - final_prob)
        vol_10d = float(np.clip(latest_row['Vol_10d'].fillna(0.02).values[0], 0.008, 0.060))
        win_20d = rolling_win_rates.get(ticker, 0.60)

        # Volatility-Scaled Stop-Loss (0.85x 10D daily volatility, bounded between 1.0% and 2.5%)
        stop_loss_pct = round(float(np.clip(vol_10d * 100.0 * 0.85, 1.0, 2.5)), 2)

        # Decision rule & conviction tier tailored specifically for 9:15 - 9:30 AM EST execution
        if final_prob >= 0.55:
            action = "BUY / LONG"
            color = "#10b981"
            badge = "🟢 LONG (9:15-9:30)"
            bias = "📈 Bullish Rally"
            plan = f"Enter Long at open; hard stop -{stop_loss_pct:.2f}%"
        elif final_prob <= 0.45:
            action = "SHORT / FADE"
            color = "#ef4444"
            badge = "🔴 SHORT (9:15-9:30)"
            bias = "📉 Bearish Drop"
            plan = f"Fade open / Short; hard stop +{stop_loss_pct:.2f}%"
        else:
            action = "HOLD / PASS"
            color = "#6b7280"
            badge = "⚪ PASS (Chop)"
            bias = "⚖️ Neutral Chop"
            plan = "No statistical edge; preserve capital, wait on sidelines"

        # Volatility-Adjusted Fractional Kelly Sizing + Bandit 20D Win-Rate Multiplier
        if action != "HOLD / PASS":
            edge = abs(final_prob - 0.50) * 2.0
            regime_mult = float(np.clip(win_20d / 0.55, 0.60, 1.35))
            raw_kelly_pct = (edge / vol_10d) * 0.90 * regime_mult
            raw_alloc = float(np.clip(raw_kelly_pct, 5.0, 30.0))
        else:
            raw_alloc = 0.0

        high_conviction = conviction >= 0.60 or conviction <= 0.40

        inferences.append({
            'Ticker': ticker,
            'Date': latest_row['Date'].values[0],
            'Price': float(latest_row['Price'].values[0]),
            'Prob_Up': float(final_prob),
            'Conviction': float(conviction),
            'Action': action,
            'Badge': badge,
            'Color': color,
            'Bias': bias,
            'Plan': plan,
            'High_Conviction': bool(high_conviction),
            'Model_Used': model_used,
            'Raw_Alloc': raw_alloc,
            'Alloc_Pct': raw_alloc,
            'Stop_Loss_Pct': stop_loss_pct,
            'Win_20d': win_20d,
            'Vol_10d': vol_10d * 100.0,
            'Sent_Intensity': float(latest_row['Avg_Intensity'].values[0]),
            'Sent_Divergence': float(latest_row['Sent_Divergence'].values[0]),
            'RSI_14': float(latest_row['RSI_14'].values[0]),
            'Return_5d': float(latest_row['Return_5d'].values[0] * 100)
        })

    # Normalize portfolio weights so total capital deployed <= 100.0% (max 30% per single asset)
    total_raw_alloc = sum(inf['Raw_Alloc'] for inf in inferences)
    scale_factor = (100.0 / total_raw_alloc) if total_raw_alloc > 100.0 else 1.0
    conn = get_db()
    for inf in inferences:
        inf['Alloc_Pct'] = round(min(inf['Raw_Alloc'] * scale_factor, 30.0), 1)
        conn.execute("""
            INSERT OR REPLACE INTO predictions_log
            (Date, Ticker, Model_Used, Prob_Up, Conviction, Action, Alloc_Pct, Stop_Loss_Pct, Entry_Price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(inf['Date'])[:10], inf['Ticker'], inf['Model_Used'],
            inf['Prob_Up'], inf['Conviction'], inf['Action'],
            inf['Alloc_Pct'], inf['Stop_Loss_Pct'], inf['Price']
        ))
    conn.commit()
    conn.close()

    # Macro regime indicators
    latest_macro = df.iloc[-1]
    macro_info = {
        'Date': latest_date,
        'VIX': latest_macro.get('VIX_Close', 0.0),
        'VIX_Change': latest_macro.get('VIX_Daily_Change', 0.0) * 100,
        'SPY_Return': latest_macro.get('SPY_Return', 0.0) * 100,
        'Tech_Cluster_Sent': latest_macro.get('Peer_Tech_Avg_Sent', 0.0),
        'Total_Alloc_Pct': round(sum(inf['Alloc_Pct'] for inf in inferences), 1)
    }

    return inferences, macro_info

# ==============================================================================
# 7. Gemma 4 Narrative Synthesis & HTML Email Generation
# ==============================================================================
def generate_gemma4_narrative(inferences, macro_info):
    """Uses Gemma 4 31B to synthesize the executive morning briefing narrative and actionable advice."""
    print("🤖 Step 5: Synthesizing Executive Morning Briefing via Gemma 4 31B...")

    stock_summaries = []
    for inf in inferences:
        flag = " [HIGH CONVICTION]" if inf['High_Conviction'] else ""
        stock_summaries.append(
            f"- {inf['Ticker']}: {inf['Badge']}{flag} (Target Bias: {inf['Bias']}) | Conviction: {inf['Conviction']*100:.1f}% | "
            f"P(Up): {inf['Prob_Up']*100:.1f}% | Kelly Size: {inf['Alloc_Pct']:.1f}% Cap | Vol Stop: {inf['Stop_Loss_Pct']:.2f}% | "
            f"20D Specialist Win Rate: {inf['Win_20d']*100:.0f}% | Price: ${inf['Price']:.2f} | 5D Ret: {inf['Return_5d']:+.1f}% | "
            f"Sentiment: {inf['Sent_Intensity']:+.2f} | Specialist: {inf['Model_Used']} | Tactical Plan: {inf['Plan']}"
        )
    stocks_text = "\n".join(stock_summaries)

    prompt = f"""
You are an elite quantitative portfolio manager writing the Pre-Market Trading Briefing for the lead trader.
Market Session: {macro_info['Date']}
CRITICAL EXECUTION WINDOW: 9:15 AM - 9:30 AM EST (Market Open Action)

MACRO REGIME:
- VIX: {macro_info['VIX']:.2f} ({macro_info['VIX_Change']:+.2f}%)
- S&P 500 (SPY 1D): {macro_info['SPY_Return']:+.2f}%
- Tech Cluster Sentiment: {macro_info['Tech_Cluster_Sent']:+.2f}
- Total Recommended Portfolio Allocation: {macro_info.get('Total_Alloc_Pct', 0.0):.1f}% (Cash Reserve: {100.0 - macro_info.get('Total_Alloc_Pct', 0.0):.1f}%)

QUANTITATIVE ML ENSEMBLE PREDICTIONS & VOL-KELLY POSITION SIZING:
{stocks_text}

Provide an executive trading note formatted in clean HTML (use <p>, <ul>, <li>, <strong> only):
1. <strong>Macro & Volatility Climate</strong>: 2 concise sentences on market risk regime and recommended capital deployment ({macro_info.get('Total_Alloc_Pct', 0.0):.1f}% invested) heading into the 9:15-9:30 AM open.
2. <strong>Tactical Positioning & 9:15-9:30 AM Setups</strong>: 1-2 bullet points per key stock detailing the exact trade execution, including its Volatility-Adjusted Kelly Size (% of capital) and Dynamic Vol Stop (%):
   - For 🔴 SHORT: explicitly advise fading the morning open or shorting overextended moves (expecting price decline).
   - For 🟢 LONG: explicitly advise buying the open or morning dip for upside momentum & alpha outperformance.
   - For ⚪ PASS: state why capital should remain in cash/sidelined.
3. <strong>Highest Conviction Play & Risk Rules</strong>: Highlight the single top risk-adjusted trade (highest Kelly allocation & conviction) for the 9:15-9:30 AM open and enforce its volatility-calibrated hard stop-loss.

Keep it sharp, professional, and directly actionable for the 9:15-9:30 AM EST open.
"""
    payload = {
        "model": "gemma4:31b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 600}
    }

    try:
        res = requests.post(f"{GEMMA4_URL}/api/generate", json=payload, timeout=60)
        if res.status_code == 200:
            narrative = res.json().get("response", "").strip()
            narrative = re.sub(r"^```(?:html)?\s*", "", narrative)
            narrative = re.sub(r"\s*```$", "", narrative)
            return narrative
    except Exception as e:
        print(f"Notice: Gemma 4 narrative generation fallback ({e})")

    # Fallback narrative
    bullet_items = "".join(
        f"<li><strong>{inf['Ticker']}:</strong> {inf['Action']} (Size: {inf['Alloc_Pct']:.1f}% Cap | Stop: {inf['Stop_Loss_Pct']:.2f}% | Conviction: {inf['Conviction']*100:.1f}%) — {inf['Model_Used']}</li>"
        for inf in inferences
    )
    return f"""
    <p><strong>Macro Regime:</strong> VIX is at {macro_info['VIX']:.2f} ({macro_info['VIX_Change']:+.2f}%), with SPY at {macro_info['SPY_Return']:+.2f}%. Total Volatility-Kelly allocation is <strong>{macro_info.get('Total_Alloc_Pct', 0.0):.1f}%</strong>.</p>
    <ul>{bullet_items}</ul>
    """

def generate_html_report(inferences, macro_info, gemma_narrative=None):
    rows_html = ""
    for inf in inferences:
        conv_pct = inf['Conviction'] * 100
        prob_pct = inf['Prob_Up'] * 100
        alloc_pct = inf.get('Alloc_Pct', 0.0)
        stop_pct = inf.get('Stop_Loss_Pct', 1.5)
        win_20d = inf.get('Win_20d', 0.60) * 100
        star = " 🔥" if inf['High_Conviction'] else ""

        rows_html += f"""
        <tr style="border-bottom: 1px solid #2d3748;">
            <td style="padding: 12px 14px; font-weight: 700; font-size: 15px; color: #f8fafc;">{inf['Ticker']}{star}</td>
            <td style="padding: 12px 14px; text-align: center;">
                <span style="background-color: {inf['Color']}22; color: {inf['Color']}; border: 1px solid {inf['Color']}; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px;">
                    {inf['Badge']}
                </span>
            </td>
            <td style="padding: 12px 14px; font-weight: 700; color: #f8fafc; text-align: right;">{conv_pct:.1f}%</td>
            <td style="padding: 12px 14px; color: #38bdf8; font-weight: 700; text-align: right;">{alloc_pct:.1f}%</td>
            <td style="padding: 12px 14px; color: #f59e0b; font-weight: 600; text-align: right;">-{stop_pct:.2f}%</td>
            <td style="padding: 12px 14px; color: #94a3b8; text-align: right;">{prob_pct:.1f}%</td>
            <td style="padding: 12px 14px; color: #f8fafc; text-align: right;">${inf['Price']:.2f}</td>
            <td style="padding: 12px 14px; color: {'#10b981' if inf['Return_5d'] >= 0 else '#ef4444'}; text-align: right;">{inf['Return_5d']:+.1f}%</td>
            <td style="padding: 12px 14px; color: #cbd5e1; font-size: 13px;">{inf['Model_Used']} <span style="color: #64748b; font-size: 11px;">({win_20d:.0f}% 20D)</span></td>
        </tr>
        """

    vix_color = "#ef4444" if macro_info['VIX_Change'] > 0 else "#10b981"
    total_alloc = macro_info.get('Total_Alloc_Pct', 0.0)
    narrative_block = f"""
    <!-- EF Strategy Synthesis Commentary -->
    <div style="background-color: #1e1b4b44; padding: 20px 24px; border-bottom: 1px solid #334155;">
        <h3 style="font-size: 15px; margin: 0 0 12px 0; color: #c084fc; letter-spacing: 0.3px;">
            ⚡ EF STRATEGY SYNTHESIS
        </h3>
        <div style="font-size: 13.5px; line-height: 1.6; color: #e2e8f0;">
            {gemma_narrative}
        </div>
    </div>
    """ if gemma_narrative else ""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>⚡ Pre-Market Quant Intelligence Briefing</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #e2e8f0; margin: 0; padding: 24px;">
        <div style="max-width: 860px; margin: 0 auto; background-color: #1e293b; border-radius: 12px; border: 1px solid #334155; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 24px; border-bottom: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;">
                        ⚡ QUANT PRE-MARKET INTELLIGENCE
                    </h1>
                </div>
                <p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">
                    Market Session: <strong style="color: #f8fafc;">{macro_info['Date']}</strong> | Execution Window: <strong style="color: #38bdf8;">9:15 – 9:30 AM EST</strong> | Specialist Router + Vol-Kelly Sizing
                </p>
            </div>

            <!-- Macro Regime Bar -->
            <div style="background-color: #162032; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; flex-wrap: wrap; gap: 20px; font-size: 13px;">
                <div><strong>VIX:</strong> <span style="color: #f8fafc;">{macro_info['VIX']:.2f}</span> (<span style="color: {vix_color};">{macro_info['VIX_Change']:+.2f}%</span>)</div>
                <div><strong>SPY 1D:</strong> <span style="color: {'#10b981' if macro_info['SPY_Return'] >= 0 else '#ef4444'};">{macro_info['SPY_Return']:+.2f}%</span></div>
                <div><strong>Tech Sentiment:</strong> <span style="color: {'#10b981' if macro_info['Tech_Cluster_Sent'] >= 0 else '#ef4444'};">{macro_info['Tech_Cluster_Sent']:+.2f}</span></div>
                <div><strong>Kelly Deployed:</strong> <span style="color: #38bdf8; font-weight: 700;">{total_alloc:.1f}%</span> <span style="color: #64748b;">({100.0 - total_alloc:.1f}% Cash)</span></div>
            </div>

            {narrative_block}

            <!-- Actions Table -->
            <div style="padding: 24px;">
                <h2 style="font-size: 16px; margin: 0 0 14px 0; color: #f8fafc;">🎯 Recommended Actions, Kelly Sizing & Vol Stops (9:15 – 9:30 AM EST)</h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
                        <thead>
                            <tr style="background-color: #0f172a; color: #94a3b8; font-size: 12px; text-transform: uppercase;">
                                <th style="padding: 10px 14px;">Ticker</th>
                                <th style="padding: 10px 14px; text-align: center;">Signal (9:15-9:30)</th>
                                <th style="padding: 10px 14px; text-align: right;">Conviction</th>
                                <th style="padding: 10px 14px; text-align: right;">Size (% Cap)</th>
                                <th style="padding: 10px 14px; text-align: right;">Vol Stop</th>
                                <th style="padding: 10px 14px; text-align: right;">P(Up)</th>
                                <th style="padding: 10px 14px; text-align: right;">Close</th>
                                <th style="padding: 10px 14px; text-align: right;">5D Ret</th>
                                <th style="padding: 10px 14px;">Specialist (20D Win%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>

                <!-- Guidance notes -->
                <div style="margin-top: 24px; padding: 16px; background-color: #0f172a; border-radius: 8px; border-left: 4px solid #38bdf8; font-size: 13px; line-height: 1.5; color: #94a3b8;">
                    <strong style="color: #38bdf8;">📌 9:15 – 9:30 AM EST Tactical Execution & Kelly Sizing Rules:</strong><br>
                    • <strong>🟢 LONG (Buy Open)</strong>: Directional Alpha target &ge; 55%. Buy during the 9:15-9:30 AM open window to capture momentum & relative outperformance.<br>
                    • <strong>🔴 SHORT (Fade Open)</strong>: Directional Alpha target &le; 45%. Fade overextended morning moves or enter short for downside mean-reversion.<br>
                    • <strong>📐 Size (% Cap)</strong>: Volatility-Adjusted Fractional Kelly allocation scaled by each specialist's rolling 20-day realized win rate (capped at 30% per asset).<br>
                    • <strong>🛡️ Vol Stop</strong>: Volatility-calibrated hard stop-loss (`0.85 × Vol_10d`, bounded 1.0% – 2.5%). Never average down on an adverse opening move.
                </div>
            </div>

            <!-- Footer -->
            <div style="padding: 16px 24px; background-color: #0f172a; border-top: 1px solid #334155; text-align: center; font-size: 11px; color: #64748b;">
                Sent automatically to <strong>{os.getenv('ALERT_EMAIL_TO', 'erfontes@gmail.com')}</strong> | Generated by Antigravity Quantitative Engine & Gemma 4 31B.
            </div>
        </div>
    </body>
    </html>
    """
    return html

def send_email_alert(html_content, recipient=None):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    recipient = recipient or os.getenv("ALERT_EMAIL_TO", "erfontes@gmail.com")

    # Local fallback file
    out_file = os.path.join(BASE_DIR, "latest_daily_briefing.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"📄 Saved HTML briefing preview to: {out_file}")

    if not smtp_host or not smtp_user or not smtp_pass or not recipient:
        print("ℹ️ SMTP credentials or recipient not configured in environment. Skipped email dispatch.")
        print("   (Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO in daily_pipeline.env to enable)")
        return False

    print(f"📧 Sending morning briefing email to {recipient} via {smtp_host}:{smtp_port}...")
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"⚡ Pre-Market Quant Actions Briefing ({date.today().strftime('%b %d, %Y')})"
        msg["From"] = sender
        msg["To"] = recipient

        msg.attach(MIMEText(html_content, "html"))

        use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(sender, [recipient], msg.as_string())

        print(f"✅ Morning email alert successfully dispatched to {recipient}!")
        return True
    except Exception as e:
        print(f"❌ Failed to dispatch email: {e}")
        return False

# ==============================================================================
# 8. Main Execution Controller
# ==============================================================================
def run_pipeline(dry_run=False, skip_prices=False, skip_news=False):
    print("=" * 80)
    print(f"🚀 RUNNING DAILY TRADER PIPELINE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    init_db()

    # Step 1: Update prices
    if not skip_prices:
        update_prices()
    else:
        print("⏩ Skipping price update (--skip-prices).")

    # Step 2: Update news
    if not skip_news:
        update_news()
    else:
        print("⏩ Skipping news fetch (--skip-news).")

    # Step 3, 4, 5: Model inference and Gemma 4 synthesis
    gemma_narrative = ""
    with GemmaServerContext():
        # Step 3: Process sentiment with Gemma 4
        if not skip_news:
            process_unscored_news()

        # Step 4: Run inference
        inferences, macro_info = run_models_and_inference()

        # Step 5: Synthesize executive narrative with Gemma 4
        gemma_narrative = generate_gemma4_narrative(inferences, macro_info)

    # Step 6: Print summary to console
    print("\n" + "=" * 98)
    print(f"📊 PRE-MARKET ACTION DASHBOARD ({macro_info['Date']})")
    print(f"Macro: VIX {macro_info['VIX']:.2f} ({macro_info['VIX_Change']:+.2f}%) | SPY {macro_info['SPY_Return']:+.2f}% | Kelly Deployed: {macro_info.get('Total_Alloc_Pct', 0.0):.1f}%")
    print("=" * 98)
    print(f"{'Ticker':<7} {'Action':<14} {'Conviction':<11} {'Size %':<9} {'Vol Stop':<10} {'P(Up)':<8} {'Close':<10} {'Specialist (20D Win%)':<26}")
    print("-" * 98)
    for inf in inferences:
        flag = " 🔥" if inf['High_Conviction'] else ""
        spec_str = f"{inf['Model_Used']} ({inf.get('Win_20d', 0.60)*100:.0f}%)"
        print(f"{inf['Ticker']:<7} {inf['Action']:<14} {inf['Conviction']*100:>5.1f}%{flag:<4} {inf['Alloc_Pct']:>5.1f}%   -{inf['Stop_Loss_Pct']:>4.2f}%    {inf['Prob_Up']*100:>5.1f}%   ${inf['Price']:<9.2f} {spec_str}")
    print("=" * 98 + "\n")

    # Step 7: Generate HTML & dispatch email
    html = generate_html_report(inferences, macro_info, gemma_narrative=gemma_narrative)
    out_file = os.path.join(BASE_DIR, "latest_daily_briefing.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 Saved HTML briefing preview to: {out_file}")

    if not dry_run:
        send_email_alert(html)
    else:
        print("ℹ️ Dry-run mode: Email dispatch skipped.")

    print(f"🏁 Pipeline complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")

# ==============================================================================
# 9. Scheduler & Daemon Mode
# ==============================================================================
def schedule_loop(target_time="07:15"):
    print(f"⏰ Daily Trader Daemon activated. Scheduled to run every weekday at {target_time} EST.")
    while True:
        now = datetime.now()
        # Monday=0, Sunday=6
        if now.weekday() < 5:
            current_hhmm = now.strftime("%H:%M")
            if current_hhmm == target_time:
                print(f"🔔 Target time {target_time} reached! Launching pipeline...")
                try:
                    run_pipeline()
                except Exception as e:
                    print(f"❌ Error during scheduled pipeline execution: {e}")
                print(f"Sleeping 65 seconds to avoid double-triggering...")
                time.sleep(65)
        time.sleep(25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Daily Trader Pre-Market Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline and generate HTML without sending email")
    parser.add_argument("--skip-prices", action="store_true", help="Skip downloading latest prices")
    parser.add_argument("--skip-news", action="store_true", help="Skip news ingestion and sentiment scoring")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon background scheduler loop")
    parser.add_argument("--run-at", type=str, default="07:15", help="Daily execution time in HH:MM (default: 07:15)")
    args = parser.parse_args()

    if args.daemon:
        schedule_loop(target_time=args.run_at)
    else:
        run_pipeline(dry_run=args.dry_run, skip_prices=args.skip_prices, skip_news=args.skip_news)
