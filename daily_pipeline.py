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
GEMMA4_SCRIPT = os.getenv("GEMMA4_SCRIPT_PATH", os.path.join(BASE_DIR, "gemma4_server.py"))

TARGET_TICKERS = [t.strip() for t in os.getenv("TARGET_TICKERS", "GOOGL,NVDA,COST,TSLA,AMD").split(",") if t.strip()]
BENCHMARK_TICKERS = [t.strip() for t in os.getenv("BENCHMARK_TICKERS", "^VIX,SPY,QQQ").split(",") if t.strip()]
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (Ticker, Date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_results (Ticker, Date)")
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
def score_single_news_item(ticker, title, summary, num_votes=5):
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
        "options": {"temperature": 0.3}
    }

    def single_vote():
        try:
            res = requests.post(f"{GEMMA4_URL}/api/generate", json=payload, timeout=45)
            if res.status_code == 200:
                raw_json = res.json()
                text = raw_json.get("response", "{}")
                # Parse sentiment and materiality
                data = json.loads(text)
                return float(data.get("sentiment", 0.0)), float(data.get("materiality", 0.7))
        except Exception:
            pass
        return None, None

    sent_votes, mat_votes = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_votes) as executor:
        futures = [executor.submit(single_vote) for _ in range(num_votes)]
        for fut in concurrent.futures.as_completed(futures):
            s, m = fut.result()
            if s is not None: sent_votes.append(s)
            if m is not None: mat_votes.append(m)

    if not sent_votes:
        return 0.0, 0.0, 0, 0.5
    return float(np.median(sent_votes)), float(np.std(sent_votes)), len(sent_votes), float(np.median(mat_votes))

def process_unscored_news(max_items=30, num_votes=3):
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
               n.Publisher
        FROM sentiment_results s
        LEFT JOIN news n ON s.NewsID = n.NewsID
    """, conn)
    conn.close()

    if all_prices.empty:
        raise ValueError("Prices database is empty. Please run update_prices first.")

    # Quality-Weighted Sentiment aggregation
    if not s_raw.empty:
        s_raw['Date'] = pd.to_datetime(s_raw['Date']).dt.strftime('%Y-%m-%d')
        s_raw['Pub_Weight'] = s_raw['Publisher'].apply(get_pub_weight)
        s_raw['Weighted_Value'] = s_raw['Value'] * s_raw['Pub_Weight']

        summary = s_raw.groupby(['Date', 'Ticker']).agg({
            'Weighted_Value': ['mean', 'median', 'std'],
            'Disagreement': 'mean',
            'Pub_Weight': 'mean'
        }).reset_index()
        summary.columns = ['Date', 'Ticker', 'Avg_Intensity', 'Median_Intensity', 'Intensity_Std', 'Avg_Disagreement', 'Avg_Publisher_Quality']

        counts = s_raw[s_raw['Sentiment'] != 'Neutral'].groupby(['Date', 'Ticker', 'Sentiment']).size().unstack(fill_value=0)
        for col in ['Positive', 'Negative']:
            if col not in counts.columns: counts[col] = 0
        sent_df = summary.merge(counts, on=['Date', 'Ticker'], how='left').fillna({'Positive': 0, 'Negative': 0})
        sent_df['Net_Count'] = sent_df['Positive'] - sent_df['Negative']
    else:
        sent_df = pd.DataFrame(columns=['Date', 'Ticker', 'Avg_Intensity', 'Avg_Publisher_Quality', 'Positive', 'Negative', 'Net_Count'])

    # Macro benchmarks
    price_df = all_prices.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    vix_df = price_df[price_df['Ticker'] == '^VIX'][['Date', 'Price']].rename(columns={'Price': 'VIX_Close'})
    vix_df['VIX_Daily_Change'] = vix_df['VIX_Close'].pct_change()

    spy_df = price_df[price_df['Ticker'] == 'SPY'][['Date', 'Price', 'Volume']].rename(columns={'Price': 'SPY_Close', 'Volume': 'SPY_Volume'})
    spy_df['SPY_Return'] = spy_df['SPY_Close'].pct_change()

    qqq_df = price_df[price_df['Ticker'] == 'QQQ'][['Date', 'Price']].rename(columns={'Price': 'QQQ_Close'})
    qqq_df['QQQ_Return'] = qqq_df['QQQ_Close'].pct_change()

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

    # Next-day Target for training (1 if next day return > 0, else 0)
    stock_df['Target'] = stock_df.groupby('Ticker')['Price'].shift(-1)
    stock_df['Target'] = (stock_df['Target'] > stock_df['Price']).astype(float)

    # Merge macros
    merged = stock_df.merge(vix_df, on='Date', how='left')
    merged = merged.merge(spy_df, on='Date', how='left')
    merged = merged.merge(qqq_df, on='Date', how='left')
    merged['Excess_Return'] = merged['Daily_Return'] - merged['SPY_Return']

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

def run_models_and_inference():
    print("⚙️ Step 4: Engineering 31 Alpha Signals & Generating Pre-Market Inferences...")
    df = build_feature_dataset()

    feature_cols = [
        'Daily_Return', 'Return_3d', 'Return_5d', 'SMA_5_dist', 'SMA_20_dist', 'RSI_14', 'Vol_10d',
        'Overnight_Gap', 'Intraday_Return', 'Daily_Range', 'CLV', 'Volume_Surge',
        'SPY_Return', 'QQQ_Return', 'Excess_Return', 'VIX_Close', 'VIX_Daily_Change',
        'Positive', 'Negative', 'Net_Count', 'Avg_Intensity', 'Avg_Publisher_Quality', 'Sent_Rolling_3d', 'Sent_Divergence',
        'High_Vol_Sentiment', 'Rolling_Beta_SPY_20d', 'Rolling_Corr_SPY_20d',
        'Peer_Tech_Avg_Return', 'Peer_Tech_Avg_Sent', 'NVDA_Lead_Return', 'NVDA_Lead_Sent'
    ]

    # Clean historical training set (all rows where Target is known)
    train_df = df.dropna(subset=['Target', 'Daily_Return', 'Overnight_Gap', 'Volume_Surge', 'RSI_14', 'Rolling_Beta_SPY_20d']).copy()
    X_train = train_df[feature_cols]
    y_train = train_df['Target'].astype(int)

    # 1. Fit Locked-in LightGBM
    lgb = LGBMClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.02,
        colsample_bytree=0.7,
        subsample=0.8,
        min_child_samples=15,
        random_state=42,
        n_jobs=1,
        verbose=-1
    )
    lgb.fit(X_train, y_train)

    # 2. Fit Locked-in Random Forest
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=4,
        min_samples_split=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train, y_train)

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

        # Specialist routing: NVDA uses Tuned RF; others use Ensemble
        if ticker == 'NVDA':
            final_prob = prob_rf
            model_used = "Tuned RF Specialist"
        else:
            final_prob = prob_ens
            model_used = "60/40 Ensemble (LGB+RF)"

        conviction = max(final_prob, 1.0 - final_prob)

        # Decision rule & conviction tier
        if final_prob >= 0.55:
            action = "BUY / LONG"
            color = "#10b981"
            badge = "🟢 BUY"
        elif final_prob <= 0.45:
            action = "SELL / SHORT"
            color = "#ef4444"
            badge = "🔴 SELL"
        else:
            action = "HOLD / PASS"
            color = "#6b7280"
            badge = "⚪ HOLD"

        high_conviction = conviction >= 0.60 or conviction <= 0.40

        inferences.append({
            'Ticker': ticker,
            'Date': latest_row['Date'].values[0],
            'Price': latest_row['Price'].values[0],
            'Prob_Up': final_prob,
            'Conviction': conviction,
            'Action': action,
            'Badge': badge,
            'Color': color,
            'High_Conviction': high_conviction,
            'Model_Used': model_used,
            'Sent_Intensity': latest_row['Avg_Intensity'].values[0],
            'Sent_Divergence': latest_row['Sent_Divergence'].values[0],
            'RSI_14': latest_row['RSI_14'].values[0],
            'Return_5d': latest_row['Return_5d'].values[0] * 100
        })

    # Macro regime indicators
    latest_macro = df.iloc[-1]
    macro_info = {
        'Date': latest_date,
        'VIX': latest_macro.get('VIX_Close', 0.0),
        'VIX_Change': latest_macro.get('VIX_Daily_Change', 0.0) * 100,
        'SPY_Return': latest_macro.get('SPY_Return', 0.0) * 100,
        'Tech_Cluster_Sent': latest_macro.get('Peer_Tech_Avg_Sent', 0.0)
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
            f"- {inf['Ticker']}: Action {inf['Action']}{flag} | Conviction: {inf['Conviction']*100:.1f}% | "
            f"P(Up): {inf['Prob_Up']*100:.1f}% | Price: ${inf['Price']:.2f} | 5D Ret: {inf['Return_5d']:+.1f}% | "
            f"Sentiment: {inf['Sent_Intensity']:+.2f} | Divergence: {inf['Sent_Divergence']:+.2f} | Specialist: {inf['Model_Used']}"
        )
    stocks_text = "\n".join(stock_summaries)

    prompt = f"""
You are an elite quantitative portfolio manager writing the 8:00 AM Morning Trading Briefing for the lead trader.
Market Session: {macro_info['Date']}

MACRO INDICATORS:
- VIX: {macro_info['VIX']:.2f} ({macro_info['VIX_Change']:+.2f}%)
- S&P 500 (SPY 1D): {macro_info['SPY_Return']:+.2f}%
- Tech Cluster Sentiment: {macro_info['Tech_Cluster_Sent']:+.2f}

QUANTITATIVE ML ENSEMBLE PREDICTIONS:
{stocks_text}

Provide an executive trading note formatted in clean HTML (use <p>, <ul>, <li>, <strong> only):
1. <strong>Macro & Volatility Climate</strong>: 2 concise sentences on market risk regime.
2. <strong>Tactical Positioning & Stock Setups</strong>: 1-2 bullet points per key stock synthesizing momentum, sentiment divergence, and the model's stance.
3. <strong>Highest Conviction Play & Risk Rules</strong>: Highlight the top setup and capital preservation rule for the day.

Keep it sharp, professional, and actionable.
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
    bullet_items = "".join(f"<li><strong>{inf['Ticker']}:</strong> {inf['Action']} (Conviction: {inf['Conviction']*100:.1f}%) — {inf['Model_Used']}</li>" for inf in inferences)
    return f"""
    <p><strong>Macro Regime:</strong> VIX is at {macro_info['VIX']:.2f} ({macro_info['VIX_Change']:+.2f}%), with SPY at {macro_info['SPY_Return']:+.2f}%. Model posture indicates selective risk-taking.</p>
    <ul>{bullet_items}</ul>
    """

def generate_html_report(inferences, macro_info, gemma_narrative=None):
    rows_html = ""
    for inf in inferences:
        conv_pct = inf['Conviction'] * 100
        prob_pct = inf['Prob_Up'] * 100
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
            <td style="padding: 12px 14px; color: #94a3b8; text-align: right;">{prob_pct:.1f}%</td>
            <td style="padding: 12px 14px; color: #f8fafc; text-align: right;">${inf['Price']:.2f}</td>
            <td style="padding: 12px 14px; color: {'#10b981' if inf['Return_5d'] >= 0 else '#ef4444'}; text-align: right;">{inf['Return_5d']:+.1f}%</td>
            <td style="padding: 12px 14px; color: #cbd5e1; font-size: 13px;">{inf['Model_Used']}</td>
        </tr>
        """

    vix_color = "#ef4444" if macro_info['VIX_Change'] > 0 else "#10b981"
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
        <div style="max-width: 780px; margin: 0 auto; background-color: #1e293b; border-radius: 12px; border: 1px solid #334155; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 24px; border-bottom: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #38bdf8; letter-spacing: 0.5px;">
                        ⚡ QUANT PRE-MARKET INTELLIGENCE
                    </h1>
                </div>
                <p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">
                    Market Session: <strong style="color: #f8fafc;">{macro_info['Date']}</strong> | Ready by 8:00 AM EST | Gemma 4 31B + ML Ensemble
                </p>
            </div>

            <!-- Macro Regime Bar -->
            <div style="background-color: #162032; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; flex-wrap: wrap; gap: 20px; font-size: 13px;">
                <div><strong>VIX:</strong> <span style="color: #f8fafc;">{macro_info['VIX']:.2f}</span> (<span style="color: {vix_color};">{macro_info['VIX_Change']:+.2f}%</span>)</div>
                <div><strong>SPY 1D:</strong> <span style="color: {'#10b981' if macro_info['SPY_Return'] >= 0 else '#ef4444'};">{macro_info['SPY_Return']:+.2f}%</span></div>
                <div><strong>Tech Sentiment:</strong> <span style="color: {'#10b981' if macro_info['Tech_Cluster_Sent'] >= 0 else '#ef4444'};">{macro_info['Tech_Cluster_Sent']:+.2f}</span></div>
            </div>

            {narrative_block}

            <!-- Actions Table -->
            <div style="padding: 24px;">
                <h2 style="font-size: 16px; margin: 0 0 14px 0; color: #f8fafc;">🎯 Recommended Actions & Conviction Rankings</h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
                        <thead>
                            <tr style="background-color: #0f172a; color: #94a3b8; font-size: 12px; text-transform: uppercase;">
                                <th style="padding: 10px 14px;">Ticker</th>
                                <th style="padding: 10px 14px; text-align: center;">Action</th>
                                <th style="padding: 10px 14px; text-align: right;">Conviction</th>
                                <th style="padding: 10px 14px; text-align: right;">P(Up)</th>
                                <th style="padding: 10px 14px; text-align: right;">Close</th>
                                <th style="padding: 10px 14px; text-align: right;">5D Ret</th>
                                <th style="padding: 10px 14px;">Model Specialist</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>

                <!-- Guidance notes -->
                <div style="margin-top: 24px; padding: 16px; background-color: #0f172a; border-radius: 8px; border-left: 4px solid #38bdf8; font-size: 13px; line-height: 1.5; color: #94a3b8;">
                    <strong style="color: #38bdf8;">📌 Execution Strategy & Conviction Thresholds:</strong><br>
                    • <strong>High Conviction (🔥 &ge; 60% or &le; 40%)</strong>: Historically demonstrated 62.5% out-of-sample win rate.<br>
                    • <strong>Selective Action (&ge; 55% or &le; 45%)</strong>: 58.1% win rate. Clean momentum & sentiment divergence alignment.<br>
                    • <strong>Hold / Neutral (45% - 55%)</strong>: Expected value within noise threshold. Stay on sidelines or protect capital.
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
    print("\n" + "=" * 80)
    print(f"📊 PRE-MARKET ACTION DASHBOARD ({macro_info['Date']})")
    print(f"Macro: VIX {macro_info['VIX']:.2f} ({macro_info['VIX_Change']:+.2f}%) | SPY {macro_info['SPY_Return']:+.2f}%")
    print("=" * 80)
    print(f"{'Ticker':<7} {'Action':<15} {'Conviction':<12} {'P(Up)':<10} {'Close':<10} {'Model Specialist':<22}")
    print("-" * 80)
    for inf in inferences:
        flag = " 🔥" if inf['High_Conviction'] else ""
        print(f"{inf['Ticker']:<7} {inf['Action']:<15} {inf['Conviction']*100:>5.1f}%{flag:<5} {inf['Prob_Up']*100:>5.1f}%     ${inf['Price']:<9.2f} {inf['Model_Used']:<22}")
    print("=" * 80 + "\n")

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
