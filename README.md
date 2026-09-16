# ⚡ Quant Trader: Pre-Market Algorithmic Trading Pipeline

An end-to-end quantitative trading pipeline designed for pre-market execution before 8:00 AM EST. The system couples intraday OHLCV geometry, cross-stock covariance, and macro volatility indicators with high-throughput local LLM (**Gemma 4 31B 4-bit with MTP Speculative Assistant**) consensus sentiment analysis, evaluated through a tuned 60/40 LightGBM + Random Forest ensemble with asset-specific specialist routing.

---

## 🌟 Architecture & Key Features

```
                               ┌───────────────────────────┐
                               │  7:00 AM Scheduled Run    │
                               └─────────────┬─────────────┘
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
┌───────────────────────────────┐                         ┌─────────────────────────────────┐
│     OHLCV Data Ingestion      │                         │     Curated News Gathering      │
│  yfinance: Stocks & Macro     │                         │ Yahoo Finance + Google News RSS │
│   (GOOGL, NVDA, TSLA, COST,   │                         │  Headline Dedup + Pub Weighting │
│       AMD, ^VIX, SPY, QQQ)    │                         └────────────────┬────────────────┘
└──────────────┬────────────────┘                                          │
               │                                                           ▼
               │                                          ┌─────────────────────────────────┐
               │                                          │   Gemma 4 31B Consensus Engine  │
               │                                          │  (Auto-starts & stops on 4090)  │
               │                                          │   Sentiment + Materiality Score │
               │                                          └────────────────┬────────────────┘
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │     31 Alpha Signals        │
                              │ • Intraday Candlestick CLV  │
                              │ • Rolling Beta & Covariance │
                              │ • Price-Sentiment Mismatch  │
                              │ • Volume Surges & RSI       │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │ Locked-in Model Ensemble    │
                              │ • 60% Tuned LightGBM        │
                              │ • 40% Tuned Random Forest   │
                              │ • NVDA Tuned RF Specialist  │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │ Selective Conviction Filter │
                              │   P >= 55%: BUY / LONG      │
                              │   P <= 45%: SELL / SHORT    │
                              │   45% - 55%: HOLD / PASS    │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  Gemma 4 Executive Summary  │
                              │  + Styled HTML Email Alert  │
                              │   Dispatched by 8:00 AM     │
                              └─────────────────────────────┘
```

---

## 📁 Repository Structure

- `daily_pipeline.py`: Production script automating the complete pre-market process from ingestion to email dispatch.
- `gemma4_server.py`: High-throughput FastAPI inference server for Gemma 4 31B 4-bit with MTP speculative decoding.
- `test_ohlcv_features.py`: Feature engineering benchmark and out-of-sample backtesting script.
- `gather_train.ipynb`: Jupyter notebook detailing model experiments, hyperparameter tuning, and historical validation.
- `daily_pipeline.env.example`: Configuration template for SMTP email and model server paths.
- `latest_daily_briefing.html`: Generated pre-market briefing preview.

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
# Activate tabular research environment
source bin/activate

# Install dependencies if needed
pip install yfinance lightgbm scikit-learn pandas numpy requests
```

### 2. Configuration
Copy the configuration template:
```bash
cp daily_pipeline.env.example .env
```
Configure your email credentials in `.env`:
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_TO=erfontes@gmail.com
```

### 3. Run Pipeline
```bash
# Full execution
python daily_pipeline.py

# Dry-run (generates latest_daily_briefing.html without sending email)
python daily_pipeline.py --dry-run
```

### 4. Schedule by 8:00 AM
Add to `crontab -e`:
```cron
0 7 * * 1-5 /home/erfontes/research/trader/bin/python /home/erfontes/research/trader/daily_pipeline.py >> /home/erfontes/research/trader/cron_pipeline.log 2>&1
```
