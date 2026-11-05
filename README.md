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
│     AMD, ^VIX, SPY, QQQ,      │                         └────────────────┬────────────────┘
│          SMH, XLP)            │                                          │
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
                              │     36 Alpha Signals        │
                              │ • Intraday Candlestick CLV  │
                              │ • SMH & XLP Sector Signals  │
                              │ • Rolling Beta & Covariance │
                              │ • Price-Sentiment Mismatch  │
                              │ • Volume Surges & RSI       │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  Asset Specialist Router    │
                              │ • NVDA 60/40 Ensemble Spec. │
                              │ • COST Staples-RF Specialist│
                              │ • TSLA Tuned RF Specialist  │
                              │ • AMD Semi-RF Specialist    │
                              │ • GOOGL AI/Semi-RF Spec.    │
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

## 🧠 Model Architecture & Recency Adaptation

The pipeline uses an **Asset Specialist Router** with **Exponential Recency Weighting** rather than treating historical data uniformly:
1. **1-Year Expanding Baseline**: Maintains ~1,300+ historical bars so models retain structural macro regime awareness (high/low VIX periods, rate prints, multi-month trends).
2. **Exponential Recency Decay**: Applies sample weights $w_t = \exp\left(-\frac{\ln(2)}{H} \Delta t\right)$ with a configurable half-life (default: `RECENCY_HALF_LIFE_DAYS=60`). Recent 30-day market conditions have **~2.8x higher influence** on tree splits than data from 90 days ago.
3. **Asset-Specific Specialization (60.87% Out-of-Sample Accuracy, 0.592 ROC-AUC)**:
   - **NVDA** $\rightarrow$ `NVDA 60/40 Ensemble Specialist` (**61.8%** accuracy, **0.552** AUC — combines non-linear LightGBM & Random Forest).
   - **COST** $\rightarrow$ `COST Staples-RF Specialist` (**61.8%** accuracy, **0.594** AUC — anchored on `XLP` consumer staples alpha & isolated from tech-peer spillover).
   - **TSLA** $\rightarrow$ `TSLA Tuned RF Specialist` (**60.7%** accuracy, **0.556** AUC — depth-regularized Random Forest).
   - **AMD** $\rightarrow$ `AMD Semi-RF Specialist` (**60.0%** accuracy, **0.640** AUC — incorporates `SMH` semiconductor sector signals & `NVDA` lead).
   - **GOOGL** $\rightarrow$ `GOOGL AI/Semi-RF Specialist` (**60.0%** accuracy, **0.534** AUC — leverages AI/semiconductor hardware cycle & peer sentiment).

---

## 📁 Repository Structure

- `daily_pipeline.py`: Production script automating the complete pre-market process from ingestion to email dispatch.
- `gemma4_fast_server.py`: Ultra-optimized FastAPI inference server for Gemma 4 31B 4-bit (`torch.compile` + Static KV Cache + FlashAttention-2 + MTP speculative decoding).
- `gemma4_server.py`: Standard FastAPI inference server for Gemma 4 31B 4-bit with MTP speculative decoding.
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
