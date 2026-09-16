# =====================================================================
# 🚀 ADVANCED MODEL PIPELINE: INTRADAY OHLCV, VOLUME & QUALITY-WEIGHTED SENTIMENT
# =====================================================================
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# -------------------------------------------------------------
# 1. Fetch Full 1-Year OHLCV & Volume Data
# -------------------------------------------------------------
print("📡 Step 1: Downloading 1-Year OHLCV & Volume Data from yfinance...")
DB_NAME = "/home/erfontes/research/trader/market_data.db"
tickers = ['GOOGL', 'NVDA', 'TSLA', 'AMD', 'COST', '^VIX', 'SPY', 'QQQ']

raw = yf.download(tickers, period="1y", interval="1d", progress=False)

df_list = []
for ticker in tickers:
    t_df = raw.xs(ticker, level=1, axis=1).copy()
    t_df['Ticker'] = ticker
    t_df = t_df.reset_index()
    df_list.append(t_df)

all_prices = pd.concat(df_list, ignore_index=True)
all_prices['Date'] = pd.to_datetime(all_prices['Date']).dt.strftime('%Y-%m-%d')
all_prices = all_prices.rename(columns={'Close': 'Price'}).dropna(subset=['Date', 'Price'])

# -------------------------------------------------------------
# 2. Extract SQLite Sentiment & Apply Publisher Tier Weights
# -------------------------------------------------------------
print("🧠 Step 2: Loading Quality-Weighted Gemma 4 Sentiment from SQLite...")

PUBLISHER_TIERS = {
    "tier_1": ["bloomberg", "reuters", "wall street journal", "wsj", "financial times", "pr newswire", "business wire", "sec", "globe newswire", "cnbc"],
    "tier_2": ["barron's", "barrons", "marketwatch", "investor's business daily", "ibd", "seeking alpha", "yahoo finance", "investing.com", "barchart", "thefly", "insider monkey", "tipranks"],
    "tier_3": ["marketbeat", "motley fool", "the motley fool", "24/7 wall st.", "stocktwits", "benzinga", "zacks", "simply wall st."]
}

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

conn = sqlite3.connect(DB_NAME)
s_raw = pd.read_sql("""
    SELECT s.Date, s.Ticker, s.Sentiment_Label as Sentiment, 
           s.Median_Intensity as Value, s.Disagreement_Std as Disagreement,
           n.Publisher
    FROM sentiment_results s
    LEFT JOIN news n ON s.NewsID = n.NewsID
""", conn)
conn.close()

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
    if col not in counts.columns:
        counts[col] = 0
sent_df = summary.merge(counts, on=['Date', 'Ticker'], how='left').fillna({'Positive': 0, 'Negative': 0})
sent_df['Net_Count'] = sent_df['Positive'] - sent_df['Negative']

# -------------------------------------------------------------
# 3. Macro Benchmarks (VIX, SPY, QQQ)
# -------------------------------------------------------------
price_df = all_prices.sort_values(['Ticker', 'Date']).reset_index(drop=True)

vix_df = price_df[price_df['Ticker'] == '^VIX'][['Date', 'Price']].rename(columns={'Price': 'VIX_Close'})
vix_df['VIX_Daily_Change'] = vix_df['VIX_Close'].pct_change()

spy_df = price_df[price_df['Ticker'] == 'SPY'][['Date', 'Price', 'Volume']].rename(columns={'Price': 'SPY_Close', 'Volume': 'SPY_Volume'})
spy_df['SPY_Return'] = spy_df['SPY_Close'].pct_change()
spy_df['SPY_Vol_SMA20'] = spy_df['SPY_Volume'].rolling(20, min_periods=5).mean()
spy_df['SPY_Vol_Surge'] = spy_df['SPY_Volume'] / (spy_df['SPY_Vol_SMA20'] + 1e-6)

qqq_df = price_df[price_df['Ticker'] == 'QQQ'][['Date', 'Price', 'Volume']].rename(columns={'Price': 'QQQ_Close', 'Volume': 'QQQ_Volume'})
qqq_df['QQQ_Return'] = qqq_df['QQQ_Close'].pct_change()

macro_tickers = ['^VIX', 'SPY', 'QQQ']
stock_df = price_df[~price_df['Ticker'].isin(macro_tickers)].copy()

# -------------------------------------------------------------
# 4. Feature Engineering: Intraday Geometry & Volume Surge
# -------------------------------------------------------------
print("⚙️ Step 3: Engineering Intraday Geometry & Volume Alpha Signals...")
stock_df['Target_Price'] = stock_df.groupby('Ticker')['Price'].shift(-1)
stock_df['Target'] = (stock_df['Target_Price'] > stock_df['Price']).astype(int)
stock_df['Daily_Return'] = stock_df.groupby('Ticker')['Price'].pct_change()
stock_df['Return_3d'] = stock_df.groupby('Ticker')['Price'].pct_change(3)
stock_df['Return_5d'] = stock_df.groupby('Ticker')['Price'].pct_change(5)
stock_df['Vol_10d'] = stock_df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(10, min_periods=3).std())

# A. Intraday Geometry
stock_df['Prev_Close'] = stock_df.groupby('Ticker')['Price'].shift(1)
stock_df['Overnight_Gap'] = (stock_df['Open'] / stock_df['Prev_Close']) - 1.0
stock_df['Intraday_Return'] = (stock_df['Price'] / stock_df['Open']) - 1.0
stock_df['Daily_Range'] = (stock_df['High'] - stock_df['Low']) / stock_df['Price']
stock_df['CLV'] = ((stock_df['Price'] - stock_df['Low']) - (stock_df['High'] - stock_df['Price'])) / (stock_df['High'] - stock_df['Low'] + 1e-6)

# B. Volume Surge
stock_df['Vol_SMA20'] = stock_df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
stock_df['Volume_Surge'] = stock_df['Volume'] / (stock_df['Vol_SMA20'] + 1e-6)

# C. Technical Indicators
stock_df['SMA_5'] = stock_df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5, min_periods=2).mean())
stock_df['SMA_20'] = stock_df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(20, min_periods=5).mean())
stock_df['SMA_5_dist'] = (stock_df['Price'] / stock_df['SMA_5']) - 1.0
stock_df['SMA_20_dist'] = (stock_df['Price'] / stock_df['SMA_20']) - 1.0

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=3).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=3).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

stock_df['RSI_14'] = stock_df.groupby('Ticker')['Price'].transform(lambda x: calc_rsi(x, 14))

# Merge
merged = stock_df.merge(vix_df, on='Date', how='left')
merged = merged.merge(spy_df, on='Date', how='left')
merged = merged.merge(qqq_df, on='Date', how='left')
merged = merged.merge(sent_df, on=['Date', 'Ticker'], how='left')

fill_cols = ['Positive', 'Negative', 'Net_Count', 'Avg_Intensity', 'Avg_Publisher_Quality']
for col in fill_cols:
    merged[col] = merged[col].fillna(0)

merged['Excess_Return'] = merged['Daily_Return'] - merged['SPY_Return']
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

df_clean = merged.dropna(subset=['Target', 'Daily_Return', 'Overnight_Gap', 'Volume_Surge', 'RSI_14', 'Rolling_Beta_SPY_20d']).sort_values('Date').reset_index(drop=True)

feature_cols = [
    'Daily_Return', 'Return_3d', 'Return_5d', 'SMA_5_dist', 'SMA_20_dist', 'RSI_14', 'Vol_10d',
    'Overnight_Gap', 'Intraday_Return', 'Daily_Range', 'CLV', 'Volume_Surge',
    'SPY_Return', 'QQQ_Return', 'Excess_Return', 'VIX_Close', 'VIX_Daily_Change',
    'Positive', 'Negative', 'Net_Count', 'Avg_Intensity', 'Avg_Publisher_Quality', 'Sent_Rolling_3d', 'Sent_Divergence',
    'High_Vol_Sentiment', 'Rolling_Beta_SPY_20d', 'Rolling_Corr_SPY_20d',
    'Peer_Tech_Avg_Return', 'Peer_Tech_Avg_Sent', 'NVDA_Lead_Return', 'NVDA_Lead_Sent'
]

split_idx = int(len(df_clean) * 0.80)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df['Target']
X_test, y_test = test_df[feature_cols], test_df['Target']

print(f"🤖 Step 4: Training Enhanced Models (Train: {len(X_train)} rows | Test: {len(X_test)} rows)...")

best_lgb = LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, colsample_bytree=0.7, subsample=0.8, min_child_samples=15, random_state=42, n_jobs=1, verbose=-1)
best_lgb.fit(X_train, y_train)

best_rf = RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_split=8, max_features='sqrt', random_state=42, n_jobs=1)
best_rf.fit(X_train, y_train)

lgb_preds = best_lgb.predict(X_test)
lgb_probs = best_lgb.predict_proba(X_test)[:, 1]

rf_preds = best_rf.predict(X_test)
rf_probs = best_rf.predict_proba(X_test)[:, 1]

ensemble_probs = (lgb_probs * 0.60) + (rf_probs * 0.40)
ensemble_preds = (ensemble_probs >= 0.50).astype(int)

test_eval = test_df.copy()
test_eval['LGB_Pred'] = lgb_preds
test_eval['RF_Pred'] = rf_preds
test_eval['Ensemble_Pred'] = ensemble_preds
test_eval['Ensemble_Prob'] = ensemble_probs

print("\n" + "="*80)
print("📊 OUT-OF-SAMPLE BREAKDOWN BY STOCK (QUALITY-WEIGHTED SENTIMENT + INTRADAY OHLCV)")
print("="*80)
print(f"{'Ticker':<7} {'Test Days':<10} {'Market Base Rate':<17} {'Tuned LGBM Acc':<15} {'Tuned RF Acc':<13} {'Ensemble Acc':<13} {'Ensemble AUC':<12}")

for ticker in ['NVDA', 'TSLA', 'GOOGL', 'AMD', 'COST']:
    sub = test_eval[test_eval['Ticker'] == ticker]
    if len(sub) > 0:
        base_rate = sub['Target'].mean() * 100
        lgb_acc = accuracy_score(sub['Target'], sub['LGB_Pred']) * 100
        rf_acc = accuracy_score(sub['Target'], sub['RF_Pred']) * 100
        ens_acc = accuracy_score(sub['Target'], sub['Ensemble_Pred']) * 100
        ens_auc = roc_auc_score(sub['Target'], sub['Ensemble_Prob']) if len(sub['Target'].unique()) > 1 else 0.5
        print(f"{ticker:>6} {len(sub):>10} {sub['Target'].sum():>5}/{len(sub)} ({base_rate:>4.1f}%) {lgb_acc:>14.1f}% {rf_acc:>12.1f}% {ens_acc:>12.1f}% {ens_auc:>12.3f}")

print("\n" + "="*50)
print("🎯 SELECTIVE TRADING CONVICTION FILTER")
print("="*50)
for threshold in [0.50, 0.53, 0.55, 0.57, 0.60]:
    high_mask = (ensemble_probs >= threshold) | (ensemble_probs <= (1 - threshold))
    n_trades = high_mask.sum()
    if n_trades > 0:
        filt_preds = (ensemble_probs[high_mask] >= 0.50).astype(int)
        win_rate = accuracy_score(y_test[high_mask], filt_preds) * 100
        print(f"Conviction >= {int(threshold*100)}% or <= {int((1-threshold)*100)}%: Trades: {n_trades:>3}/{len(y_test)} | Win Rate: {win_rate:.2f}%")

imp = pd.DataFrame({
    'Feature': feature_cols,
    'LGBM_Splits': best_lgb.feature_importances_,
    'RF_Importance': best_rf.feature_importances_ * 100
}).sort_values('LGBM_Splits', ascending=False)

print("\n" + "="*50)
print("🔍 TOP FEATURE IMPORTANCES")
print("="*50)
print(imp.head(15).to_string(index=False))
