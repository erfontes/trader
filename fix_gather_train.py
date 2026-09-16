"""
Patches gather_train.ipynb to fix the TypeError when yf.download()
fails or returns an empty DataFrame under yfinance 1.x.
"""
import json
import re

NB_PATH = "/home/erfontes/research/trader/gather_train.ipynb"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

# -----------------------------------------------------------------------
# New update_prices source (as a list of strings matching notebook format)
# -----------------------------------------------------------------------
NEW_UPDATE_PRICES = [
    "def update_prices():\n",
    "    \"\"\"Fetch and upsert last 5 days of prices and clean up temp tables.\"\"\"\n",
    "    print(\"📈 Updating Price Data (last 5 days)...\")\n",
    "    conn = get_db_connection()\n",
    "    try:\n",
    "        # Download data\n",
    "        raw = yf.download(tickers, period=\"5d\", interval=\"1d\")\n",
    "\n",
    "        # Guard: empty result means all downloads failed\n",
    "        if raw is None or raw.empty:\n",
    "            print(\"⚠️  No price data returned — all tickers may have failed. Skipping price update.\")\n",
    "            return\n",
    "\n",
    "        # yfinance 1.x returns a MultiIndex DataFrame (Price_Field, Ticker).\n",
    "        # Extract the 'Close' slice — works for both single and multi-ticker.\n",
    "        if isinstance(raw.columns, pd.MultiIndex):\n",
    "            raw_prices = raw[\"Close\"]\n",
    "        else:\n",
    "            # Single-ticker fallback: make it look like multi-ticker\n",
    "            raw_prices = raw[[\"Close\"]].rename(columns={\"Close\": tickers[0]})\n",
    "\n",
    "        # Guard: if Close slice is empty or all NaN, nothing to do\n",
    "        if raw_prices.empty or raw_prices.isna().all().all():\n",
    "            print(\"⚠️  Close prices are all NaN — skipping price update.\")\n",
    "            return\n",
    "\n",
    "        # Melt/Stack the data into Long format\n",
    "        # future_stack=True silences the FutureWarning in pandas 2.x\n",
    "        try:\n",
    "            price_df = raw_prices.stack(future_stack=True).reset_index()\n",
    "        except TypeError:\n",
    "            # pandas < 2.1 doesn't have future_stack\n",
    "            price_df = raw_prices.stack().reset_index()\n",
    "\n",
    "        price_df.columns = [\"Date\", \"Ticker\", \"Price\"]\n",
    "\n",
    "        # --- FIX FOR NaT ISSUES ---\n",
    "        # 1. Ensure Date is datetime type\n",
    "        price_df[\"Date\"] = pd.to_datetime(price_df[\"Date\"], errors=\"coerce\")\n",
    "\n",
    "        # 2. Drop any rows where Date is NaT or Price is NaN\n",
    "        price_df = price_df.dropna(subset=[\"Date\", \"Price\"])\n",
    "\n",
    "        # 3. Convert to clean string format for SQLite\n",
    "        price_df[\"Date\"] = price_df[\"Date\"].dt.strftime(\"%Y-%m-%d\")\n",
    "        # --------------------------\n",
    "\n",
    "        # Sync to DB via temp staging\n",
    "        price_df.to_sql(\"temp_prices\", conn, if_exists=\"replace\", index=False)\n",
    "        cursor = conn.execute(\"\"\"\n",
    "            INSERT OR IGNORE INTO prices (Date, Ticker, Price)\n",
    "            SELECT Date, Ticker, Price FROM temp_prices\n",
    "        \"\"\")\n",
    "\n",
    "        # Clean up the temp table after sync\n",
    "        conn.execute(\"DROP TABLE IF EXISTS temp_prices\")\n",
    "\n",
    "        conn.commit()\n",
    "        print(f\"Successfully synced {cursor.rowcount} NEW price records.\")\n",
    "    except Exception as e:\n",
    "        print(f\"Error updating prices: {e}\")\n",
    "    finally:\n",
    "        conn.close()\n",
]

# -----------------------------------------------------------------------
# Helper: find the first code cell and patch update_prices in its source
# -----------------------------------------------------------------------
def patch_cell(source_lines):
    """
    Given the source lines of a code cell, replace the update_prices
    function body with NEW_UPDATE_PRICES.  Returns patched lines or
    None if the function wasn't found.
    """
    # Find where the function starts and ends
    start = None
    end = None
    for i, line in enumerate(source_lines):
        if re.match(r'^def update_prices\(\)', line.strip()):
            start = i
        elif start is not None and i > start:
            # Function ends at the next top-level def / class / end-of-list
            if re.match(r'^def ', line.strip()) or re.match(r'^class ', line.strip()):
                end = i
                break
    if start is None:
        return None
    if end is None:
        end = len(source_lines)

    patched = source_lines[:start] + NEW_UPDATE_PRICES + source_lines[end:]
    return patched


patched = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    result = patch_cell(src)
    if result is not None:
        cell["source"] = result
        # Clear stale outputs so the notebook re-runs cleanly
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    print("❌  Could not find update_prices in any code cell — no changes made.")
else:
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✅  gather_train.ipynb patched successfully.")
