#!/usr/bin/env bash
# ==============================================================================
# Safe Server Switcher: Toggle between gemma4_server.py (Original) and
# gemma4_fast_server.py (Fast GGUF v2) on port 11434 without modifying either.
# ==============================================================================
set -e

MODE="${1:-status}"
TRADER_DIR="/home/erfontes/research/trader"
VEL_PYTHON="/home/erfontes/miniconda3/envs/vel/bin/python"

if [ "$MODE" = "status" ]; then
    echo "🔍 Checking current server on http://127.0.0.1:11434/health ..."
    curl -s http://127.0.0.1:11434/health | python3 -m json.tool || echo "Server is currently offline."
    exit 0
fi

echo "🛑 Stopping any running Gemma 4 server on port 11434 (releasing GPU VRAM)..."
pkill -f "gemma4_server.py" 2>/dev/null || true
pkill -f "gemma4_fast_server.py" 2>/dev/null || true
pkill -f "127.0.0.1:11435" 2>/dev/null || true
sleep 3

cd "$TRADER_DIR"

if [ "$MODE" = "original" ]; then
    echo "🐢 Starting ORIGINAL gemma4_server.py (HF + bitsandbytes MTP)..."
    nohup "$VEL_PYTHON" "$TRADER_DIR/gemma4_server.py" > /tmp/gemma4_server.log 2>&1 &
    echo "⏳ Waiting for original server to load weights (~30-45s)..."
elif [ "$MODE" = "fast" ] || [ "$MODE" = "fast31b" ]; then
    echo "⚡ Starting FAST gemma4_fast_server.py (Gemma 4 31B GGUF Q4_K_M + FlashAttention)..."
    GEMMA4_FAST_MODEL="gemma4:31b" nohup "$VEL_PYTHON" "$TRADER_DIR/gemma4_fast_server.py" > /tmp/gemma4_fast_server.log 2>&1 &
    echo "⏳ Waiting for fast 31B server to load weights (~8-15s)..."
elif [ "$MODE" = "fast26b" ]; then
    echo "🚀 Starting ULTRA-FAST gemma4_fast_server.py (Gemma 4 26B MoE GGUF Q4_K_M)..."
    GEMMA4_FAST_MODEL="gemma4:26b" nohup "$VEL_PYTHON" "$TRADER_DIR/gemma4_fast_server.py" > /tmp/gemma4_fast_server.log 2>&1 &
    echo "⏳ Waiting for ultra-fast 26B server to load weights (~6-12s)..."
else
    echo "Usage: ./switch_gemma_server.sh [status | original | fast | fast26b]"
    exit 1
fi

for i in $(seq 1 45); do
    if curl -s http://127.0.0.1:11434/health | grep -q '"status":"online"'; then
        echo "✅ Server is ONLINE!"
        curl -s http://127.0.0.1:11434/health | python3 -m json.tool
        exit 0
    fi
    sleep 2
done

echo "⚠️ Timed out waiting for server. Check /tmp/gemma4_server.log or /tmp/gemma4_fast_server.log"
exit 1
