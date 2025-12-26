import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v24.8.1 — GITHUB STABILITY FIXED
# ============================================================

# Récupération sécurisée des variables d'environnement
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOTAL_CAPITAL = 1000 

OFFENSIVE_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "ASML", "AVGO", "SMH", "VRT",
    "PLTR", "PANW", "RKLB", "CRWD", "SMCI", "ARM", "APP", "BTC-USD", "ETH-USD", "SOL-USD"
]

DEFENSIVE_TICKERS = [
    "LLY", "UNH", "ISRG", "ETN", "URNM", "XLE", "COPX", "SIL", "REMX", "GLD", "ITA", "RACE", "MC.PA",
    "PDBC", "XLU"
]

ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def send_telegram(message):
    if not TOKEN or not CHAT_ID:
        print("❌ Erreur: Secrets TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID manquants.")
        return
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"❌ Erreur Telegram ({response.status_code}): {response.text}")
        else:
            print("✅ Notification Telegram envoyée avec succès.")
    except Exception as e:
        print(f"❌ Erreur de connexion Telegram: {e}")

def run():
    print(f"🚀 Lancement APEX v24.8.1 — {datetime.now().strftime('%Y-%m-%d')}")

    # --- 1. TÉLÉCHARGEMENT ---
    try:
        tickers = ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"]
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        
        if data.empty:
            print("❌ Erreur: Aucune donnée reçue."); return
            
        close = data['Close'].ffill().bfill() if 'Close' in data else data.ffill().bfill()
        high = data['High'].ffill().bfill() if 'High' in data else close
        low = data['Low'].ffill().bfill() if 'Low' in data else close
    except Exception as e:
        print(f"❌ Erreur Data: {e}"); return

    # --- 2. RÉGIME ---
    prices = close[ALL_TICKERS]
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0

    ma200 = spy.rolling(200).mean()
    vix_threshold = vix.rolling(252).quantile(0.4).iloc[-1]

    score = 0
    if float(spy.iloc[-1]) > float(ma200.iloc[-1]): score += 1
    if float(ma200.iloc[-1]) > float(ma200.iloc[-20]): score += 1
    if float(vix.iloc[-1]) < float(vix_threshold): score += 1
    try:
        if (float(close["^TNX"].iloc[-1]) - float(close["^IRX"].iloc[-1])) > 0: score += 1
    except: pass

    exposure_map = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
    exposure = exposure_map.get(score, 0.0)
    regime_names = {0: "🔴 BEAR", 1: "🟡 CAUTIOUS", 2: "🟢 BULL", 3: "🟢🟢 STRONG", 4: "🟢🟢🟢 MAX"}
    
    # --- 3. RAPPORT ---
    if exposure == 0.0:
        msg = f"🤖 APEX v24.8.1\n**Régime:** {regime_names.get(score)} | Expo: **0%**\n💰 Capital: **{TOTAL_CAPITAL}€**\n━━━━━━━━━━━━━\n⚠️ **100% CASH**"
        send_telegram(msg)
        return

    ground = DEFENSIVE_TICKERS if score <= 1 else ALL_TICKERS
    active_prices = prices[ground].copy()
    m = (0.2 * (active_prices/active_prices.shift(63)-1) + 0.3 * (active_prices/active_prices.shift(126)-1) + 0.5 * (active_prices/active_prices.shift(252)-1))
    rs = (active_prices/active_prices.shift(126)).div((spy/spy.shift(126)), axis=0)
    z_mom = (m.iloc[-1] - m.iloc[-1].mean()) / m.iloc[-1].std()
    z_rs = (rs.iloc[-1] - rs.iloc[-1].mean()) / rs.iloc[-1].std()
    final_scores = (0.6 * z_mom + 0.4 * z_rs)
    
    rsi_vals = active_prices.apply(calculate_rsi).iloc[-1]
    ma150 = active_prices.rolling(150).mean().iloc[-1]
    valid = (final_scores > 0) & (rsi_vals < 80) & (active_prices.iloc[-1] > ma150)
    candidates = final_scores[valid].nlargest(10)

    msg = f"🤖 APEX v24.8.1 | {datetime.now().strftime('%d/%m/%Y')}\n"
    msg += f"**Régime:** {regime_names.get(score)} | Expo: **{int(exposure*100)}%**\n"
    msg += f"💰 Capital: **{TOTAL_CAPITAL}€**\n━━━━━━━━━━━━━\n\n"

    for n in [2, 3]:
        selected = list(candidates.nlargest(n).index)
        is_rec = (n == 2 and TOTAL_CAPITAL < 5000) or (n == 3 and TOTAL_CAPITAL >= 5000)
        status = "⭐ CONSEILLÉ" if is_rec else "🔹 Option"
        msg += f"🏆 **TOP {n}** | {status}\n"
        if not selected:
            msg += "   Aucun signal.\n"
        else:
            vols = active_prices[selected].pct_change().std() * np.sqrt(252)
            weights = (1 / vols.clip(lower=0.15))
            weights = (weights / weights.sum()) * exposure
            for t in selected:
                p_eur = float(active_prices[t].iloc[-1]) * (1 if t.endswith(".PA") else fx)
                tr = pd.concat([high[t]-low[t], abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))], axis=1).max(axis=1)
                sl_eur = (float(active_prices[t].iloc[-1]) - (4.0 * tr.rolling(14).mean().iloc[-1])) * (1 if t.endswith(".PA") else fx)
                msg += f"• **{t}**: {weights[t]*100:.1f}% ({(TOTAL_CAPITAL * weights[t]):.0f}€)\n  Prix: {p_eur:.2f}€ → **SL: {sl_eur:.2f}€**\n"
        msg += "\n"

    msg += "Process > Conviction | Never Average Down"
    send_telegram(msg)

if __name__ == "__main__":
    run()
