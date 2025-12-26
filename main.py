import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v23.6 — MULTI-LEVEL ADVISOR (TOP 1 / 2 / 3)
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- PARAMÈTRES ---
TOTAL_CAPITAL = 1000  # Ton capital actuel (pour marquer la version active)

OFFENSIVE_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "ASML", "AVGO", "SMH", "VRT", "PLTR", "PANW", "RKLB",
    "CRWD", "SMCI", "ARM", "APP", "BTC-USD", "ETH-USD", "SOL-USD"
]

DEFENSIVE_TICKERS = [
    "LLY", "UNH", "ISRG", "ETN", "URNM", "XLE",
    "COPX", "SIL", "REMX", "GLD", "ITA", "RACE", "MC.PA"
]

ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"

def calculate_rsi(series):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    return 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

def run():
    print("🚀 Lancement APEX v23.6...")
    
    try:
        raw = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX"], period="2y", auto_adjust=True, progress=False)
        close, high, low = raw["Close"].ffill(), raw["High"].ffill(), raw["Low"].ffill()
    except Exception as e:
        print(f"❌ Erreur Data: {e}"); return

    prices = close[ALL_TICKERS]
    spy = close[MARKET_INDEX].reindex(prices.index).ffill()
    vix = close["^VIX"].reindex(prices.index).ffill()
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0

    # --- RÉGIME ---
    ma200 = spy.rolling(200).mean()
    vix_threshold = vix.rolling(252).quantile(0.4).iloc[-1]
    score_regime = sum([float(spy.iloc[-1]) > float(ma200.iloc[-1]), 
                        float(ma200.iloc[-1]) > float(ma200.iloc[-20]),
                        float(vix.iloc[-1]) < float(vix_threshold)])

    regimes = {3: ("🟢🟢🟢", "MAX BULL", 1.0, ALL_TICKERS), 
               2: ("🟢🟢", "STRONG BULL", 0.9, ALL_TICKERS),
               1: ("🟡", "CAUTIOUS", 0.8, ALL_TICKERS),
               0: ("🔴", "DEFENSIVE", 0.8, DEFENSIVE_TICKERS)}
    
    icon, name, exposure, ground = regimes.get(score_regime, regimes[0])

    # --- SCORING ---
    active_prices = prices[ground]
    m = (0.2 * (active_prices/active_prices.shift(63)-1) + 0.3 * (active_prices/active_prices.shift(126)-1) + 0.5 * (active_prices/active_prices.shift(252)-1))
    z_mom = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(0.001)
    rs = (active_prices/active_prices.shift(126)) / (spy/spy.shift(126)).values.reshape(-1,1)
    rs_z = (rs - rs.mean(axis=1).values.reshape(-1,1)) / rs.std(axis=1).values.reshape(-1,1).clip(0.001)
    final_scores = z_mom.iloc[-1] + (rs_z.iloc[-1] * 0.5)
    
    rsi_vals = active_prices.apply(calculate_rsi).iloc[-1]
    valid = (z_mom.iloc[-1] > 0) & (rs_z.iloc[-1] > 0) & (rsi_vals < 80)
    candidates = final_scores[valid].nlargest(8)

    # --- CONSTRUCTION DU MESSAGE ---
    msg = f"🤖 **APEX v23.6 — ANALYSE MULTI-NIVEAUX**\n"
    msg += f"🌍 Régime: {icon} {name} | Expo: {int(exposure*100)}%\n"
    msg += f"💰 Capital Actuel: {TOTAL_CAPITAL}€\n"
    msg += "⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n\n"

    # Boucle sur les 3 niveaux de TOP
    for n in [1, 2, 3]:
        selected = []
        for t in candidates.index:
            if not selected: selected.append(t)
            elif active_prices[selected + [t]].pct_change().iloc[-63:].corr().loc[t, selected].max() < 0.85:
                selected.append(t)
            if len(selected) == n: break
        
        # Déterminer si c'est la version recommandée pour le capital actuel
        status = "⭐ **CONSEILLÉ**" if (n==1 and TOTAL_CAPITAL<3000) or (n==2 and 3000<=TOTAL_CAPITAL<6000) or (n==3 and TOTAL_CAPITAL>=6000) else "🔹 Option"
        
        msg += f"🏆 **NIVEAU TOP {n}** | {status}\n"
        
        if not selected:
            msg += "   Pas d'actifs valides.\n"
        else:
            vols = active_prices[selected].pct_change().iloc[-126:].std() * np.sqrt(252)
            weights = (1 / vols.clip(lower=0.15))
            weights = (weights / weights.sum()) * exposure
            
            for t in selected:
                p_eur = float(prices[t].iloc[-1]) * (1 if t.endswith(".PA") else fx)
                tr = np.maximum(high[t]-low[t], np.maximum(abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))))
                stop = max(prices[t].iloc[-1] - (4.0 * tr.rolling(14).mean().iloc[-1]), prices[t].rolling(50).mean().iloc[-1] * 0.95)
                stop_eur = float(stop) * (1 if t.endswith(".PA") else fx)
                
                msg += f"   • {t}: {weights[t]*100:.1f}% ({p_eur:.2f}€ | SL: {stop_eur:.2f}€)\n"
        msg += "\n"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    print(msg)

if __name__ == "__main__":
    run()
