import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v25.2 — INTELLIGENCE ADAPTATIVE (TOP 2 à 8)
# ============================================================

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOTAL_CAPITAL = 1000 
RISK_PER_TRADE = 0.02 # On risque 2% du capital (20€) par position
ATR_MULT = 3.3

OFFENSIVE_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "ASML", "AVGO", "SMH", "VRT", "AAPL",
    "PLTR", "MU", "PANW", "RKLB", "CRWD", "SMCI", "ARM", "APP", "BTC-USD", "ETH-USD", "SOL-USD"
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

def run():
    print(f"🚀 Lancement APEX v25.2 Adaptatif — {datetime.now().strftime('%Y-%m-%d')}")

    try:
        # Téléchargement des données
        data = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"], period="2y", progress=False)
        close = data['Close'].ffill().bfill()
        high = data['High'].ffill().bfill()
        low = data['Low'].ffill().bfill()
    except Exception as e:
        print(f"❌ Erreur Data: {e}"); return

    # --- RÉGIME v25.2 ---
    spy, vix = close[MARKET_INDEX], close["^VIX"]
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0
    
    regime_score = (0.4*(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]) + 
                    0.3*(vix.iloc[-1] < vix.rolling(50).mean().iloc[-1]) + 
                    0.2*(spy.iloc[-1] > spy.iloc[-63]) + 
                    0.1*((close["^TNX"] - close["^IRX"]).iloc[-1] > 0))
    
    exposure = 1.0 if regime_score >= 0.65 else 0.75 if regime_score >= 0.45 else 0.5 if regime_score >= 0.3 else 0.0
    regime_name = {1.0: "🟢🟢🟢 MAX", 0.75: "🟢🟢 STRONG", 0.5: "🟢 BULL", 0.0: "🔴 BEAR"}.get(exposure, "🟡 CAUTIOUS")

    if exposure == 0:
        msg = f"🤖 APEX v25.2\n**Régime:** 🔴 BEAR | Expo: **0%**\n⚠️ **100% CASH**"
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg})
        return

    # --- SÉLECTION ADAPTATIVE (Jusqu'à 8 candidats) ---
    ground = ALL_TICKERS if exposure >= 0.5 else DEFENSIVE_TICKERS
    active_p = close[ground].dropna(axis=1)
    
    mom = (active_p.iloc[-1] / active_p.iloc[-126]) - 1
    rsi_vals = active_p.apply(calculate_rsi).iloc[-1]
    ma150 = active_p.rolling(150).mean().iloc[-1]
    
    valid = (rsi_vals < 75) & (active_p.iloc[-1] > ma150)
    candidates = mom[valid].nlargest(8).index.tolist() # On regarde le TOP 8 potentiel

    # --- CONSTRUCTION DU PORTEFEUILLE ---
    msg = f"🤖 APEX v25.2 | {regime_name} ({int(exposure*100)}%)\n💰 Capital: {TOTAL_CAPITAL}€\n━━━━━━━━━━━━━━\n\n"
    
    count = 0
    total_allocated_weight = 0
    
    for t in candidates:
        if count >= 8 or total_allocated_weight >= exposure: break
        
        p_eur = float(active_p[t].iloc[-1]) * (1 if t.endswith(".PA") else fx)
        # Calcul ATR pour Stop
        tr = pd.concat([high[t]-low[t], abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        sl_eur = (float(active_p[t].iloc[-1]) - ATR_MULT * atr_val) * (1 if t.endswith(".PA") else fx)
        
        risk_dist = p_eur - sl_eur
        if risk_dist > 0:
            # Sizing par le risque : combien d'unités pour perdre 2% (20€) si le SL est touché
            # Le poids max d'une ligne est capé à 25% pour forcer la diversification si possible
            weight = min(((TOTAL_CAPITAL * RISK_PER_TRADE) / risk_dist) * p_eur / TOTAL_CAPITAL, 0.25) * exposure
            
            if weight > 0.05: # On ignore les lignes trop petites (< 5%)
                msg += f"• **{t}**: {weight*100:.1f}% ({(TOTAL_CAPITAL * weight):.0f}€)\n  Prix: {p_eur:.2f}€ | **SL: {sl_eur:.2f}€**\n\n"
                total_allocated_weight += weight
                count += 1

    if count < 2:
        msg += "⚠️ Peu de signaux validés. Prudence accrue."

    msg += f"━━━━━━━━━━━━━━\n🎯 Signal: TOP {count} détecté.\n⚡ Process > Emotion"
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    run()
