import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ============================================================
# 1. CONFIGURATION (GitHub Secrets)
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOP_N = 3
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AAPL", "TSLA", "SMH",
    "BTC-USD", "ETH-USD", "SOL-USD",
    "MC.PA", "RMS.PA", "RACE",
    "LLY", "UNH",
    "URNM", "COPX", "XLE",
    "ITA", "NDIA.L", "GLD"
]
MARKET_INDEX = "SPY"

# ============================================================
# 2. LOGIQUE V9.0 APEX CORE
# ============================================================
def run():
    # Données sur 2 ans pour les moyennes mobiles longues
    data = yf.download(TICKERS + [MARKET_INDEX, "EURUSD=X"], period="2y", auto_adjust=True, progress=False)["Close"].ffill()
    
    prices, spy = data[TICKERS], data[MARKET_INDEX]
    fx = 1 / data["EURUSD=X"].iloc[-1]
    today = data.index[-1]

    # Indicateurs
    returns = prices.pct_change()
    vol_daily = returns.rolling(20).std() * 100
    vol_ann = returns.rolling(252).std() * np.sqrt(252) # Volatilité stable sur 1 an
    vol_ann = vol_ann.clip(lower=0.10)

    # Momentum V9 : Multi-horizons pondéré / Volatilité 1 an
    m3, m6, m12 = prices/prices.shift(63)-1, prices/prices.shift(126)-1, prices/prices.shift(252)-1
    momentum = (0.2 * m3 + 0.3 * m6 + 0.5 * m12) / vol_ann

    ma100, ma150 = prices.rolling(100).mean(), prices.rolling(150).mean()
    hwm_55 = prices.rolling(55).max()
    ma200_spy = spy.rolling(200).mean()
    slope_spy = ma200_spy.diff(20)

    # Régime de Marché
    if spy.loc[today] > ma200_spy.loc[today] and slope_spy.loc[today] > 0:
        regime_txt, exposure = "🟢 BULL FORT", 1.0
    elif spy.loc[today] > ma200_spy.loc[today]:
        regime_txt, exposure = "🟡 BULL FAIBLE", 0.5
    else:
        regime_txt, exposure = "🔴 BEAR", 0.25

    # Sélection stricte (V9 : Momentum doit être positif)
    valid = (prices.loc[today] > ma150.loc[today]) & (momentum.loc[today] > 0)
    scores = momentum.loc[today][valid].dropna()

    # En mode Bear, on ne garde que le meilleur actif
    if exposure < 0.5 and not scores.empty:
        scores = scores.nlargest(1)

    top_assets = scores.nlargest(TOP_N)

    # Allocation (Risk Parity)
    weights = pd.Series(dtype=float)
    if not top_assets.empty:
        inv_vol = 1 / vol_ann.loc[today, top_assets.index]
        weights = (inv_vol / inv_vol.sum()) * exposure

    # Message Telegram
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **ALGO ELITE V9.0 — APEX CORE**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME : {regime_txt} ({int(exposure*100)}%)**\n\n"

    if not weights.empty:
        msg += "🎯 **ALLOCATION & STOPS :**\n"
        for t, w in weights.items():
            p_raw = prices.loc[today, t]
            p_eur = p_raw if t.endswith(".PA") else p_raw * fx
            
            # Stop Hybride V9
            buffer = max(min(vol_daily.loc[today, t] * 2.5, 18), 6)
            stop_raw = max(ma100.loc[today, t], hwm_55.loc[today, t] * (1 - buffer / 100))
            stop_eur = stop_raw if t.endswith(".PA") else stop_raw * fx

            msg += f"• **{t}** : **{w*100:.1f}%**\n"
            msg += f"  Prix : {p_eur:.2f}€ | 🛡️ **STOP : {stop_eur:.2f}€**\n\n"
    else:
        msg += "⚠️ **TOTAL CASH — Aucune opportunité valide**\n\n"

    msg += "🔍 **RADAR :**\n"
    for t in TICKERS:
        if t in top_assets.index: msg += f"✅ {t}\n"
        elif prices.loc[today, t] < ma150.loc[today, t]: msg += f"❌ {t}\n"

    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += "📅 *Mise à jour Hebdo — Discipline > Émotion.*"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    print(msg)

if __name__ == "__main__":
    run()
