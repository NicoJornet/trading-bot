import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ============================================================
# CONFIGURATION
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOP_MAX = 3
TICKERS = [
    "NVDA","MSFT","GOOGL","AAPL","TSLA","SMH",
    "BTC-USD","ETH-USD","SOL-USD",
    "MC.PA","RMS.PA","RACE",
    "LLY","UNH",
    "URNM","COPX","XLE","ITA","GLD"
]
MARKET_INDEX = "SPY"

# ============================================================
def run():
    # 1. RÉCUPÉRATION DES DONNÉES
    try:
        # Téléchargement large pour calcul des indicateurs longs (MA200, Z-Score)
        data = yf.download(
            TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX"],
            period="2y",
            auto_adjust=True,
            progress=False
        )["Close"].ffill()
    except Exception as e:
        print(f"Erreur Yahoo Finance : {e}")
        return

    if len(data) < 260:
        print("Erreur : Historique insuffisant pour le calcul des indicateurs.")
        return

    prices = data[TICKERS]
    spy = data[MARKET_INDEX]
    vix = data["^VIX"]
    fx = 1 / data["EURUSD=X"].iloc[-1] # Conversion pour Trade Republic
    today = data.index[-1]
    returns = prices.pct_change()

    # 2. RÉGIME DE MARCHÉ (VARIATEUR D'EXPOSITION)
    ma200 = spy.rolling(200).mean()
    slope = ma200.diff(20)
    vix_med = vix.rolling(252).median()

    # Calcul du score de confiance (0 à 3)
    score = (
        int(spy.loc[today] > ma200.loc[today]) +
        int(slope.loc[today] > 0) +
        int(vix.loc[today] < vix_med.loc[today])
    )

    exposure_map = {0:0.25, 1:0.50, 2:0.75, 3:1.0}
    exposure = exposure_map[score]
    regime_txt = {0:"🔴 BEAR", 1:"🟠 CAUTION", 2:"🟡 BULL", 3:"🟢 STRONG BULL"}[score]

    # 3. MOMENTUM Z-SCORE (SÉLECTION DES LEADERS)
    # Calcul pondéré : 20% court terme, 30% moyen terme, 50% long terme
    m = (
        0.2*(prices/prices.shift(63)-1) +
        0.3*(prices/prices.shift(126)-1) + 
        0.5*(prices/prices.shift(252)-1)
    )
    # Normalisation par rapport au groupe (Z-Score)
    momentum = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1)

    ma150 = prices.rolling(150).mean()
    valid = (prices.loc[today] > ma150.loc[today]) & (momentum.loc[today] > 0)
    candidates = momentum.loc[today][valid].nlargest(6)

    # 4. FILTRE DE CORRÉLATION (DIVERSIFICATION INTELLIGENTE)
    selected = []
    for t in candidates.index:
        if not selected:
            selected.append(t)
        else:
            # Vérification de la corrélation sur les 60 derniers jours
            corr = returns[selected + [t]].iloc[-60:].corr().iloc[-1]
            if corr.loc[selected].max() < 0.80:
                selected.append(t)
        
        # En régime de crise, on ne garde qu'un seul leader
        max_slots = 1 if exposure <= 0.25 else TOP_MAX
        if len(selected) == max_slots:
            break

    # 5. ALLOCATION (INVERSE VOLATILITY)
    weights = pd.Series(dtype=float)
    if selected:
        vol_ann = returns.rolling(252).std() * np.sqrt(252)
        inv_vol = 1 / vol_ann.loc[today, selected].clip(lower=0.1)
        weights = (inv_vol / inv_vol.sum()) * exposure

    # 6. CONSTRUCTION DU MESSAGE TELEGRAM
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **APEX HYBRID v10.2**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📊 RÉGIME : {regime_txt} ({int(exposure*100)}%)\n\n"

    if not weights.empty:
        ma100 = prices.rolling(100).mean()
        hwm_55 = prices.rolling(55).max()
        vol_d = returns.rolling(20).std() * 100

        msg += "🎯 **ALLOCATION & STOPS :**\n"
        for t, w in weights.items():
            price_eur = prices.loc[today, t] * (1 if t.endswith(".PA") else fx)
            
            # Stop adaptatif : plus large en période de stress
            mult = 2.2 if exposure > 0.5 else 3.0
            buffer = np.clip(vol_d.loc[today, t] * mult, 6, 20)

            stop = max(ma100.loc[today, t], hwm_55.loc[today, t] * (1 - buffer/100))
            stop_eur = stop * (1 if t.endswith(".PA") else fx)

            msg += f"• **{t}** : **{w*100:.1f}%**\n"
            msg += f"  Prix : {price_eur:.2f}€ | 🛡️ **STOP : {stop_eur:.2f}€**\n\n"
    else:
        msg += "⚠️ **TOTAL CASH — Sécurité maximale**\n\n"

    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "📅 *Process > Conviction*"

    if TOKEN and CHAT_ID:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id":CHAT_ID, "text":msg, "parse_mode":"Markdown"},
            timeout=10
        )
    print(msg)

if __name__ == "__main__":
    run()
