import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# 1. CONFIGURATION
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOP_MAX = 3
TICKERS = [
    "NVDA","MSFT","GOOGL","AAPL","TSLA","SMH","PLTR","ASML", # IA & Tech
    "BTC-USD","ETH-USD","SOL-USD",                          # Cryptos
    "MC.PA","RMS.PA","RACE",                                # Luxe
    "LLY","UNH","ISRG",                                     # Santé & Robotique (Ajout ISRG)
    "PANW",                                                 # Cybersécurité (Ajout PANW)
    "URNM","COPX","XLE","ALB",                              # Énergie & Métaux Tech (Ajout ALB)
    "GLD","SIL",                                            # Métaux Précieux (Ajout SIL)
    "ITA"                                                   # Défense
]
MARKET_INDEX = "SPY"

# Paramètres de gestion du risque Professional v11.0
MAX_CRYPTO_WEIGHT = 0.20  
MAX_SINGLE_POSITION = 0.35  

# ============================================================
# 2. FONCTIONS
# ============================================================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_market_breadth(prices, ma_period=50):
    above_ma = (prices > prices.rolling(ma_period).mean()).sum(axis=1)
    return above_ma / len(prices.columns)

def run():
    # --- 1. RÉCUPÉRATION DES DONNÉES ---
    try:
        raw_data = yf.download(TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX"], 
                               period="2y", auto_adjust=True, progress=False)
        if raw_data.empty: return
        
        close = raw_data["Close"].ffill()
        high = raw_data["High"].ffill()
        low = raw_data["Low"].ffill()
    except Exception as e:
        print(f"Erreur : {e}"); return
    
    valid_assets = [t for t in TICKERS if t in close.columns]
    prices = close[valid_assets]
    fx = 1 / close["EURUSD=X"].iloc[-1] if "EURUSD=X" in close.columns else 1

    # --- 2. RÉGIME DE MARCHÉ AMÉLIORÉ ---
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    ma200 = spy.rolling(200).mean()
    vix_med = vix.rolling(252).median()
    breadth = calculate_market_breadth(prices, ma_period=50)

    # Score de 0 à 4 (Breadth inclus)
    score = int(spy.iloc[-1] > ma200.iloc[-1]) + \
            int(ma200.diff(20).iloc[-1] > 0) + \
            int(vix.iloc[-1] < vix_med.iloc[-1]) + \
            int(breadth.iloc[-1] > 0.60)

    exposure = {0:0.25, 1:0.40, 2:0.60, 3:0.80, 4:1.0}[score]
    regime_txt = {0:"🔴 BEAR", 1:"🟠 CAUTION", 2:"🟡 NEUTRAL", 3:"🟢 BULL", 4:"🚀 STRONG BULL"}[score]

    # --- 3. MOMENTUM & RSI 80 ---
    m = (0.2*(prices/prices.shift(63)-1) + 0.3*(prices/prices.shift(126)-1) + 0.5*(prices/prices.shift(252)-1))
    z_mom = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(min=0.001)
    rsi_vals = prices.apply(rsi).iloc[-1]
    ma150 = prices.rolling(150).mean()
    
    valid = (prices.iloc[-1] > ma150.iloc[-1]) & (z_mom.iloc[-1] > 0) & (rsi_vals < 80)
    candidates = z_mom.iloc[-1][valid].nlargest(6)

    # --- 4. DIVERSIFICATION & SÉLECTION ---
    selected = []
    returns = prices.pct_change(fill_method=None)
    for t in candidates.index:
        if not selected: selected.append(t)
        else:
            current_corr = returns[selected + [t]].iloc[-60:].corr().iloc[-1]
            if current_corr.loc[selected].max() < 0.80: selected.append(t)
        if len(selected) == (1 if exposure <= 0.25 else TOP_MAX): break

    # --- 5. ALLOCATION & MESSAGE ---
    msg = "━━━━━━━━━━━━━━━━━━━━━━━━\n🚀 **APEX TOTAL DOMINANCE v11.0**\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📊 RÉGIME : {regime_txt} ({int(exposure*100)}%)\n"
    msg += f"📈 Market Breadth : {breadth.iloc[-1]*100:.0f}%\n\n"

    if selected:
        vol_ann = returns.rolling(252).std() * np.sqrt(252)
        inv_vol = 1 / vol_ann.iloc[-1][selected].clip(lower=0.1)
        weights = (inv_vol / inv_vol.sum()) * exposure
        
        # Contraintes Crypto & Single Pos
        crypto = [t for t in selected if "USD" in t]
        if crypto and weights[crypto].sum() > MAX_CRYPTO_WEIGHT:
            weights[crypto] *= MAX_CRYPTO_WEIGHT / weights[crypto].sum()
        weights = weights.clip(max=MAX_SINGLE_POSITION)
        weights *= exposure / weights.sum()

        msg += "🎯 **ALLOCATION & STOPS :**\n"
        for t in selected:
            p_eur = prices[t].iloc[-1] * (1 if t.endswith(".PA") else fx)
            ticker_tr = pd.concat([high[t]-low[t], abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))], axis=1).max(axis=1)
            ticker_atr = ticker_tr.rolling(14).mean().iloc[-1]
            st_val = max(prices[t].iloc[-1] - (3.0 * ticker_atr), prices[t].rolling(100).mean().iloc[-1])
            st_eur = st_val * (1 if t.endswith(".PA") else fx)

            msg += f"• **{t}** : **{weights[t]*100:.1f}%**\n  Prix : {p_eur:.2f}€ | 🛡️ **STOP : {st_eur:.2f}€**\n\n"
    else:
        msg += "⚠️ **TOTAL CASH — Protection active**\n"

    msg += f"\n💰 Exposition Totale : {weights.sum()*100 if selected else 0:.0f}%\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n💡 *Discipline > Chance*"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                      data={"chat_id":CHAT_ID,"text":msg,"parse_mode":"Markdown"})
    print(msg)

if __name__ == "__main__":
    run()
