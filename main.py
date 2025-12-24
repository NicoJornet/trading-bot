import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ============================================================
# 1. CONFIGURATION
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

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def run():
    # --- 1. RÉCUPÉRATION DES DONNÉES (FLAT STRUCTURE) ---
    try:
        raw_data = yf.download(TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX"], 
                               period="2y", auto_adjust=True, progress=False)
        if raw_data.empty: return
        
        # Extraction propre des colonnes
        close = raw_data["Close"].ffill()
        high = raw_data["High"].ffill()
        low = raw_data["Low"].ffill()
    except Exception as e:
        print(f"Erreur technique : {e}")
        return
    
    # On ne garde que les actifs disponibles
    valid_assets = [t for t in TICKERS if t in close.columns]
    prices = close[valid_assets]
    today = prices.index[-1]
    
    if len(close) < 260: return

    # Taux de change EUR/USD
    fx = 1 / close["EURUSD=X"].iloc[-1] if "EURUSD=X" in close.columns else 1

    # --- 2. RÉGIME DE MARCHÉ (PROTECTION ANTI-KRACH) ---
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    ma200 = spy.rolling(200).mean()
    vix_med = vix.rolling(252).median()
    
    # Score de 0 à 3 basé sur Tendance (Prix > MA200) + Santé (Pente MA200) + Peur (VIX)
    score = int(spy.iloc[-1] > ma200.iloc[-1]) + \
            int(ma200.diff(20).iloc[-1] > 0) + \
            int(vix.iloc[-1] < vix_med.iloc[-1])
    
    exposure = {0:0.25, 1:0.50, 2:0.75, 3:1.0}[score]
    regime_txt = {0:"🔴 BEAR", 1:"🟠 CAUTION", 2:"🟡 BULL", 3:"🟢 STRONG BULL"}[score]

    # --- 3. SÉLECTION MOMENTUM + RSI (AGRESSIF 80) ---
    m = (0.2*(prices/prices.shift(63)-1) + 0.3*(prices/prices.shift(126)-1) + 0.5*(prices/prices.shift(252)-1))
    momentum = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(min=0.001)
    
    rsi_vals = prices.apply(rsi).iloc[-1]
    ma150 = prices.rolling(150).mean()
    
    # Filtre de sélection : Tendance haussière + Momentum + RSI < 80
    valid = (prices.iloc[-1] > ma150.iloc[-1]) & (momentum.iloc[-1] > 0) & (rsi_vals < 80)
    candidates = momentum.iloc[-1][valid].nlargest(6)

    # --- 4. DIVERSIFICATION (FILTRE CORRÉLATION) ---
    selected = []
    returns = prices.pct_change(fill_method=None)
    for t in candidates.index:
        if not selected:
            selected.append(t)
        else:
            # Vérifie que le nouvel actif n'est pas corrélé à plus de 0.80 aux déjà choisis
            current_corr = returns[selected + [t]].iloc[-60:].corr().iloc[-1]
            if current_corr.loc[selected].max() < 0.80:
                selected.append(t)
        
        max_slots = 1 if exposure <= 0.25 else TOP_MAX
        if len(selected) == max_slots: break

    # --- 5. CALCUL ALLOCATIONS & STOPS ATR ---
    msg = "━━━━━━━━━━━━━━━━━━━━\n🚀 **APEX AGGRESSIVE v10.5**\n━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📊 RÉGIME : {regime_txt} ({int(exposure*100)}%)\n"
    msg += f"⚡ CONFIG : RSI < 80 (Aggressive)\n\n"

    if selected:
        vol_ann = returns.rolling(252).std() * np.sqrt(252)
        inv_vol = 1 / vol_ann.iloc[-1][selected].clip(lower=0.1)
        weights = (inv_vol / inv_vol.sum()) * exposure

        msg += "🎯 **ALLOCATION & STOPS :**\n"
        for t in selected:
            w = weights[t]
            p_eur = prices[t].iloc[-1] * (1 if t.endswith(".PA") else fx)
            
            # Calcul ATR pour Stop dynamique (Sensibilité 3.0)
            ticker_tr = pd.concat([high[t]-low[t], abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))], axis=1).max(axis=1)
            ticker_atr = ticker_tr.rolling(14).mean().iloc[-1]
            ticker_ma100 = prices[t].rolling(100).mean().iloc[-1]
            
            st_raw = prices[t].iloc[-1] - (3.0 * ticker_atr)
            st_final = max(st_raw, ticker_ma100)
            st_eur = st_final * (1 if t.endswith(".PA") else fx)

            msg += f"• **{t}** : **{w*100:.1f}%**\n"
            msg += f"  Prix : {p_eur:.2f}€ | 🛡️ **STOP : {st_eur:.2f}€**\n\n"
    else:
        msg += "⚠️ **TOTAL CASH — Sécurité maximale**\n"
        msg += "(Marché en surchauffe totale RSI > 80)\n\n"

    msg += "━━━━━━━━━━━━━━━━━━━━\n📅 *Discipline > Chance*"

    # Envoi Telegram
    if TOKEN and CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                          data={"chat_id":CHAT_ID,"text":msg,"parse_mode":"Markdown"}, 
                          timeout=10)
        except:
            print("Erreur d'envoi Telegram")
    print(msg)

if __name__ == "__main__":
    run()
