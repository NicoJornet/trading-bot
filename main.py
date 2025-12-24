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
    # --- 1. RÉCUPÉRATION DES DONNÉES AVEC SÉCURITÉ ---
    try:
        data = yf.download(TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX"], period="2y", auto_adjust=True, progress=False)
        
        if data.empty:
            print("Erreur : Aucune donnée reçue de Yahoo Finance.")
            return

        # Gestion Multi-Index de yfinance
        close = data["Close"].ffill()
        # Sécurité pour High/Low : si absents, on utilise Close pour éviter le plantage
        high = data["High"].ffill() if "High" in data else close
        low = data["Low"].ffill() if "Low" in data else close
        
    except Exception as e:
        print(f"Erreur technique lors du téléchargement : {e}")
        return
    
    if len(close) < 260:
        print("Erreur : Historique insuffisant.")
        return

    prices = close[TICKERS]
    fx_rate = close["EURUSD=X"].iloc[-1]
    fx = 1 / fx_rate if fx_rate > 0 else 1
    today = close.index[-1]
    returns = prices.pct_change()

    # --- 2. RÉGIME DE MARCHÉ HYBRIDE ---
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    ma200 = spy.rolling(200).mean()
    vix_med = vix.rolling(252).median()
    
    score = int(spy.loc[today] > ma200.loc[today]) + \
            int(ma200.diff(20).loc[today] > 0) + \
            int(vix.loc[today] < vix_med.loc[today])
    
    exposure = {0:0.25, 1:0.50, 2:0.75, 3:1.0}[score]
    regime_txt = {0:"🔴 BEAR", 1:"🟠 CAUTION", 2:"🟡 BULL", 3:"🟢 STRONG BULL"}[score]

    # --- 3. SÉLECTION MOMENTUM + FILTRE RSI ---
    m = (0.2*(prices/prices.shift(63)-1) + 0.3*(prices/prices.shift(126)-1) + 0.5*(prices/prices.shift(252)-1))
    momentum = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(min=0.001)
    
    rsi_vals = prices.apply(rsi).loc[today]
    ma150 = prices.rolling(150).mean()
    
    # Filtres : Tendance + Momentum + RSI sain (<70)
    valid = (prices.loc[today] > ma150.loc[today]) & (momentum.loc[today] > 0) & (rsi_vals < 70)
    candidates = momentum.loc[today][valid].nlargest(6)

    # --- 4. DIVERSIFICATION (CORRÉLATION) ---
    selected = []
    for t in candidates.index:
        if not selected:
            selected.append(t)
        else:
            current_corr = returns[selected + [t]].iloc[-60:].corr().iloc[-1]
            if current_corr.loc[selected].max() < 0.80:
                selected.append(t)
        
        max_slots = 1 if exposure <= 0.25 else TOP_MAX
        if len(selected) == max_slots: break

    # --- 5. CALCUL ALLOCATIONS & STOPS ---
    msg = "━━━━━━━━━━━━━━━━━━━━\n🏛️ **APEX HYBRID v10.5**\n━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📊 RÉGIME : {regime_txt} ({int(exposure*100)}%)\n\n"

    if selected:
        vol_ann = returns.rolling(252).std() * np.sqrt(252)
        inv_vol = 1 / vol_ann.loc[today, selected].clip(lower=0.1)
        weights = (inv_vol / inv_vol.sum()) * exposure

        # Calcul de l'ATR sécurisé
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr_all = tr.rolling(14).mean()
        ma100_all = prices.rolling(100).mean()

        msg += "🎯 **ALLOCATION & STOPS :**\n"
        for t in selected:
            w = weights[t]
            p_eur = prices.loc[today, t] * (1 if t.endswith(".PA") else fx)
            
            # Calcul du stop
            st_raw = prices.loc[today, t] - (3.0 * atr_all.loc[today, t])
            st_final = max(st_raw, ma100_all.loc[today, t])
            st_eur = st_final * (1 if t.endswith(".PA") else fx)

            msg += f"• **{t}** : **{w*100:.1f}%**\n"
            msg += f"  Prix : {p_eur:.2f}€ | 🛡️ **STOP : {st_eur:.2f}€**\n\n"
    else:
        msg += "⚠️ **TOTAL CASH — Attente opportunité**\n"
        msg += "(Marché surchauffé ou pas de leader sain)\n\n"

    msg += "━━━━━━━━━━━━━━━━━━━━\n📅 *Discipline > Chance*"

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
