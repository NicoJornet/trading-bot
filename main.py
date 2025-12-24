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
    # --- 1. RÉCUPÉRATION DES DONNÉES ---
    try:
        # On télécharge tout d'un coup
        raw_data = yf.download(TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX"], period="2y", auto_adjust=True, progress=False)
        
        if raw_data.empty:
            print("Erreur : Aucune donnée reçue.")
            return

        # Extraction propre des colonnes pour éviter l'IndexingError
        close = raw_data["Close"].ffill()
        high = raw_data["High"].ffill() if "High" in raw_data else close
        low = raw_data["Low"].ffill() if "Low" in raw_data else close
        
    except Exception as e:
        print(f"Erreur technique : {e}")
        return
    
    # Sécurité : on ne garde que les tickers qui ont bien été téléchargés
    available_tickers = [t for t in TICKERS if t in close.columns]
    prices = close[available_tickers]
    
    if len(close) < 260: return

    fx_rate = close["EURUSD=X"].iloc[-1]
    fx = 1 / fx_rate if fx_rate > 0 else 1
    today = close.index[-1]
    returns = prices.pct_change(fill_method=None) # Correction du warning FutureWarning

    # --- 2. RÉGIME DE MARCHÉ ---
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    ma200 = spy.rolling(200).mean()
    vix_med = vix.rolling(252).median()
    
    score = int(spy.loc[today] > ma200.loc[today]) + \
            int(ma200.diff(20).loc[today] > 0) + \
            int(vix.loc[today] < vix_med.loc[today])
    
    exposure = {0:0.25, 1:0.50, 2:0.75, 3:1.0}[score]
    regime_txt = {0:"🔴 BEAR", 1:"🟠 CAUTION", 2:"🟡 BULL", 3:"🟢 STRONG BULL"}[score]

    # --- 3. SÉLECTION MOMENTUM + RSI ---
    m = (0.2*(prices/prices.shift(63)-1) + 0.3*(prices/prices.shift(126)-1) + 0.5*(prices/prices.shift(252)-1))
    momentum = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(min=0.001)
    
    rsi_vals = prices.apply(rsi).loc[today]
    ma150 = prices.rolling(150).mean()
    
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

        # Calcul ATR et MA100
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr_all = tr.rolling(14).mean()
        ma100_all = prices.rolling(100).mean()

        msg += "🎯 **ALLOCATION & STOPS :**\n"
        for t in selected:
            w = weights[t]
            p_eur = prices.loc[today, t] * (1 if t.endswith(".PA") else fx)
            
            # Correction ici pour l'IndexingError
            current_atr = atr_all[t].loc[today]
            current_ma100 = ma100_all[t].loc[today]
            
            st_raw = prices.loc[today, t] - (3.0 * current_atr)
            st_final = max(st_raw, current_ma100)
            st_eur = st_final * (1 if t.endswith(".PA") else fx)

            msg += f"• **{t}** : **{w*100:.1f}%**\n"
            msg += f"  Prix : {p_eur:.2f}€ | 🛡️ **STOP : {st_eur:.2f}€**\n\n"
    else:
        msg += "⚠️ **TOTAL CASH — Attente**\n"
        msg += "(Marché surchauffé ou pas de leader sain)\n\n"

    msg += "━━━━━━━━━━━━━━━━━━━━\n📅 *Discipline > Chance*"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id":CHAT_ID,"text":msg,"parse_mode":"Markdown"})
    print(msg)

if __name__ == "__main__":
    run()
