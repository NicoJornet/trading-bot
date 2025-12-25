import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# CONFIGURATION - APEX v17.0 PRODUCTION
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOP_MAX = 3
# La liste des 25 actifs stratégiques
TICKERS = [
    "NVDA","MSFT","GOOGL","AAPL","TSLA","SMH","PLTR","ASML", 
    "BTC-USD","ETH-USD","SOL-USD",                          
    "MC.PA","RMS.PA","RACE",                                
    "LLY","UNH","ISRG","PANW",                              
    "URNM","COPX","XLE","ALB","GLD","SIL","ITA"            
]
MARKET_INDEX = "SPY"

# Paramètres du Moteur v17 (Smart Aggressive)
MAX_CRYPTO_WEIGHT = 0.35   
MAX_SINGLE_POSITION = 0.40 
CORRELATION_THRESHOLD = 0.85

def run():
    # --- 1. CHARGEMENT DES DONNÉES ---
    try:
        # Téléchargement optimisé
        raw_data = yf.download(TICKERS + [MARKET_INDEX, "EURUSD=X"], 
                               period="2y", auto_adjust=True, progress=False)
        if raw_data.empty: return
        
        # Gestion de la structure de données (MultiIndex vs SingleIndex)
        if isinstance(raw_data.columns, pd.MultiIndex):
            close = raw_data['Close'].ffill()
            high = raw_data['High'].ffill()
            low = raw_data['Low'].ffill()
        else:
            close = raw_data['Close'].ffill()
            high = raw_data['High'].ffill()
            low = raw_data['Low'].ffill()
            
    except Exception as e:
        print(f"Data Error: {e}"); return
    
    # Filtrage des actifs valides
    valid_assets = [t for t in TICKERS if t in close.columns]
    prices = close[valid_assets]
    fx = 1 / close["EURUSD=X"].iloc[-1] if "EURUSD=X" in close.columns else 1.0

    # --- 2. LE PARACHUTE (MA200) ---
    # Si le SPY est sous sa moyenne 200 jours -> On passe 100% Cash
    spy = close[MARKET_INDEX]
    bull_market = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
    exposure = 1.0 if bull_market else 0.0

    # --- 3. SÉLECTION (Momentum v17) ---
    m = (0.2*(prices/prices.shift(63)-1) + 0.3*(prices/prices.shift(126)-1) + 0.5*(prices/prices.shift(252)-1))
    z_mom = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(min=0.001)
    
    rsi_vals = 100 - (100 / (1 + prices.diff().where(lambda x: x>0,0).rolling(14).mean() / -prices.diff().where(lambda x: x<0,0).rolling(14).mean()))
    
    valid = (prices.iloc[-1] > prices.rolling(150).mean().iloc[-1]) & (z_mom.iloc[-1] > 0) & (rsi_vals.iloc[-1] < 80)
    candidates = z_mom.iloc[-1][valid].nlargest(6)

    selected = []
    returns = prices.pct_change(fill_method=None)
    for t in candidates.index:
        if not selected: selected.append(t)
        else:
            if returns[selected + [t]].iloc[-126:].corr().iloc[-1].loc[selected].max() < CORRELATION_THRESHOLD:
                selected.append(t)
        if len(selected) == TOP_MAX: break

    # --- 4. CALCULS ET FORMATTAGE TELEGRAM ---
    msg = "🤖 **APEX v17 DUO**\n\n"

    if selected and exposure > 0:
        # Allocation Equal Weight (Le secret de la performance)
        w_val = 1.0 / len(selected)
        weights = pd.Series(w_val, index=selected) * exposure
        
        # Plafonds de sécurité
        crypto = [t for t in selected if "USD" in t]
        if crypto and weights[crypto].sum() > MAX_CRYPTO_WEIGHT:
            weights[crypto] *= MAX_CRYPTO_WEIGHT / weights[crypto].sum()
            
        weights = weights.clip(upper=MAX_SINGLE_POSITION)
        weights *= exposure / weights.sum()

        msg += "✅ **ACTIONS À DÉTENIR :**\n"
        for t in selected:
            # Prix actuel converti en EUR
            p_eur = prices[t].iloc[-1] * (1 if t.endswith(".PA") else fx)
            
            # Calcul du Stop Suiveur (4 ATR)
            # C'est un stop large pour laisser respirer l'action
            tr = np.maximum(high[t]-low[t], np.maximum(abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))))
            atr = tr.rolling(14).mean().iloc[-1]
            stop_level = prices[t].iloc[-1] - (4.0 * atr)
            stop_eur = stop_level * (1 if t.endswith(".PA") else fx)
            
            # Distance du stop en %
            risk = (p_eur - stop_eur) / p_eur * 100

            msg += f"🔹 **{t}**\n"
            msg += f"   📊 Alloc : **{weights[t]*100:.0f}%**\n"
            msg += f"   💰 Prix  : {p_eur:.2f}€\n"
            msg += f"   🛡️ **Stop Suiveur : {stop_eur:.2f}€** (-{risk:.1f}%)\n\n"
            
        msg += f"💵 Investissement total : {weights.sum()*100:.0f}%\n"
        msg += "ℹ️ *Sur Trade Republic : Si le 'Stop Suiveur' monte, annulez l'ancien stop et placez le nouveau.*"
        
    else:
        msg += "🛑 **MODE CASH (100% Liquide)**\n"
        msg += "Le marché est baissier (Sous Moyenne 200). On protège le capital."

    # Envoi Telegram
    if TOKEN and CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                          data={"chat_id":CHAT_ID,"text":msg,"parse_mode":"Markdown"})
        except Exception as e:
            print(f"Erreur Telegram: {e}")
    
    # Affichage console pour les logs GitHub
    print(msg)

if __name__ == "__main__":
    run()
