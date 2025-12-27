import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import sys
from datetime import datetime

# ============================================================
# APEX SCANNER - TELEGRAM EDITION
# ============================================================

DATABASE = [
    "NVDA","MSFT","GOOGL","AMZN","AAPL","META","TSLA","AVGO","AMD","MU",
    "ASML","TSM","ARM","LRCX","AMAT","PLTR","APP","CRWD","PANW","NET",
    "DDOG","ZS","SNOW","RKLB","SHOP","ABNB","VRT","SMCI","UBER",
    "COIN","MSTR","MARA","RIOT"
]

VOLATILE_SET = ["COIN", "MSTR", "MARA", "RIOT", "RKLB", "SMCI", "TSLA", "AMD", "NVDA", "APP"]

def send_telegram(message):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("⚠️ Erreur : Les secrets TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID ne sont pas configurés dans GitHub.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Message Telegram envoyé !")
        else:
            print(f"❌ Erreur envoi Telegram : {response.text}")
    except Exception as e:
        print(f"❌ Erreur connexion Telegram : {e}")

def run_scan():
    print(f"1. Démarrage du scan...")
    try:
        data = yf.download(DATABASE, period="6mo", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                close = data['Close']
            except KeyError:
                close = data['Adj Close']
        else:
            close = data
    except Exception as e:
        print(f"❌ Erreur data : {e}")
        return

    scores = {}
    current_prices = {}
    
    for ticker in DATABASE:
        try:
            if ticker not in close.columns: continue
            series = close[ticker].dropna()
            if len(series) < 50: continue
            
            # Score Momentum 3 mois
            r3 = series.iloc[-1] / series.iloc[-min(60, len(series)-1)] - 1
            scores[ticker] = r3
            current_prices[ticker] = series.iloc[-1]
        except:
            continue

    if not scores:
        print("❌ Aucun score.")
        return

    # Classement
    df_scores = pd.Series(scores).sort_values(ascending=False)
    
    # --- PRÉPARATION DU MESSAGE TELEGRAM ---
    top_picks = df_scores.head(2)
    
    msg = f"🚀 *APEX SCANNER* | {datetime.now().strftime('%d/%m')}\n"
    msg += f"🔥 *TOP 2 ACTIONS À ACHETER :*\n\n"
    
    rank = 1
    for ticker, score in top_picks.items():
        price = current_prices[ticker]
        sl_pct = 0.15 if ticker in VOLATILE_SET else 0.20
        stop_price = price * (1 - sl_pct)
        
        msg += f"*{rank}. {ticker}*\n"
        msg += f"💰 Prix : {price:.2f}$\n"
        msg += f"📈 Force : +{score*100:.1f}%\n"
        msg += f"🛡️ Stop Loss : {stop_price:.2f}$\n\n"
        rank += 1
    
    msg += "⚠️ _Ceci n'est pas un conseil en investissement._"

    # Affichage dans les logs GitHub
    print(msg)
    
    # Envoi sur Telegram
    send_telegram(msg)

if __name__ == "__main__":
    run_scan()
