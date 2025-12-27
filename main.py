import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# ============================================================
# APEX SCANNER - DEBUG EDITION
# ============================================================

DATABASE = [
    "NVDA","MSFT","GOOGL","AMZN","AAPL","META","TSLA","AVGO","AMD","MU",
    "ASML","TSM","ARM","LRCX","AMAT","PLTR","APP","CRWD","PANW","NET",
    "DDOG","ZS","SNOW","RKLB","SHOP","ABNB","VRT","SMCI","UBER",
    "COIN","MSTR","MARA","RIOT"
]

VOLATILE_SET = ["COIN", "MSTR", "MARA", "RIOT", "RKLB", "SMCI", "TSLA", "AMD", "NVDA", "APP"]

def run_scan():
    print(f"1. Démarrage du scan... ({len(DATABASE)} actifs)")
    
    # Téléchargement
    try:
        # On force le téléchargement en groupe
        data = yf.download(DATABASE, period="6mo", progress=False)
        
        # Gestion des colonnes MultiIndex (problème fréquent yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            # On essaye de récupérer 'Close' ou 'Adj Close'
            try:
                close = data['Close']
            except KeyError:
                close = data['Adj Close']
        else:
            close = data

        print(f"2. Données récupérées. Analyse en cours...")
        
    except Exception as e:
        print(f"❌ ERREUR TÉLÉCHARGEMENT : {e}")
        return

    scores = {}
    current_prices = {}
    
    # Calcul
    for ticker in DATABASE:
        try:
            # On vérifie si le ticker est bien dans les colonnes
            if ticker not in close.columns:
                continue
                
            series = close[ticker].dropna()
            if len(series) < 50: # Pas assez de données
                continue
            
            # Score simplifié (Momentum 3 mois) pour éviter les erreurs de calcul
            # Prix actuel / Prix il y a 60 jours
            r3 = series.iloc[-1] / series.iloc[-min(60, len(series)-1)] - 1
            
            scores[ticker] = r3
            current_prices[ticker] = series.iloc[-1]
        except Exception as e:
            continue

    if not scores:
        print("❌ AUCUN SCORE CALCULÉ. Vérifie la liste des tickers.")
        return

    # Classement
    df_scores = pd.Series(scores).sort_values(ascending=False)
    
    print("\n" + "="*50)
    print(f"🏆 RÉSULTAT DU {datetime.now().strftime('%d/%m/%Y')}")
    print("="*50)
    
    rank = 1
    # On affiche le TOP 3
    for ticker, score in df_scores.head(3).items():
        price = current_prices[ticker]
        sl_pct = 0.15 if ticker in VOLATILE_SET else 0.20
        stop_price = price * (1 - sl_pct)
        
        print(f"#{rank} {ticker}")
        print(f"   Prix: {price:.2f}$")
        print(f"   Force: {score*100:.1f}%")
        print(f"   Stop Loss suggéré: {stop_price:.2f}$")
        print("-" * 20)
        rank += 1

if __name__ == "__main__":
    run_scan()
