import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# APEX SCANNER - GITHUB EDITION
# ============================================================

# 1. CONFIGURATION
# ----------------
# Univers d'investissement (Trade Republic Large)
DATABASE = [
    "NVDA","MSFT","GOOGL","AMZN","AAPL","META","TSLA","AVGO","AMD","MU",
    "ASML","TSM","ARM","LRCX","AMAT","PLTR","APP","CRWD","PANW","NET",
    "DDOG","ZS","SNOW","RKLB","SHOP","ABNB","VRT","SMCI","UBER",
    "COIN","MSTR","MARA","RIOT",
    "MC.PA","RMS.PA","OR.PA","SAP","AIR.PA","BNP.PA",
    "LLY","NVO","UNH","JNJ","ABBV","TMO","DHR","ISRG",
    "WMT","COST","PG","KO","PEP","XLE","XOM","CVX",
    "QQQ","SPY","GLD","SLV","TLT"
]

VOLATILE_SET = ["COIN", "MSTR", "MARA", "RIOT", "RKLB", "SMCI", "TSLA", "AMD", "NVDA", "APP"]

def get_momentum_score(prices):
    """Calcul du score APEX (Moyenne ROC 3 mois + 6 mois)"""
    # On s'assure d'avoir assez de données
    if len(prices) < 130: return -999
    
    r3 = prices.iloc[-1] / prices.iloc[-63] - 1
    r6 = prices.iloc[-1] / prices.iloc[-126] - 1
    return (r3 + r6) / 2

def run_scan():
    print(f"🚀 APEX SCANNER | Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    # 1. Téléchargement des données (6 mois + buffer)
    print("📡 Téléchargement des données de marché...")
    try:
        data = yf.download(DATABASE, period="1y", auto_adjust=True, progress=False)
        # Gestion multi-index de yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
    except Exception as e:
        print(f"❌ Erreur critique téléchargement: {e}")
        return

    # 2. Calcul des Scores
    scores = {}
    current_prices = {}
    
    for ticker in DATABASE:
        try:
            if ticker not in close.columns: continue
            series = close[ticker].dropna()
            if series.empty: continue
            
            score = get_momentum_score(series)
            scores[ticker] = score
            current_prices[ticker] = series.iloc[-1]
        except:
            continue
            
    # Création du classement
    df_scores = pd.Series(scores).sort_values(ascending=False)
    top_5 = df_scores.head(5)
    
    print("\n" + "="*60)
    print(f"🏆 TOP 2 ACTIFS À ACHETER (Si tu as du Cash)")
    print("="*60)
    
    rank = 1
    for ticker, score in top_5.head(2).items():
        price = current_prices[ticker]
        # Définition du Stop Loss
        sl_pct = 0.15 if ticker in VOLATILE_SET else 0.20
        stop_loss_price = price * (1 - sl_pct)
        
        print(f"#{rank} {ticker}")
        print(f"   ► Prix Actuel : {price:.2f}")
        print(f"   ► Score APEX  : {score:.4f}")
        print(f"   🛡️ STOP LOSS À PLACER : {stop_loss_price:.2f} (-{sl_pct*100:.0f}%)")
        print("-" * 30)
        rank += 1
        
    print("\n🧐 SURVEILLANCE (Remplaçants potentiels)")
    for ticker, score in top_5.iloc[2:].items():
        print(f"   - {ticker} (Score: {score:.4f})")

if __name__ == "__main__":
    run_scan()
