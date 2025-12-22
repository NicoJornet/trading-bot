import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# ==========================================
# CONFIGURATION STRATÉGIQUE
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Liste d'actifs diversifiés (Tech, Crypto, Défense, Or, Conso)
TICKERS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "TSLA", "AMD", "NFLX", "SMH", "QQQ",
    "BTC-USD", "ETH-USD", "LMT", "RTX", "XLI", "ITA",
    "WMT", "MCD", "KO", "COST", "PG", "XLP", "GLD", "USO", "NEM"
]

SAFE_ASSET = "TLT"         # Obligations (Refuge si MM200 cassée)
MARKET_INDEX = "SPY"       # S&P 500 (Indicateur de régime)
MAX_POSITIONS = 3          # On garde toujours le Top 3
LOOKBACK_MOMENTUM = 126    # 6 mois de recul
STOP_LOSS_COEF = 3.0       # Sécurité standard ATR

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erreur Telegram: {e}")

def get_signals():
    # 1. Récupération des données
    all_tickers = list(set(TICKERS + [SAFE_ASSET, MARKET_INDEX]))
    data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
    closes = data['Close'].ffill()
    
    last_prices = closes.iloc[-1]
    ma200_spy = closes[MARKET_INDEX].rolling(200).mean().iloc[-1]
    is_bull_market = last_prices[MARKET_INDEX] > ma200_spy
    
    # 2. Analyse du Momentum
    # Calcul de la performance sur 6 mois
    ret_mom = (last_prices / closes.iloc[-LOOKBACK_MOMENTUM]) - 1
    # Filtre de tendance (Prix > MM50)
    ma50 = closes.rolling(50).mean().iloc[-1]
    
    # 3. Construction du message
    today = datetime.date.today()
    msg = f"🏛️ *BOT ALGO ELITE - SIGNAL DU {today}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if not is_bull_market:
        msg += "📈 *RÉGIME : BAISSIER (🔴)*\n"
        msg += "⚠️ *ACTION : SÉCURITÉ MAXIMALE*\n\n"
        msg += "1️⃣ **VENDRE** : Tout ton portefeuille (Actions & Cryptos).\n"
        
        ma50_tlt = closes[SAFE_ASSET].rolling(50).mean().iloc[-1]
        if last_prices[SAFE_ASSET] > ma50_tlt:
            msg += f"2️⃣ **ACHETER** : {SAFE_ASSET} (Obligations) pour 100% du capital.\n"
        else:
            msg += "2️⃣ **RESTER EN CASH** : Attendre un signal de reprise.\n"
    else:
        msg += "📈 *RÉGIME : HAUSSIER (🟢)*\n"
        msg += "🚀 *ACTIONS À MENER IMMÉDIATEMENT :*\n\n"
        
        valid_assets = ret_mom[(ret_mom.index.isin(TICKERS)) & (last_prices > ma50)]
        top_assets = valid_assets.sort_values(ascending=False).head(MAX_POSITIONS)
        
        msg += "1️⃣ **VENDRE** : Tout actif qui n'est pas dans la liste ci-dessous.\n\n"
        msg += "2️⃣ **ACHETER / RÉÉQUILIBRER** :\n"
        
        for t, score in top_assets.items():
            price = last_prices[t]
            # Calcul ATR simplifié pour le Stop Loss
            vol = closes[t].pct_change().abs().tail(14).mean()
            stop_price = price - (vol * STOP_LOSS_COEF * price)
            
            msg += f"• *{t}* ➡️ Acheter pour **33%** du capital\n"
            msg += f"  └ 🚀 Momentum : *+{score:.1%}*\n"
            msg += f"  └ 🛡️ Stop Loss : *{stop_price:.2f}$*\n\n"

    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += "💬 *Note : Ajoute tes 200€ de DCA au capital total avant de diviser par 3.*"
    
    send_telegram(msg)

if __name__ == "__main__":
    get_signals()
