import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# ==========================================
# CONFIGURATION ÉLITE V5.1 (PRIX & MOMENTUM)
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "TSLA", "AMD", "NFLX", "SMH", "QQQ",
    "BTC-USD", "ETH-USD", "LMT", "RTX", "XLI", "ITA",
    "WMT", "MCD", "KO", "COST", "PG", "XLP", "GLD", "USO", "NEM"
]

STOP_LOSS_ATR = 3.0
REBALANCE_THRESHOLD = 0.40 

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

def get_signals():
    # On télécharge les données (1 an pour les moyennes mobiles et momentum)
    all_tickers = list(set(TICKERS + ["SPY", "TLT"]))
    data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
    closes = data['Close'].ffill()
    
    last_prices = closes.iloc[-1]
    ma200_spy = closes["SPY"].rolling(200).mean().iloc[-1]
    is_bull_market = last_prices["SPY"] > ma200_spy
    
    msg = f"🏛️ *BOT ALGO ELITE V5.1 - {datetime.date.today()}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if not is_bull_market:
        msg += "📉 *RÉGIME : SÉCURITÉ (🔴)*\n👉 *RESTER EN CASH OU TLT*\n"
    else:
        msg += "📈 *RÉGIME : HAUSSIER (🟢)*\n\n"
        
        # Calcul Momentum (Performance sur 6 mois / 126 jours de trading)
        ret_mom = (last_prices / closes.iloc[-126]) - 1
        ma50 = closes.rolling(50).mean().iloc[-1]
        
        # Filtrage : Doit être dans la liste, au-dessus de sa moyenne 50 jours
        valid_assets = ret_mom[(ret_mom.index.isin(TICKERS)) & (last_prices > ma50)]
        top_assets = valid_assets.sort_values(ascending=False).head(3)
        
        msg += "🏆 *TOP 3 MOMENTUM :*\n"
        for t, score in top_assets.items():
            price = last_prices[t]
            # Calcul du Stop Loss basé sur la volatilité réelle (ATR 14 jours)
            vol = closes[t].pct_change().abs().tail(14).mean()
            stop_price = price - (vol * STOP_LOSS_ATR * price)
            
            # Affichage : Ticker | Prix Actuel | Momentum % | Stop Loss
            msg += f"• *{t}* : *{price:.2f}$*\n"
            msg += f"  └ Momentum : `+{score:.1%}`\n"
            msg += f"  └ Stop Loss : *{stop_price:.2f}$*\n\n"
        
        msg += f"⚖️ *GESTION :* Si une ligne > *40%* du total, rééquilibrez à 33%.\n"

    send_telegram(msg)

if __name__ == "__main__":
    get_signals()
