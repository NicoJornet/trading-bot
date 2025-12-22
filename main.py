import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# ==========================================
# CONFIGURATION ÉLITE V5.2 (ALERTE 40%)
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 🚨 AJOUTE ICI TES POSITIONS ACTUELLES (Exemple)
# Si tu as 1.5 Nvidia et 0.02 Bitcoin, écris-le ici :
MY_PORTFOLIO = {
    "NVDA": 10.5,    # Nombre d'actions
    "BTC-USD": 0.05, # Nombre de BTC
    "META": 15.2     # Nombre d'actions
}
CASH_DISPO = 200 # Ton cash non investi sur Trade Republic

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
    all_tickers = list(set(TICKERS + ["SPY", "TLT"] + list(MY_PORTFOLIO.keys())))
    data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
    closes = data['Close'].ffill()
    last_prices = closes.iloc[-1]
    
    # 1. Calcul de la valeur réelle du portefeuille
    total_val = CASH_DISPO
    portfolio_details = {}
    for t, qty in MY_PORTFOLIO.items():
        val = qty * last_prices[t]
        portfolio_details[t] = val
        total_val += val

    # 2. Analyse du marché
    ma200_spy = closes["SPY"].rolling(200).mean().iloc[-1]
    is_bull_market = last_prices["SPY"] > ma200_spy
    
    msg = f"🏛️ *BOT ALGO ELITE V5.2 - {datetime.date.today()}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if not is_bull_market:
        msg += "📉 *RÉGIME : SÉCURITÉ (🔴)*\n👉 *PASSAGE EN CASH*\n"
    else:
        msg += "📈 *RÉGIME : HAUSSIER (🟢)*\n\n"
        
        # 3. Alerte Rééquilibrage
        msg += "⚖️ *ÉTAT DE TES LIGNES :*\n"
        for t, val in portfolio_details.items():
            poids = val / total_val
            statut = "✅"
            if poids >= REBALANCE_THRESHOLD:
                statut = "⚠️ *ALERTE (TROP LOURD)*"
            msg += f"• {t} : `{poids:.1%}` du total {statut}\n"
        
        # 4. Momentum et Top 3
        ret_mom = (last_prices / closes.iloc[-126]) - 1
        ma50 = closes.rolling(50).mean().iloc[-1]
        valid_assets = ret_mom[(ret_mom.index.isin(TICKERS)) & (last_prices > ma50)]
        top_assets = valid_assets.sort_values(ascending=False).head(3)
        
        msg += "\n🏆 *TOP 3 MOMENTUM :*\n"
        for t, score in top_assets.items():
            price = last_prices[t]
            vol = closes[t].pct_change().abs().tail(14).mean()
            stop_price = price - (vol * STOP_LOSS_ATR * price)
            msg += f"• *{t}* : *{price:.2f}$* (Mom: `+{score:.1%}`)\n"
            msg += f"  └ Stop Loss : *{stop_price:.2f}$*\n"

    send_telegram(msg)

if __name__ == "__main__":
    get_signals()
