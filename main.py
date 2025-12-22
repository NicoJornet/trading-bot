import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# ==========================================
# CONFIGURATION ÉLITE V5.0
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "TSLA", "AMD", "NFLX", "SMH", "QQQ",
    "BTC-USD", "ETH-USD", "LMT", "RTX", "XLI", "ITA",
    "WMT", "MCD", "KO", "COST", "PG", "XLP", "GLD", "USO", "NEM"
]

STOP_LOSS_ATR = 3.0
REBALANCE_THRESHOLD = 0.40  # Alerte si une ligne dépasse 40%

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

def get_signals():
    all_tickers = list(set(TICKERS + ["SPY", "TLT"]))
    data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
    closes = data['Close'].ffill()
    
    last_prices = closes.iloc[-1]
    ma200_spy = closes["SPY"].rolling(200).mean().iloc[-1]
    is_bull_market = last_prices["SPY"] > ma200_spy
    
    msg = f"🏛️ *BOT ALGO ELITE V5.0 - {datetime.date.today()}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if not is_bull_market:
        msg += "📉 *RÉGIME : SÉCURITÉ (🔴)*\n👉 *RESTER EN CASH OU TLT*\n"
    else:
        msg += "📈 *RÉGIME : HAUSSIER (🟢)*\n\n"
        # Calcul Momentum
        ret_mom = (last_prices / closes.iloc[-126]) - 1
        ma50 = closes.rolling(50).mean().iloc[-1]
        valid_assets = ret_mom[(ret_mom.index.isin(TICKERS)) & (last_prices > ma50)]
        top_assets = valid_assets.sort_values(ascending=False).head(3)
        
        msg += "🏆 *TOP 3 MOMENTUM :*\n"
        for t, score in top_assets.items():
            price = last_prices[t]
            vol = closes[t].pct_change().abs().tail(14).mean()
            stop_price = price - (vol * STOP_LOSS_ATR * price)
            msg += f"• *{t}* : Stop à *${stop_price:.2f}*\n"
        
        msg += f"\n⚖️ *RÈGLE DE GESTION :*\n"
        msg += f"Si une ligne dépasse *40%* de votre total, vendez le surplus pour revenir à 33% et sécuriser vos gains.\n"

    send_telegram(msg)

if __name__ == "__main__":
    get_signals()
