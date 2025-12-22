import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# ==========================================
# CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "TSLA", "AMD", "NFLX",
    "SMH", "QQQ", "BTC-USD", "ETH-USD", "LMT", "RTX", "XLI", "ITA",
    "WMT", "MCD", "KO", "COST", "PG", "XLP", "GLD", "USO", "NEM"
]

SAFE_ASSET = "TLT"
MARKET_INDEX = "SPY"
MAX_POSITIONS = 3
LOOKBACK_DAYS = 126

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload, timeout=10)
        print("Message sent to Telegram!")
    except Exception as e:
        print(f"Error sending message: {e}")

def get_signals():
    print("Analyzing market...")
    all_tickers = list(set(TICKERS + [SAFE_ASSET, MARKET_INDEX]))
    data = yf.download(all_tickers, period="1y", auto_adjust=True, progress=False)
    closes = data['Close'].ffill()
    last_prices = closes.iloc[-1]
    
    ma200_spy = closes[MARKET_INDEX].rolling(200).mean().iloc[-1]
    is_bull_market = last_prices[MARKET_INDEX] > ma200_spy
    
    msg = f"ALGO ELITE Report - {datetime.date.today()}\n"
    msg += "="*30 + "\n\n"
    
    if not is_bull_market:
        msg += "REGIME: BEARISH (CAUTION)\n"
        price_tlt = last_prices[SAFE_ASSET]
        ma50_tlt = closes[SAFE_ASSET].rolling(50).mean().iloc[-1]
        if price_tlt > ma50_tlt:
            msg += f"BUY: {SAFE_ASSET} (Safe Asset)"
        else:
            msg += "ACTION: STAY IN CASH"
    else:
        msg += "REGIME: BULLISH (ATTACK)\n\n"
        start_price = closes.iloc[-LOOKBACK_DAYS]
        returns = (last_prices / start_price) - 1
        ma50 = closes.rolling(50).mean().iloc[-1]
        valid_assets = returns[(returns.index.isin(TICKERS)) & (last_prices > ma50)]
        top_3 = valid_assets.sort_values(ascending=False).head(MAX_POSITIONS)
        
        if top_3.empty:
            msg += "WARNING: No valid signals. Stay in Cash."
        else:
            msg += "TOP 3 TO HOLD:\n"
            for ticker, score in top_3.items():
                price = last_prices[ticker]
                vol = closes[ticker].pct_change().abs().tail(14).mean()
                stop = price - (vol * 3.0 * price)
                msg += f"- {ticker} (Momentum: {score:+.1%})\n"
                msg += f"  Price: ${price:.2f} | Stop: ${stop:.2f}\n"
    
    msg += "\n" + "="*30
    send_telegram(msg)
    print("Signal sent!")

if __name__ == "__main__":
    get_signals()
