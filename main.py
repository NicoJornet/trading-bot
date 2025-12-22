import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# ==========================================
# 1. CONFIGURATION & SECRETS
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKERS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "TSLA", "AMD", "NFLX", "SMH", "QQQ",
    "BTC-USD", "ETH-USD", "LMT", "RTX", "XLI", "ITA",
    "WMT", "MCD", "KO", "COST", "PG", "XLP", "GLD", "USO", "NEM"
]

SAFE_ASSET = "TLT"         
MARKET_INDEX = "SPY"
MAX_POSITIONS = 3
LOOKBACK_MOMENTUM = 126  

# ==========================================
# 2. FONCTIONS DE COMMUNICATION
# ==========================================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

# ==========================================
# 3. MOTEUR FINAL
# ==========================================
def get_signals():
    all_tickers = list(set(TICKERS + [SAFE_ASSET, MARKET_INDEX]))
    # On prend 16 ans pour stabiliser les moyennes mobiles
    data = yf.download(all_tickers, period="16y", auto_adjust=True, progress=False)
    closes = data['Close'].ffill()
    
    # --- ANALYSE PRÉSENTE ---
    last_prices = closes.iloc[-1]
    ma200_spy = closes[MARKET_INDEX].rolling(200).mean().iloc[-1]
    is_bull_market = last_prices[MARKET_INDEX] > ma200_spy
    
    # --- COMPARAISON (T-11 jours de bourse ~ 15 jours calendaires) ---
    prev_ma200_spy = closes[MARKET_INDEX].rolling(200).mean().iloc[-11]
    was_bull_market = prev_prices_index = closes[MARKET_INDEX].iloc[-11] > prev_ma200_spy

    msg = f"🏛️ *BOT ALGO ELITE - RAPPORT DU {datetime.date.today()}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if not is_bull_market:
        msg += "📉 *RÉGIME : MARCHÉ BAISSIER (🔴)*\n"
        ma50_tlt = closes[SAFE_ASSET].rolling(50).mean().iloc[-1]
        if last_prices[SAFE_ASSET] > ma50_tlt:
            msg += f"👉 *ORDRE : ACHETER {SAFE_ASSET}* (Refuge)\n"
        else:
            msg += "👉 *ORDRE : RESTER EN CASH* (Protection)\n"
    else:
        msg += "📈 *RÉGIME : MARCHÉ HAUSSIER (🟢)*\n\n"
        # Momentum + Filtre MM50
        ret_mom = (last_prices / closes.iloc[-LOOKBACK_MOMENTUM]) - 1
        ma50 = closes.rolling(50).mean().iloc[-1]
        valid_assets = ret_mom[(ret_mom.index.isin(TICKERS)) & (last_prices > ma50)]
        top_assets = valid_assets.sort_values(ascending=False).head(MAX_POSITIONS)
        
        if top_assets.empty:
            msg += "⚠️ Aucune action en tendance. Rester en Cash.\n"
        else:
            msg += "🏆 *TOP 3 MOMENTUM :*\n"
            for t, score in top_assets.items():
                price = last_prices[t]
                # Calcul du Stop Loss ATR (3.0)
                vol = closes[t].pct_change().abs().tail(14).mean()
                stop_price = price - (vol * 3.0 * price)
                msg += f"• *{t}* (Mom: {score:+.1%})\n  └ Prix: ${price:.2f} | Stop: *${stop_price:.2f}*\n"

    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🔄 *ÉTAT : STABLE*" if is_bull_market == was_bull_market else "⚡ *ALERTE : CHANGEMENT DE RÉGIME*"
    
    send_telegram(msg)

if __name__ == "__main__":
    get_signals()
