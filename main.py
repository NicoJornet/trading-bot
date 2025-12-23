import yfinance as yf
import pandas as pd
import requests
import os

# --- 1. CONFIGURATION ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DCA_MENSUEL = 200
TICKERS = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "BTC-USD", "ETH-USD", "GLD", "NEM"]
MARKET_INDEX = "SPY"

def get_data():
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y")['Close'].ffill()
    
    # Régime de Marché
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if data[MARKET_INDEX].iloc[-1] > ma200_spy else "PRUDENCE / CASH (🔴)"
    
    # Santé technique (MA50)
    ma50 = data[TICKERS].rolling(window=50).mean().iloc[-1]
    prices_now = data[TICKERS].iloc[-1]
    
    # Momentum 6 mois
    returns = ((prices_now / data[TICKERS].iloc[-126]) - 1) * 100
    radar = returns.sort_values(ascending=False)
    
    # Top 3 (doit être sain > MA50)
    assets_sains = [t for t in TICKERS if prices_now[t] > ma50[t]]
    top_3 = returns[assets_sains].nlargest(3)
    
    # Alertes Sortie (sous MA50)
    sorties = [t for t in TICKERS if prices_now[t] < ma50[t]]
    
    # Volatilité ATR
    volatility = data[TICKERS].pct_change().rolling(window=14).std() * 100
    
    return regime, top_3, sorties, radar, prices_now, usd_to_eur, volatility.iloc[-1]

def format_and_send():
    regime, top_3, sorties, radar, prices_usd, fx_rate, current_vol = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **ALGO ELITE V5.7 PRO (€)**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🟢 **TOP 3 ACHATS (MOMENTUM) :**\n"
        for ticker, momentum in top_3.items():
            price_eur = prices_usd[ticker] * fx_rate
            dist_stop = max(min(current_vol[ticker] * 3, 15), 5) 
            stop_eur = price_eur * (1 - (dist_stop / 100))
            msg += f"• **{ticker}** : {price_eur:.2f}€ | 🛑 Stop : {stop_eur:.2f}€\n"
    
    msg += "\n🔴 **ALERTES SORTIE (VENDRE SI DÉTENU) :**\n"
    msg += f"{', '.join(sorties) if sorties else 'Aucune. Santé OK ✅'}\n"
    
    msg += "\n🔍 **DASHBOARD RADAR :**\n"
    for ticker, momentum in radar.items():
        status = "✅" if ticker in top_3.index else ("⚪" if ticker not in sorties else "❌")
        msg += f"{status} {ticker} : {momentum:+.1f}%\n"
        
    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA : {DCA_MENSUEL}€** | *V5.7 Pro Active*\n"
    
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    format_and_send()
