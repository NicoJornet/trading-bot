import yfinance as yf
import pandas as pd
import requests
import os

# --- 1. CONFIGURATION (Utilise les Secrets GitHub) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DCA_MENSUEL = 200

# Liste d'actifs surveillés (Tech, Crypto, Or, Mines)
TICKERS = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "BTC-USD", "ETH-USD", "GLD", "NEM"]
MARKET_INDEX = "SPY"

def get_data():
    # A. Taux de change EUR/USD
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    
    # B. Données historiques (1 an)
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y")['Close'].ffill()
    
    # C. Régime de Marché (MA200)
    current_spy = data[MARKET_INDEX].iloc[-1]
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if current_spy > ma200_spy else "PRUDENCE / CASH (🔴)"
    
    # D. Momentum Radar (6 mois)
    all_returns = ((data[TICKERS].iloc[-1] / data[TICKERS].iloc[-126]) - 1) * 100
    radar = all_returns.sort_values(ascending=False)
    top_3 = radar.head(3)
    
    # E. Volatilité Dynamique (ATR pour Stop Loss)
    # On calcule l'écart type des rendements sur 14 jours
    volatility = data[TICKERS].pct_change().rolling(window=14).std() * 100
    
    prices_usd = data[TICKERS].iloc[-1]
    
    return regime, top_3, radar, prices_usd, usd_to_eur, volatility.iloc[-1]

def format_and_send():
    regime, top_3, radar, prices_usd, fx_rate, current_vol = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **ALGO ELITE V5.5 DYNAMIQUE (€)**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🏆 **TOP 3 À DÉTENIR :**\n"
        for ticker, momentum in top_3.items():
            price_eur = prices_usd[ticker] * fx_rate
            
            # Calcul du Stop Dynamique (3x Volatilité, min 5%, max 15%)
            dist_stop = max(min(current_vol[ticker] * 3, 15), 5) 
            stop_eur = price_eur * (1 - (dist_stop / 100))
            
            msg += f"• **{ticker}** : {price_eur:.2f}€\n"
            msg += f"  └ 🔥 Mom : +{momentum:.1f}% | 🛑 Stop : {stop_eur:.2f}€ (-{dist_stop:.1f}%)\n"
    else:
        msg += "⚠️ **SIGNAL CASH GUARD ACTIVÉ**\n"
        msg += "Le marché est sous sa MA200. Protégez votre capital.\n"
    
    msg += "\n🔍 **DASHBOARD RADAR :**\n"
    for ticker, momentum in radar.items():
        symbol = "✅" if ticker in top_3.index else "⚪"
        msg += f"{symbol} {ticker} : {momentum:+.1f}%\n"
        
    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA : {DCA_MENSUEL}€** | 📊 *Auto-Calcul ATR*\n"
    
    # --- ENVOI TELEGRAM ---
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

if __name__ == "__main__":
    format_and_send()
