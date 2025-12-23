import yfinance as yf
import pandas as pd
import requests
import os

# --- 1. CONFIGURATION (Secrets GitHub) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DCA_MENSUEL = 200

# Les 13 Champions (Elite V5.7 Ultime)
TICKERS = [
    "NVDA", "AAPL", "MSFT", "SMH",   # Tech & Semi-conducteurs
    "BTC-USD", "ETH-USD", "SOL-USD", # Accélérateurs Crypto
    "XLE", "URNM", "COPX",           # Énergie, Uranium, Cuivre
    "ITA", "NDIA.L", "GLD"           # Défense, Inde, Or
]
MARKET_INDEX = "SPY"

def get_data():
    # Taux de change EUR/USD
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    
    # Données historiques (1 an pour MA et Momentum)
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y", auto_adjust=True)['Close'].ffill()
    
    # 1. Régime de Marché (MA200 SPY)
    current_spy = data[MARKET_INDEX].iloc[-1]
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if current_spy > ma200_spy else "PRUDENCE / CASH (🔴)"
    
    # 2. Filtre Individuel (MA50) et Momentum (6 mois)
    prices_now = data[TICKERS].iloc[-1]
    ma50 = data[TICKERS].rolling(window=50).mean().iloc[-1]
    momentum = ((prices_now / data[TICKERS].iloc[-126]) - 1) * 100
    
    # 3. Sélection Top 3 parmi les actifs sains (Prix > MA50)
    assets_sains = [t for t in TICKERS if prices_now[t] > ma50[t]]
    top_3 = momentum[assets_sains].nlargest(3)
    
    # 4. Volatilité (ATR 14j) pour Stop Dynamique
    vol = data[TICKERS].pct_change().rolling(window=14).std() * 100
    
    return regime, top_3, momentum, prices_now, usd_to_eur, vol.iloc[-1], ma50

def format_and_send():
    regime, top_3, radar, prices_usd, fx_rate, current_vol, ma50 = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **ALGO ELITE V5.7 ULTIME**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME GLOBAL : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🟢 **SÉLECTION TOP 3 (ACHAT) :**\n"
        if not top_3.empty:
            for ticker, mom in top_3.items():
                price_eur = prices_usd[ticker] * fx_rate
                # Stop Dynamique : 3x Volatilité (min 5%, max 15%)
                dist_stop = max(min(current_vol[ticker] * 3, 15), 5) 
                stop_eur = price_eur * (1 - (dist_stop / 100))
                msg += f"• **{ticker}** : {price_eur:.2f}€\n"
                msg += f"  └ 🔥 Mom : +{mom:.1f}% | 🛑 Stop : {stop_eur:.2f}€\n"
        else:
            msg += "⚠️ Aucun actif sain (sous sa MA50).\n"
    else:
        msg += "⚠️ **SIGNAL CASH GUARD ACTIVÉ**\n"
        msg += "Le marché est risqué. Conservez vos liquidités.\n"

    msg += "\n🔍 **DASHBOARD DU RADAR :**\n"
    for ticker in TICKERS:
        status = "✅" if ticker in top_3.index else ("⚪" if prices_usd[ticker] > ma50[ticker] else "❌")
        msg += f"{status} {ticker} : {radar[ticker]:+.1f}%\n"

    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA : {DCA_MENSUEL}€** | 🚀 *Elite Mode*"

    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    format_and_send()
