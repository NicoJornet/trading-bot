import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
TOKEN = "TON_TOKEN_TELEGRAM"
CHAT_ID = "TON_CHAT_ID"
CAPITAL_DEPART = 1000
DCA_MENSUEL = 200
# Liste d'actifs surveillés par l'Algo
TICKERS = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "BTC-USD", "ETH-USD", "GLD", "NEM", "AMD", "NFLX"]
MARKET_INDEX = "SPY"  # Indice pour le régime de marché (Cash Guard)

def get_data():
    # 1. Récupérer le taux de change USD/EUR
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    
    # 2. Télécharger les prix (6 mois d'historique pour le momentum)
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y")['Close'].ffill()
    
    # 3. Calcul du Régime de Marché (MA200)
    current_spy = data[MARKET_INDEX].iloc[-1]
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if current_spy > ma200_spy else "PRUDENCE / CASH (🔴)"
    
    # 4. Calcul du Momentum (Performance 6 mois)
    returns = (data[TICKERS].iloc[-1] / data[TICKERS].iloc[-126]) - 1
    top_3 = returns.nlargest(3)
    
    # 5. Calcul des Stop Loss (environ 5% sous le prix actuel)
    prices_usd = data[TICKERS].iloc[-1]
    
    return regime, top_3, prices_usd, usd_to_eur

def format_message():
    regime, top_3, prices_usd, fx_rate = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **BOT ALGO ELITE V5.2 (€)**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🏆 **TOP 3 MOMENTUM (€) :**\n"
        for ticker, perf in top_3.items():
            price_eur = prices_usd[ticker] * fx_rate
            stop_eur = price_eur * 0.95  # Stop Loss à -5%
            msg += f"• **{ticker}** : {price_eur:.2f}€\n"
            msg += f"  └ 🛑 Stop Loss : {stop_eur:.2f}€\n"
    else:
        msg += "⚠️ **SIGNAL CASH GUARD ACTIVÉ**\n"
        msg += "Vendre les positions et rester en liquide.\n"
        
    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA À INJECTER : {DCA_MENSUEL}€**\n"
    msg += "📊 *Conversion réalisée au taux du jour.*\n"
    
    return msg

# Pour envoyer le message (nécessite la lib requests)
# import requests
# requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={format_message()}&parse_mode=Markdown")

print(format_message())
