import yfinance as yf
import pandas as pd
import requests
import os

# --- 1. CONFIGURATION (Utilise les Secrets GitHub pour la sécurité) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DCA_MENSUEL = 200

# Liste d'actifs surveillés
TICKERS = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "BTC-USD", "GLD", "NEM"]
MARKET_INDEX = "SPY"

def get_data():
    # A. Récupérer le taux de change USD/EUR
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    
    # B. Télécharger les prix (1 an pour avoir MA200 et Momentum)
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y")['Close'].ffill()
    
    # C. Calcul du Régime de Marché (MA200)
    current_spy = data[MARKET_INDEX].iloc[-1]
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if current_spy > ma200_spy else "PRUDENCE / CASH (🔴)"
    
    # D. Calcul du Momentum (Top 3 sur 6 mois / 126 jours de trading)
    returns = (data[TICKERS].iloc[-1] / data[TICKERS].iloc[-126]) - 1
    top_3 = returns.nlargest(3)
    
    # E. Prix actuels
    prices_usd = data[TICKERS].iloc[-1]
    
    return regime, top_3, prices_usd, usd_to_eur

def format_and_send():
    regime, top_3, prices_usd, fx_rate = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **BOT ALGO ELITE V5.2 (€)**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🏆 **TOP 3 MOMENTUM (€) :**\n"
        for ticker, perf in top_3.items():
            price_eur = prices_usd[ticker] * fx_rate
            stop_eur = price_eur * 0.95  # Sécurité à -5%
            msg += f"• **{ticker}** : {price_eur:.2f}€\n"
            msg += f"  └ 🛑 Stop Loss : {stop_eur:.2f}€\n"
    else:
        msg += "⚠️ **SIGNAL CASH GUARD ACTIVÉ**\n"
        msg += "Le marché est risqué. Vendre et rester en Cash.\n"
        
    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA À INJECTER : {DCA_MENSUEL}€**\n"
    msg += "📊 *Signal généré automatiquement.*\n"
    
    # --- ENVOI RÉEL VERS TELEGRAM ---
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CH_ID, "text": msg, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("✅ Message envoyé avec succès à Telegram !")
        else:
            print(f"❌ Erreur lors de l'envoi : {response.text}")
    except Exception as e:
        print(f"⚠️ Erreur de connexion : {e}")

# Lancement du script
if __name__ == "__main__":
    format_and_send()
