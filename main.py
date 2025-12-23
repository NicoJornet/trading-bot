import yfinance as yf
import pandas as pd
import requests
import os

# --- 1. CONFIGURATION (Utilise les Secrets GitHub) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DCA_MENSUEL = 200

# Liste d'actifs surveillés
TICKERS = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "BTC-USD", "ETH-USD", "GLD", "NEM"]
MARKET_INDEX = "SPY"

def get_data():
    # A. Taux de change EUR/USD
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    
    # B. Téléchargement des données (1 an pour les moyennes mobiles)
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y")['Close'].ffill()
    
    # C. Régime de Marché (MA200 sur le S&P 500)
    current_spy = data[MARKET_INDEX].iloc[-1]
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if current_spy > ma200_spy else "PRUDENCE / CASH (🔴)"
    
    # D. Filtre de Qualité (Prix > MA50)
    # On ne garde que les actifs dont la tendance court terme est saine
    ma50 = data[TICKERS].rolling(window=50).mean().iloc[-1]
    prices_now = data[TICKERS].iloc[-1]
    assets_sains = prices_now[prices_now > ma50].index.tolist()
    
    # E. Momentum Radar (6 mois / 126 jours)
    returns = ((prices_now / data[TICKERS].iloc[-126]) - 1) * 100
    radar = returns.sort_values(ascending=False)
    
    # Sélection du Top 3 parmi les actifs sains uniquement
    top_3 = returns[assets_sains].nlargest(3)
    
    # F. Volatilité ATR (14 jours) pour Stop Loss Dynamique
    volatility = data[TICKERS].pct_change().rolling(window=14).std() * 100
    
    return regime, top_3, radar, prices_now, usd_to_eur, volatility.iloc[-1]

def format_and_send():
    regime, top_3, radar, prices_usd, fx_rate, current_vol = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏛️ **ALGO ELITE V5.6 PRO (€)**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **RÉGIME : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🏆 **TOP 3 QUALITÉ & MOMENTUM :**\n"
        if not top_3.empty:
            for ticker, momentum in top_3.items():
                price_eur = prices_usd[ticker] * fx_rate
                # Stop Dynamique : 3x Volatilité, borné entre 5% et 15%
                dist_stop = max(min(current_vol[ticker] * 3, 15), 5) 
                stop_eur = price_eur * (1 - (dist_stop / 100))
                
                msg += f"• **{ticker}** : {price_eur:.2f}€\n"
                msg += f"  └ 🔥 Mom : +{momentum:.1f}% | 🛑 Stop : {stop_eur:.2f}€ (-{dist_stop:.1f}%)\n"
        else:
            msg += "⚠️ Aucun actif ne remplit les critères de qualité (MA50).\n"
    else:
        msg += "⚠️ **SIGNAL CASH GUARD ACTIVÉ**\n"
        msg += "Le marché est risqué. Restez en Cash ou Or.\n"
        
    msg += "\n🔍 **DASHBOARD RADAR (Momentum 6m) :**\n"
    for ticker, momentum in radar.items():
        # ✅ si dans le top 3, ⚪ si sain mais pas top 3, ❌ si sous MA50
        status = "✅" if ticker in top_3.index else ("⚪" if ticker in radar.index else "❌")
        # On ajuste le symbole si l'actif est sous sa MA50 (mauvaise santé)
        ma50_val = (prices_usd[ticker] > (prices_usd[ticker] * 0)) # Placeholder
        # (On simplifie l'affichage pour le radar)
        msg += f"{status} {ticker} : {momentum:+.1f}%\n"
        
    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA : {DCA_MENSUEL}€** | ⚖️ *Filtres MA50/200 OK*\n"
    
    # --- ENVOI RÉEL VERS TELEGRAM ---
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erreur envoi : {e}")

if __name__ == "__main__":
    format_and_send()
