import yfinance as yf
import pandas as pd
import requests
import os

# --- 1. CONFIGURATION (Secrets GitHub) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
DCA_MENSUEL = 200

# La Liste des 20 Champions (Radar 360°)
TICKERS = [
    "NVDA", "MSFT", "GOOGL", "AAPL", "TSLA", "SMH",  # Tech & IA
    "BTC-USD", "ETH-USD", "SOL-USD",                # Crypto
    "MC.PA", "RMS.PA", "RACE",                      # Luxe
    "LLY", "UNH",                                   # Santé
    "URNM", "COPX", "XLE",                          # Énergie & Métaux
    "ITA", "NDIA.L", "GLD"                          # Défense, Inde, Or
]
MARKET_INDEX = "SPY"

def get_data():
    fx = yf.Ticker("EURUSD=X")
    usd_to_eur = 1 / fx.history(period="1d")['Close'].iloc[-1]
    
    # Données sur 1 an pour les calculs
    data = yf.download(TICKERS + [MARKET_INDEX], period="1y", auto_adjust=True)['Close'].ffill()
    
    # 1. Régime de Marché (MA200 SPY)
    current_spy = data[MARKET_INDEX].iloc[-1]
    ma200_spy = data[MARKET_INDEX].rolling(window=200).mean().iloc[-1]
    regime = "HAUSSIER (🟢)" if current_spy > ma200_spy else "PRUDENCE (🔴)"
    
    # 2. Indicateurs Individuels
    prices_now = data[TICKERS].iloc[-1]
    ma50 = data[TICKERS].rolling(window=50).mean().iloc[-1]
    momentum = ((prices_now / data[TICKERS].iloc[-126]) - 1) * 100
    
    # 3. Sélection Top 3
    assets_sains = [t for t in TICKERS if prices_now[t] > ma50[t]]
    top_3 = momentum[assets_sains].nlargest(3)
    
    # 4. Volatilité pour Stop Loss
    vol = data[TICKERS].pct_change().rolling(window=14).std() * 100
    
    return regime, top_3, momentum, prices_now, usd_to_eur, vol.iloc[-1], ma50

def format_and_send():
    regime, top_3, radar, prices, fx, vol, ma50 = get_data()
    
    msg = "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "🏆 **ALGO ELITE V5.7 - FINAL 360°**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 **MARCHÉ GLOBAL : {regime}**\n\n"
    
    if "HAUSSIER" in regime:
        msg += "🟢 **SÉLECTION TOP 3 (ACHAT) :**\n"
        if not top_3.empty:
            for t, mom in top_3.items():
                # Gestion devise (EUR pour .PA, sinon USD->EUR)
                p_eur = prices[t] if t.endswith(".PA") else prices[t] * fx
                dist_stop = max(min(vol[t] * 3, 15), 5) 
                msg += f"• **{t}** : {p_eur:.2f}€ (+{mom:.1f}%)\n"
                msg += f"  └ 🛑 Stop : {(p_eur*(1-dist_stop/100)):.2f}€\n"
    else:
        msg += "⚠️ **SIGNAL CASH GUARD** : Restez en liquidités.\n"

    # SECTION SURVEILLANCE SPÉCIALE
    msg += "\n⚡ **FOCUS OPPORTUNITÉS :**\n"
    for watch in ["TSLA", "GOOGL"]:
        m = radar[watch]
        if watch in top_3.index:
            msg += f"✅ **{watch}** est dans le Top 3 !\n"
        elif prices[watch] < ma50[watch]:
            msg += f"❌ **{watch}** est en zone de baisse (Prix < MA50).\n"
        else:
            diff = top_3.iloc[-1] - m
            msg += f"⚪ **{watch}** est saine mais manque {diff:.1f}% de force.\n"

    msg += "\n🔍 **DASHBOARD SECTORIEL (Leader) :**\n"
    sects = {"Tech": ["NVDA", "SMH"], "Crypto": ["BTC-USD", "SOL-USD"], "Luxe": ["MC.PA", "RACE"], "Indus": ["URNM", "COPX"]}
    for s_name, t_list in sects.items():
        leader = radar[t_list].idxmax()
        msg += f"• {s_name} : {leader} (+{radar[leader]:.0f}%)\n"

    msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 **DCA : {DCA_MENSUEL}€** | 📅 *Scan Mensuel actif*"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                      data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    format_and_send()
