"""
APEX v29.0 - Système de Trading Momentum
=========================================

Performance historique:
    - 2015-2025: +13,781% ROI, Ratio G/P 6.10x
    - 2020-2025: +2,098% ROI, Ratio G/P 6.12x

Exécution: python main.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMÈTRES
# ============================================================

MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60
MAX_DRAWDOWN = 0.25

# ============================================================
# CATÉGORIES
# ============================================================

ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"COIN", "MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML"}
TECH = {"APP", "TSLA", "NVDA", "ARM", "PLTR", "SNOW", "DDOG"}

STOP_LOSS = {'ultra': 0.10, 'crypto': 0.10, 'semi': 0.12, 'tech': 0.15, 'other': 0.18}
ATR_THRESHOLD = {'ultra': 0.04, 'crypto': 0.05, 'semi': 0.06, 'tech': 0.06, 'other': 0.04}

# ============================================================
# UNIVERS D'INVESTISSEMENT
# ============================================================

DATABASE = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA", "AVGO",
    "AMD", "MU", "ASML", "TSM", "ARM", "LRCX", "AMAT",
    "PLTR", "APP", "CRWD", "PANW", "NET", "DDOG", "ZS", "SNOW",
    "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER",
    "COIN", "MSTR", "MARA", "RIOT",
    "MC.PA", "RMS.PA", "OR.PA", "SAP", "AIR.PA", "BNP.PA",
    "LLY", "NVO", "UNH", "JNJ", "ABBV", "TMO", "DHR", "ISRG", "PFE",
    "WMT", "COST", "PG", "KO", "PEP",
    "XLE", "XOM", "CVX", "NEE", "XLU",
    "LMT", "RTX",
    "QQQ", "SPY", "GLD", "SLV", "TLT", "ITA"
]

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    """Envoie un message sur Telegram"""
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("⚠️ Secrets TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID non configurés")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Message Telegram envoyé !")
            return True
        else:
            print(f"❌ Erreur Telegram : {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion Telegram : {e}")
        return False

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_category(ticker):
    if ticker in ULTRA_VOLATILE:
        return "ultra"
    elif ticker in CRYPTO:
        return "crypto"
    elif ticker in SEMI:
        return "semi"
    elif ticker in TECH:
        return "tech"
    return "other"

def get_stop_loss(ticker, defensive=False):
    cat = get_category(ticker)
    sl = STOP_LOSS.get(cat, 0.18)
    return sl * 0.80 if defensive else sl

def get_atr_threshold(ticker):
    return ATR_THRESHOLD.get(get_category(ticker), 0.04)

def get_market_regime(vix):
    if pd.isna(vix):
        return "NORMAL", MAX_POSITIONS_NORMAL, False
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "ULTRA_DEFENSIVE", MAX_POSITIONS_ULTRA_DEFENSIVE, True
    elif vix >= VIX_DEFENSIVE:
        return "DEFENSIVE", MAX_POSITIONS_DEFENSIVE, True
    return "NORMAL", MAX_POSITIONS_NORMAL, False

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def calculate_atr_percent(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr / close

def calculate_momentum_score(prices, idx):
    if idx < 126:
        return pd.Series(dtype=float)
    ret_3m = prices.iloc[idx] / prices.iloc[max(0, idx-63)] - 1
    ret_6m = prices.iloc[idx] / prices.iloc[max(0, idx-126)] - 1
    return (ret_3m + ret_6m) / 2

# ============================================================
# FILTRES
# ============================================================

def can_enter(ticker, idx, prices, atr_pct_data, sma_data, high_data):
    """Triple filtre + momentum 1M"""
    if ticker not in prices.columns:
        return True, "OK"
    
    price = prices[ticker].iloc[idx]
    if pd.isna(price):
        return True, "OK"
    
    if ticker in atr_pct_data.columns:
        atr = atr_pct_data[ticker].iloc[idx]
        if not pd.isna(atr) and atr >= get_atr_threshold(ticker):
            return False, "ATR élevé"
    
    if ticker in sma_data.columns:
        sma = sma_data[ticker].iloc[idx]
        if not pd.isna(sma) and price <= sma:
            return False, "< SMA20"
    
    if ticker in high_data.columns:
        high = high_data[ticker].iloc[idx]
        if not pd.isna(high) and high > 0:
            if (price / high - 1) < -MAX_DRAWDOWN:
                return False, "Drawdown"
    
    if idx >= 21:
        ret_1m = prices[ticker].iloc[idx] / prices[ticker].iloc[idx - 21] - 1
        if not pd.isna(ret_1m) and ret_1m < 0:
            return False, "Mom 1M < 0"
    
    return True, "OK"

# ============================================================
# ANALYSE PRINCIPALE
# ============================================================

def analyze():
    """Analyse et envoie les signaux sur Telegram"""
    print("="*60)
    print(f"🔍 APEX v29.0 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    # Téléchargement
    print("\n📡 Téléchargement des données...")
    tickers = DATABASE + ["SPY", "^VIX"]
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    
    if data.empty:
        print("❌ Erreur: Pas de données")
        return
    
    close = data['Close'].ffill().bfill()
    high = data['High'].ffill().bfill()
    low = data['Low'].ffill().bfill()
    
    # VIX et régime
    vix = close["^VIX"].iloc[-1] if "^VIX" in close.columns else 15
    regime, max_pos, defensive = get_market_regime(vix)
    
    print(f"\n🌡️ VIX: {vix:.1f} | Régime: {regime} | Max: {max_pos} positions")
    
    # Préparer les données
    prices = close.drop(columns=["SPY", "^VIX"], errors="ignore")
    high_prices = high.drop(columns=["SPY", "^VIX"], errors="ignore")
    low_prices = low.drop(columns=["SPY", "^VIX"], errors="ignore")
    
    # Calculer les indicateurs
    print("📈 Calcul des indicateurs...")
    atr_pct = pd.DataFrame(index=prices.index)
    sma = pd.DataFrame(index=prices.index)
    high_60 = pd.DataFrame(index=prices.index)
    
    for t in prices.columns:
        if t in high_prices.columns:
            atr_pct[t] = calculate_atr_percent(high_prices[t], low_prices[t], prices[t], ATR_PERIOD)
        sma[t] = prices[t].rolling(SMA_PERIOD).mean()
        high_60[t] = prices[t].rolling(HIGH_LOOKBACK).max()
    
    # Calculer les scores
    idx = len(prices) - 1
    scores = calculate_momentum_score(prices, idx).dropna().sort_values(ascending=False)
    
    # Analyser chaque ticker
    signals = []
    for t in scores.index:
        try:
            price = prices[t].iloc[-1]
            ret_1m = (prices[t].iloc[-1] / prices[t].iloc[-21] - 1) * 100
            ret_3m = (prices[t].iloc[-1] / prices[t].iloc[-63] - 1) * 100
            
            valid, reason = can_enter(t, idx, prices, atr_pct, sma, high_60)
            cat = get_category(t)
            sl = get_stop_loss(t, defensive)
            stop_price = price * (1 - sl)
            
            signals.append({
                'ticker': t,
                'category': cat,
                'score': scores[t],
                'price': price,
                'ret_1m': ret_1m,
                'ret_3m': ret_3m,
                'valid': valid,
                'reason': reason,
                'sl_pct': sl,
                'stop_price': stop_price
            })
        except:
            continue
    
    # Filtrer les valides
    valid_signals = [s for s in signals if s['valid']]
    top_picks = valid_signals[:max_pos]
    
    # Affichage console
    print(f"\n🏆 TOP {max_pos} RECOMMANDÉS:")
    for i, s in enumerate(top_picks, 1):
        print(f"   {i}. {s['ticker']} @ {s['price']:.2f}$ | SL: {s['stop_price']:.2f}$ (-{s['sl_pct']*100:.0f}%)")
    
    # ==================== MESSAGE TELEGRAM ====================
    
    msg = f"🚀 *APEX v29.0* | {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
    
    # Régime VIX
    if defensive:
        msg += f"⚠️ *MODE DÉFENSIF* (VIX: {vix:.1f})\n\n"
    else:
        msg += f"🌡️ VIX: {vix:.1f} | Régime: {regime}\n\n"
    
    msg += f"🔥 *TOP {max_pos} ACTIONS À ACHETER :*\n\n"
    
    for i, s in enumerate(top_picks, 1):
        cat_emoji = {
            'ultra': '⚡', 'crypto': '₿', 'semi': '💾', 
            'tech': '💻', 'other': '📊'
        }.get(s['category'], '📊')
        
        msg += f"*{i}. {s['ticker']}* {cat_emoji}\n"
        msg += f"💰 Prix : {s['price']:.2f}$\n"
        msg += f"📈 Mom 1M : {s['ret_1m']:+.1f}% | 3M : {s['ret_3m']:+.1f}%\n"
        msg += f"🛡️ Stop Loss : {s['stop_price']:.2f}$ (-{s['sl_pct']*100:.0f}%)\n"
        msg += f"📊 Score : {s['score']:.3f}\n\n"
    
    # Résumé
    msg += f"━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 Tickers analysés : {len(signals)}\n"
    msg += f"✅ Tickers valides : {len(valid_signals)}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    msg += "_⚠️ Ceci n'est pas un conseil en investissement._"
    
    # Affichage console
    print("\n" + "="*60)
    print("📱 MESSAGE TELEGRAM:")
    print("="*60)
    print(msg)
    
    # Envoi Telegram
    send_telegram(msg)
    
    return top_picks

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    analyze()
