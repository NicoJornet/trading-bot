"""
APEX v30.0 HYBRIDE - VERSION PRODUCTION
========================================

Stratégie optimisée combinant:
- Dual Momentum (VIX < 25): Exclut les actions en baisse
- Momentum Simple (VIX >= 25): Prend les "moins pires"

Performance backtest 2020-2025:
- ROI: +1769%
- Ratio G/P: 4.62x
- Win Rate: 57.7%
- Rotations: ~1.3/mois

Alertes Telegram incluses.
Compatible GitHub Actions.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Telegram (à configurer via variables d'environnement)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Devise
DISPLAY_CURRENCY = "EUR"  # EUR ou USD

def get_eur_usd_rate():
    """Récupère le taux de change EUR/USD"""
    try:
        ticker = yf.Ticker("EURUSD=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    return 1.08  # Taux par défaut si erreur

# Taux de change global (mis à jour au démarrage)
EUR_USD_RATE = None

def usd_to_eur(usd_amount):
    """Convertit USD en EUR"""
    global EUR_USD_RATE
    if EUR_USD_RATE is None:
        EUR_USD_RATE = get_eur_usd_rate()
    return usd_amount / EUR_USD_RATE

def format_price(price_usd, show_both=False):
    """Formate un prix en EUR (avec option USD)"""
    price_eur = usd_to_eur(price_usd)
    if show_both:
        return f"{price_eur:.2f}€ (${price_usd:.2f})"
    return f"{price_eur:.2f}€"

# Paramètres de la stratégie
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60
MAX_DRAWDOWN = 0.25

TRAILING_STOP_ACTIVATION = 1.40
TRAILING_STOP_PCT = 0.20
MINI_TRAIL_ACTIVATION = 1.25
MINI_TRAIL_PCT = 0.15

# ============================================================
# UNIVERS D'ACTIONS US
# ============================================================

DATABASE = [
    # Tech Giants
    "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA", "AVGO",
    # Semiconducteurs
    "AMD", "MU", "ASML", "TSM", "ARM", "LRCX", "AMAT",
    # Software & Cloud
    "PLTR", "APP", "CRWD", "PANW", "NET", "DDOG", "ZS", "SNOW",
    # Growth
    "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER",
    # Crypto-related
    "COIN", "MSTR", "MARA", "RIOT",
    # Healthcare
    "LLY", "NVO", "UNH", "JNJ", "ABBV",
    # Consumer
    "WMT", "COST", "PG", "KO",
    # Energy
    "XOM", "CVX",
    # ETFs
    "QQQ", "SPY", "GLD", "SLV",
]

# Catégories pour stops adaptatifs
ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"COIN", "MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML"}
TECH = {"APP", "TSLA", "NVDA", "ARM", "PLTR", "SNOW", "DDOG"}

# Stop Loss par catégorie
STOP_LOSS = {
    'ultra': 0.10,
    'crypto': 0.10,
    'semi': 0.12,
    'tech': 0.15,
    'other': 0.18
}

# Seuils ATR par catégorie
ATR_THRESHOLD = {
    'ultra': 0.04,
    'crypto': 0.05,
    'semi': 0.06,
    'tech': 0.06,
    'other': 0.04
}

# Seuils de rotation par catégorie
ROTATION_THRESHOLD = {
    'ultra': 0.05,
    'crypto': 0.07,
    'semi': 0.06,
    'tech': 0.07,
    'other': 0.08
}

# Jours minimum de détention par catégorie
MIN_HOLDING_DAYS = {
    'ultra': 7,
    'crypto': 7,
    'semi': 14,
    'tech': 14,
    'other': 21
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_category(ticker):
    """Retourne la catégorie d'un ticker"""
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
    """Retourne le stop loss adapté au ticker"""
    sl = STOP_LOSS.get(get_category(ticker), 0.18)
    return sl * 0.80 if defensive else sl

def get_rotation_threshold(ticker):
    """Retourne le seuil de rotation pour un ticker"""
    return ROTATION_THRESHOLD.get(get_category(ticker), 0.08)

def get_min_holding_days(ticker):
    """Retourne le nombre minimum de jours de détention"""
    return MIN_HOLDING_DAYS.get(get_category(ticker), 21)

def get_atr_threshold(ticker):
    """Retourne le seuil ATR pour un ticker"""
    return ATR_THRESHOLD.get(get_category(ticker), 0.04)

def get_market_regime(vix):
    """Détermine le régime de marché selon le VIX"""
    if pd.isna(vix):
        return "NORMAL", MAX_POSITIONS_NORMAL, False
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "ULTRA_DEFENSIVE", MAX_POSITIONS_ULTRA_DEFENSIVE, True
    elif vix >= VIX_DEFENSIVE:
        return "DEFENSIVE", MAX_POSITIONS_DEFENSIVE, True
    return "NORMAL", MAX_POSITIONS_NORMAL, False

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message, parse_mode="HTML"):
    """Envoie un message Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[TELEGRAM DISABLED] {message}")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

def format_currency(value_usd):
    """Formate un montant en euros"""
    value_eur = usd_to_eur(value_usd)
    return f"{value_eur:,.2f}€".replace(",", " ").replace(".", ",")

def format_percent(value):
    """Formate un pourcentage"""
    return f"{value:+.2f}%"

def calculate_stop_loss_price(entry_price_usd, ticker, defensive=False):
    """Calcule le prix du stop loss en USD et EUR"""
    sl_pct = get_stop_loss(ticker, defensive)
    sl_price_usd = entry_price_usd * (1 - sl_pct)
    sl_price_eur = usd_to_eur(sl_price_usd)
    return sl_price_usd, sl_price_eur, sl_pct

# ============================================================
# MÉTHODES DE SCORING HYBRIDE
# ============================================================

def score_momentum_simple(prices, idx):
    """Momentum Simple (Original APEX v29) - moyenne 3M et 6M"""
    if idx < 126:
        return pd.Series(dtype=float)
    ret_3m = prices.iloc[idx] / prices.iloc[idx - 63] - 1
    ret_6m = prices.iloc[idx] / prices.iloc[idx - 126] - 1
    return (ret_3m + ret_6m) / 2

def score_dual_momentum(prices, idx):
    """Dual Momentum - seulement les actions avec momentum positif"""
    if idx < 63:
        return pd.Series(dtype=float)
    
    scores = {}
    for ticker in prices.columns:
        try:
            p = prices[ticker]
            mom = float(p.iloc[idx]) / float(p.iloc[idx - 63]) - 1
            # CLEF: Seulement si momentum positif
            if mom > 0:
                scores[ticker] = mom
        except:
            pass
    return pd.Series(scores)

def score_hybride(prices, idx, vix_value):
    """
    HYBRIDE: Combine Dual et Simple selon le VIX
    - VIX < 25: Dual Momentum (exclut les perdants)
    - VIX >= 25: Momentum Simple (prend les moins pires)
    """
    if vix_value < VIX_DEFENSIVE:
        return score_dual_momentum(prices, idx)
    else:
        return score_momentum_simple(prices, idx)

# ============================================================
# ANALYSE ET SIGNAUX
# ============================================================

def download_data(lookback_days=200):
    """Télécharge les données de marché"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    tickers = DATABASE + ["^VIX"]
    
    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )
    
    return data

def calculate_indicators(data):
    """Calcule tous les indicateurs techniques"""
    close = data['Close'].ffill().bfill()
    high = data['High'].ffill().bfill()
    low = data['Low'].ffill().bfill()
    
    # VIX
    vix = close["^VIX"] if "^VIX" in close.columns else pd.Series(index=close.index, data=15)
    prices = close.drop(columns=["^VIX"], errors="ignore")
    high_prices = high.drop(columns=["^VIX"], errors="ignore")
    low_prices = low.drop(columns=["^VIX"], errors="ignore")
    
    # ATR%
    atr_pct = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        if ticker in high_prices.columns and ticker in low_prices.columns:
            tr1 = high_prices[ticker] - low_prices[ticker]
            tr2 = abs(high_prices[ticker] - prices[ticker].shift(1))
            tr3 = abs(low_prices[ticker] - prices[ticker].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_pct[ticker] = tr.rolling(ATR_PERIOD).mean() / prices[ticker]
    
    # SMA
    sma = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        sma[ticker] = prices[ticker].rolling(SMA_PERIOD).mean()
    
    # High 60 jours
    high_60 = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        high_60[ticker] = prices[ticker].rolling(HIGH_LOOKBACK).max()
    
    return {
        'prices': prices,
        'vix': vix,
        'atr_pct': atr_pct,
        'sma': sma,
        'high_60': high_60
    }

def check_entry_conditions(ticker, price, indicators, idx):
    """Vérifie les conditions d'entrée pour un ticker"""
    # ATR filter
    if ticker in indicators['atr_pct'].columns:
        atr = indicators['atr_pct'][ticker].iloc[idx]
        if not pd.isna(atr) and atr >= get_atr_threshold(ticker):
            return False, f"ATR trop élevé ({atr:.2%})"
    
    # SMA filter
    if ticker in indicators['sma'].columns:
        sma = indicators['sma'][ticker].iloc[idx]
        if not pd.isna(sma) and price <= float(sma):
            return False, f"Prix sous SMA20 ({price:.2f} < {sma:.2f})"
    
    # Drawdown filter
    if ticker in indicators['high_60'].columns:
        high = indicators['high_60'][ticker].iloc[idx]
        if not pd.isna(high) and float(high) > 0:
            dd = (price / float(high) - 1)
            if dd < -MAX_DRAWDOWN:
                return False, f"Drawdown excessif ({dd:.1%})"
    
    # Momentum 1M filter
    if idx >= 21:
        ret_1m = float(indicators['prices'][ticker].iloc[idx]) / float(indicators['prices'][ticker].iloc[idx - 21]) - 1
        if ret_1m < 0:
            return False, f"Momentum 1M négatif ({ret_1m:.1%})"
    
    return True, "OK"

def generate_signals(portfolio=None, entry_dates=None):
    """
    Génère les signaux de trading
    
    Args:
        portfolio: dict des positions actuelles {ticker: {'entry': prix, 'shares': nb}}
        entry_dates: dict des dates d'entrée {ticker: date}
    
    Returns:
        dict avec les signaux et recommandations
    """
    if portfolio is None:
        portfolio = {}
    if entry_dates is None:
        entry_dates = {}
    
    # Téléchargement des données
    data = download_data(200)
    if data.empty:
        return {"error": "Impossible de télécharger les données"}
    
    indicators = calculate_indicators(data)
    prices = indicators['prices']
    vix = indicators['vix']
    
    idx = len(prices) - 1
    current_prices = prices.iloc[idx]
    current_vix = float(vix.iloc[idx]) if not pd.isna(vix.iloc[idx]) else 15
    current_date = prices.index[idx]
    
    # Régime de marché
    regime, max_positions, defensive = get_market_regime(current_vix)
    
    # Calculer les scores HYBRIDES
    scores = score_hybride(prices, idx, current_vix)
    scores = scores.dropna().sort_values(ascending=False)
    
    # Filtrer les candidats valides
    valid_candidates = []
    for ticker in scores.index:
        if ticker not in current_prices.index:
            continue
        price = current_prices[ticker]
        if pd.isna(price):
            continue
        
        can_enter, reason = check_entry_conditions(ticker, float(price), indicators, idx)
        if can_enter:
            valid_candidates.append({
                'ticker': ticker,
                'price': float(price),
                'score': scores[ticker],
                'category': get_category(ticker)
            })
    
    # Top N selon le régime
    top_candidates = valid_candidates[:max_positions]
    
    # Analyser le portfolio actuel
    portfolio_analysis = []
    signals = []
    
    for ticker, position in portfolio.items():
        if ticker not in current_prices.index:
            continue
        
        current_price = float(current_prices[ticker])
        entry_price = position['entry']
        shares = position['shares']
        
        gain_pct = (current_price / entry_price - 1) * 100
        gain_eur = shares * (current_price - entry_price)
        
        # Vérifier les trailing stops
        peak_price = position.get('peak', entry_price)
        peak_price = max(peak_price, current_price)
        
        action = "HOLD"
        reason = ""
        
        # Trailing Stop 40%
        if current_price >= entry_price * TRAILING_STOP_ACTIVATION:
            if current_price < peak_price * (1 - TRAILING_STOP_PCT):
                action = "SELL"
                reason = f"🎯 TRAILING STOP 40% (peak: {peak_price:.2f})"
        
        # Mini Trail 25%
        elif current_price >= entry_price * MINI_TRAIL_ACTIVATION:
            if current_price < peak_price * (1 - MINI_TRAIL_PCT):
                action = "SELL"
                reason = f"🎯 MINI-TRAIL 25% (peak: {peak_price:.2f})"
        
        # Stop Loss
        sl_pct = get_stop_loss(ticker, defensive)
        if current_price < entry_price * (1 - sl_pct):
            action = "SELL"
            reason = f"🔴 STOP LOSS (-{sl_pct:.0%})"
        
        # Vérifier rotation possible
        if action == "HOLD" and ticker in entry_dates:
            days_held = (current_date - entry_dates[ticker]).days
            min_days = get_min_holding_days(ticker)
            
            if days_held >= min_days:
                # Chercher un meilleur candidat
                current_score = scores.get(ticker, 0)
                for candidate in top_candidates:
                    if candidate['ticker'] not in portfolio:
                        rot_threshold = get_rotation_threshold(ticker)
                        if candidate['score'] > current_score * (1 + rot_threshold):
                            action = "ROTATE"
                            reason = f"🔄 ROTATION vers {candidate['ticker']} (score: {candidate['score']:.3f})"
                            break
        
        portfolio_analysis.append({
            'ticker': ticker,
            'entry': entry_price,
            'current': current_price,
            'shares': shares,
            'gain_pct': gain_pct,
            'gain_eur': gain_eur,
            'action': action,
            'reason': reason,
            'peak': peak_price,
            'category': get_category(ticker)
        })
        
        if action != "HOLD":
            signals.append({
                'type': action,
                'ticker': ticker,
                'price': current_price,
                'reason': reason
            })
    
    # Signaux d'achat pour nouvelles positions
    open_slots = max_positions - len([p for p in portfolio_analysis if p['action'] == 'HOLD'])
    
    for candidate in top_candidates[:open_slots]:
        if candidate['ticker'] not in portfolio:
            signals.append({
                'type': 'BUY',
                'ticker': candidate['ticker'],
                'price': candidate['price'],
                'score': candidate['score'],
                'reason': f"🟢 TOP {len(signals)+1} - Score: {candidate['score']:.3f}"
            })
    
    return {
        'date': current_date.strftime('%Y-%m-%d'),
        'vix': current_vix,
        'regime': regime,
        'max_positions': max_positions,
        'defensive': defensive,
        'scoring_method': 'DUAL' if current_vix < VIX_DEFENSIVE else 'SIMPLE',
        'portfolio_analysis': portfolio_analysis,
        'signals': signals,
        'top_candidates': top_candidates[:10],
        'valid_candidates_count': len(valid_candidates)
    }

def format_daily_report(result):
    """Formate le rapport quotidien pour Telegram"""
    
    # Emoji selon le régime
    regime_emoji = "🟢" if result['regime'] == "NORMAL" else "🟡" if result['regime'] == "DEFENSIVE" else "🔴"
    method_emoji = "🎯" if result['scoring_method'] == "DUAL" else "📊"
    
    # Taux de change
    global EUR_USD_RATE
    if EUR_USD_RATE is None:
        EUR_USD_RATE = get_eur_usd_rate()
    
    msg = f"""
<b>📊 APEX v30.0 HYBRIDE - {result['date']}</b>

{regime_emoji} <b>Régime:</b> {result['regime']}
📈 <b>VIX:</b> {result['vix']:.1f}
{method_emoji} <b>Méthode:</b> {result['scoring_method']} Momentum
🎰 <b>Positions max:</b> {result['max_positions']}
💱 <b>EUR/USD:</b> {EUR_USD_RATE:.4f}
"""
    
    # Portfolio actuel
    if result['portfolio_analysis']:
        msg += "\n<b>📂 PORTFOLIO:</b>\n"
        total_gain_eur = 0
        for p in result['portfolio_analysis']:
            emoji = "📈" if p['gain_pct'] >= 0 else "📉"
            action_emoji = "✅" if p['action'] == "HOLD" else "⚠️"
            
            # Convertir en EUR
            current_eur = usd_to_eur(p['current'])
            entry_eur = usd_to_eur(p['entry'])
            gain_eur = usd_to_eur(p['gain_eur'])
            
            # Calculer stop loss
            sl_usd, sl_eur, sl_pct = calculate_stop_loss_price(p['entry'], p['ticker'], result['defensive'])
            
            msg += f"{action_emoji} <b>{p['ticker']}</b>: {current_eur:.2f}€ ({p['gain_pct']:+.1f}%) {emoji}\n"
            msg += f"    └ Entrée: {entry_eur:.2f}€ | Stop: {sl_eur:.2f}€ (-{sl_pct:.0%})\n"
            
            total_gain_eur += gain_eur
        
        msg += f"\n<b>💰 Total P&L:</b> {total_gain_eur:+,.0f}€\n"
    
    # Signaux
    if result['signals']:
        msg += "\n<b>🚨 SIGNAUX:</b>\n"
        for s in result['signals']:
            price_eur = usd_to_eur(s['price'])
            
            if s['type'] == 'BUY':
                # Calculer le stop loss pour les achats
                sl_usd, sl_eur, sl_pct = calculate_stop_loss_price(s['price'], s['ticker'], result['defensive'])
                msg += f"🟢 <b>ACHETER {s['ticker']}</b>\n"
                msg += f"    └ Prix: {price_eur:.2f}€\n"
                msg += f"    └ Stop Loss: {sl_eur:.2f}€ (-{sl_pct:.0%})\n"
                msg += f"    └ Score: {s.get('score', 0):.3f}\n"
            elif s['type'] == 'SELL':
                msg += f"🔴 <b>VENDRE {s['ticker']}</b> @ {price_eur:.2f}€\n"
                msg += f"    └ {s['reason']}\n"
            elif s['type'] == 'ROTATE':
                msg += f"🔄 <b>ROTATION {s['ticker']}</b> @ {price_eur:.2f}€\n"
                msg += f"    └ {s['reason']}\n"
    else:
        msg += "\n✅ <b>Aucun signal</b> - Positions maintenues\n"
    
    # Top candidats avec prix EUR et stop loss
    msg += "\n<b>🏆 TOP 5 CANDIDATS:</b>\n"
    for i, c in enumerate(result['top_candidates'][:5], 1):
        price_eur = usd_to_eur(c['price'])
        sl_usd, sl_eur, sl_pct = calculate_stop_loss_price(c['price'], c['ticker'], result['defensive'])
        msg += f"{i}. <b>{c['ticker']}</b> @ {price_eur:.2f}€ (SL: {sl_eur:.2f}€)\n"
    
    return msg

# ============================================================
# MAIN
# ============================================================

def main():
    """Point d'entrée principal"""
    print("="*60)
    print("🚀 APEX v30.0 HYBRIDE - Analyse quotidienne")
    print("="*60)
    
    # Initialiser le taux de change
    global EUR_USD_RATE
    EUR_USD_RATE = get_eur_usd_rate()
    print(f"💱 Taux EUR/USD: {EUR_USD_RATE:.4f}")
    
    # Exemple de portfolio (à adapter selon votre situation)
    # En production, charger depuis un fichier JSON
    portfolio = {
        # "NVDA": {"entry": 120.0, "shares": 10, "peak": 140.0},
        # "PLTR": {"entry": 25.0, "shares": 50, "peak": 35.0},
    }
    
    entry_dates = {
        # "NVDA": datetime(2024, 6, 1),
        # "PLTR": datetime(2024, 8, 15),
    }
    
    # Générer les signaux
    result = generate_signals(portfolio, entry_dates)
    
    if "error" in result:
        print(f"❌ Erreur: {result['error']}")
        return
    
    # Afficher le rapport
    print(f"\n📅 Date: {result['date']}")
    print(f"📊 VIX: {result['vix']:.1f}")
    print(f"🎯 Régime: {result['regime']}")
    print(f"📈 Méthode: {result['scoring_method']} Momentum")
    print(f"🎰 Positions max: {result['max_positions']}")
    
    print(f"\n🏆 TOP 10 CANDIDATS:")
    print("-"*70)
    print(f"{'#':<3} {'Ticker':<8} {'Prix USD':<12} {'Prix EUR':<12} {'Stop EUR':<12} {'Score':<10} {'Cat.'}")
    print("-"*70)
    
    for i, c in enumerate(result['top_candidates'], 1):
        price_eur = usd_to_eur(c['price'])
        sl_usd, sl_eur, sl_pct = calculate_stop_loss_price(c['price'], c['ticker'], result['defensive'])
        print(f"{i:<3} {c['ticker']:<8} ${c['price']:<11.2f} {price_eur:<11.2f}€ {sl_eur:<11.2f}€ {c['score']:<10.4f} [{c['category']}]")
    
    if result['signals']:
        print(f"\n🚨 SIGNAUX ({len(result['signals'])}):")
        print("-"*70)
        for s in result['signals']:
            price_eur = usd_to_eur(s['price'])
            if s['type'] == 'BUY':
                sl_usd, sl_eur, sl_pct = calculate_stop_loss_price(s['price'], s['ticker'], result['defensive'])
                print(f"   🟢 ACHETER: {s['ticker']}")
                print(f"      Prix:     ${s['price']:.2f} = {price_eur:.2f}€")
                print(f"      Stop:     ${sl_usd:.2f} = {sl_eur:.2f}€ (-{sl_pct:.0%})")
                print(f"      Score:    {s.get('score', 0):.4f}")
            else:
                print(f"   {s['type']}: {s['ticker']} @ {price_eur:.2f}€")
                print(f"      {s['reason']}")
    else:
        print("\n✅ Aucun signal - Positions maintenues")
    
    # Envoyer le rapport Telegram
    telegram_msg = format_daily_report(result)
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        success = send_telegram(telegram_msg)
        print(f"\n📱 Telegram: {'✅ Envoyé' if success else '❌ Erreur'}")
    else:
        print("\n📱 Telegram: Désactivé (configurer TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID)")
    
    print("\n" + "="*60)
    
    return result

if __name__ == "__main__":
    main()
