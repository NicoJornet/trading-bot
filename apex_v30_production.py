"""
APEX v30.5 HYBRIDE - PRODUCTION
================================

Améliorations vs v30.0:
1. Allocation pondérée : 50% rang #1, 30% rang #2, 20% rang #3
2. Pyramiding : +50% sur position si gain >= +15% ET nouveau high 20j

Capital: 1,500€ initial + 100€/mois DCA
Tracking: portfolio.json + trades_history.json

Performance backtestée (2020-2025):
- v30.0 : +1710% ROI
- v30.5 : +1829% ROI (+118.6%)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests

# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 1500
MONTHLY_DCA = 100
COST_PER_TRADE = 1.0

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Paramètres APEX
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

# v30.5 - Paramètres pyramiding
PYRAMID_GAIN_THRESHOLD = 0.15  # +15% minimum pour pyramider
PYRAMID_LOOKBACK = 20          # New high sur 20 jours
PYRAMID_ADD_PCT = 0.50         # Ajouter 50% de la position initiale

# Univers
DATABASE = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA", "AVGO",
    "AMD", "MU", "ASML", "TSM", "ARM", "LRCX", "AMAT",
    "PLTR", "APP", "CRWD", "PANW", "NET", "DDOG", "ZS", "SNOW",
    "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER",
    "COIN", "MSTR", "MARA", "RIOT",
    "LLY", "NVO", "UNH", "JNJ", "ABBV",
    "WMT", "COST", "PG", "KO",
    "XOM", "CVX",
    "QQQ", "SPY", "GLD", "SLV",
]

ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"COIN", "MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML"}
TECH = {"APP", "TSLA", "NVDA", "ARM", "PLTR", "SNOW", "DDOG"}

STOP_LOSS = {'ultra': 0.10, 'crypto': 0.10, 'semi': 0.12, 'tech': 0.15, 'other': 0.18}
ATR_THRESHOLD = {'ultra': 0.04, 'crypto': 0.05, 'semi': 0.06, 'tech': 0.06, 'other': 0.04}
ROTATION_THRESHOLD = {'ultra': 0.05, 'crypto': 0.07, 'semi': 0.06, 'tech': 0.07, 'other': 0.08}
MIN_HOLDING_DAYS = {'ultra': 7, 'crypto': 7, 'semi': 14, 'tech': 14, 'other': 21}

def get_category(ticker):
    if ticker in ULTRA_VOLATILE: return "ultra"
    elif ticker in CRYPTO: return "crypto"
    elif ticker in SEMI: return "semi"
    elif ticker in TECH: return "tech"
    return "other"

def get_stop_loss_pct(ticker, defensive=False):
    sl = STOP_LOSS.get(get_category(ticker), 0.18)
    return sl * 0.80 if defensive else sl

def get_min_holding_days(ticker):
    return MIN_HOLDING_DAYS.get(get_category(ticker), 21)

def get_rotation_threshold(ticker):
    return ROTATION_THRESHOLD.get(get_category(ticker), 0.08)

def get_atr_threshold(ticker):
    return ATR_THRESHOLD.get(get_category(ticker), 0.04)

# ============================================================
# v30.5 - ALLOCATION PONDÉRÉE
# ============================================================

def get_weighted_allocation(rank, num_positions, total_capital):
    """
    Allocation pondérée par rang.
    Rang 1 = 50%, Rang 2 = 30%, Rang 3 = 20%
    """
    if num_positions == 1:
        return total_capital
    elif num_positions == 2:
        weights = {1: 0.60, 2: 0.40}
    elif num_positions == 3:
        weights = {1: 0.50, 2: 0.30, 3: 0.20}
    else:
        total_weight = sum(range(1, num_positions + 1))
        weights = {i: (num_positions - i + 1) / total_weight for i in range(1, num_positions + 1)}
    
    return total_capital * weights.get(rank, 1.0 / num_positions)

# ============================================================
# v30.5 - PYRAMIDING
# ============================================================

def check_pyramid_signal(pos, current_price_eur, high_prices, ticker, idx):
    """
    Vérifie si on doit pyramider une position.
    Conditions:
    - Gain >= +15%
    - Prix fait un nouveau high 20 jours
    - Position pas déjà pyramidée
    
    Retourne: (should_pyramid, add_amount)
    """
    entry_price_eur = pos["entry_price_eur"]
    gain_pct = (current_price_eur / entry_price_eur) - 1
    
    # Déjà pyramidé ?
    if pos.get("pyramided", False):
        return False, 0
    
    # En profit suffisant ?
    if gain_pct < PYRAMID_GAIN_THRESHOLD:
        return False, 0
    
    # Nouveau high 20j ?
    if idx < PYRAMID_LOOKBACK:
        return False, 0
    
    try:
        if ticker in high_prices.columns:
            recent_high = float(high_prices[ticker].iloc[idx-PYRAMID_LOOKBACK:idx].max())
            current_price_usd = float(high_prices[ticker].iloc[idx])
            
            if current_price_usd > recent_high:
                # Signal de pyramide !
                initial_amount = pos.get("initial_amount_eur", pos.get("amount_invested_eur", 500))
                add_amount = initial_amount * PYRAMID_ADD_PCT
                return True, add_amount
    except:
        pass
    
    return False, 0

# ============================================================
# CONVERSION EUR/USD
# ============================================================

def get_eur_usd_rate():
    try:
        eur_usd = yf.Ticker("EURUSD=X")
        rate = eur_usd.info.get('regularMarketPrice') or eur_usd.info.get('previousClose')
        if rate and rate > 0:
            return rate
    except:
        pass
    return 1.08

def usd_to_eur(amount_usd, rate=None):
    if rate is None:
        rate = get_eur_usd_rate()
    return amount_usd / rate

def eur_to_usd(amount_eur, rate=None):
    if rate is None:
        rate = get_eur_usd_rate()
    return amount_eur * rate

# ============================================================
# GESTION PORTFOLIO
# ============================================================

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": INITIAL_CAPITAL,
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {},
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

def load_trades_history():
    if os.path.exists(TRADES_HISTORY_FILE):
        try:
            with open(TRADES_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "trades": [],
        "summary": {
            "total_trades": 0,
            "buys": 0,
            "sells": 0,
            "pyramids": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl_eur": 0,
            "total_fees_eur": 0,
            "best_trade_eur": 0,
            "worst_trade_eur": 0,
            "win_rate": 0
        }
    }

def save_trades_history(history):
    trades = history["trades"]
    sells = [t for t in trades if t["action"] == "SELL"]
    pyramids = [t for t in trades if t["action"] == "PYRAMID"]
    
    winning = [t for t in sells if t.get("pnl_eur", 0) > 0]
    losing = [t for t in sells if t.get("pnl_eur", 0) < 0]
    
    history["summary"] = {
        "total_trades": len(trades),
        "buys": len([t for t in trades if t["action"] == "BUY"]),
        "sells": len(sells),
        "pyramids": len(pyramids),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "total_pnl_eur": sum(t.get("pnl_eur", 0) for t in sells),
        "total_fees_eur": sum(t.get("fee_eur", 0) for t in trades),
        "best_trade_eur": max([t.get("pnl_eur", 0) for t in sells], default=0),
        "worst_trade_eur": min([t.get("pnl_eur", 0) for t in sells], default=0),
        "win_rate": len(winning) / len(sells) * 100 if sells else 0
    }
    
    with open(TRADES_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def log_trade(history, action, ticker, price_usd, price_eur, shares, amount_eur, eur_rate, reason="signal", pnl_eur=None, pnl_pct=None):
    fee_eur = usd_to_eur(COST_PER_TRADE, eur_rate)
    
    trade = {
        "id": len(history["trades"]) + 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "ticker": ticker,
        "shares": round(shares, 4),
        "price_usd": round(price_usd, 2),
        "price_eur": round(price_eur, 2),
        "amount_eur": round(amount_eur, 2),
        "fee_eur": round(fee_eur, 2),
        "eur_usd_rate": round(eur_rate, 4),
        "reason": reason
    }
    
    if action == "SELL" and pnl_eur is not None:
        trade["pnl_eur"] = round(pnl_eur, 2)
        trade["pnl_pct"] = round(pnl_pct, 2)
    
    history["trades"].append(trade)
    return trade

# ============================================================
# SCORING
# ============================================================

def score_dual_momentum(prices, idx):
    if idx < 63:
        return pd.Series(dtype=float)
    scores = {}
    for ticker in prices.columns:
        try:
            mom = float(prices[ticker].iloc[idx]) / float(prices[ticker].iloc[idx - 63]) - 1
            if mom > 0:
                scores[ticker] = mom
        except:
            pass
    return pd.Series(scores)

def score_momentum_simple(prices, idx):
    if idx < 126:
        return pd.Series(dtype=float)
    ret_3m = prices.iloc[idx] / prices.iloc[idx - 63] - 1
    ret_6m = prices.iloc[idx] / prices.iloc[idx - 126] - 1
    return (ret_3m + ret_6m) / 2

def score_hybride(prices, idx, vix_value):
    if vix_value < VIX_DEFENSIVE:
        return score_dual_momentum(prices, idx)
    else:
        return score_momentum_simple(prices, idx)

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configuré")
        print(message)
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur Telegram: {e}")
        return False

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"{'='*70}")
    print(f"📊 APEX v30.5 HYBRIDE (Weighted + Pyramid) - {today}")
    print(f"{'='*70}")
    
    portfolio = load_portfolio()
    history = load_trades_history()
    eur_rate = get_eur_usd_rate()
    
    print(f"💱 EUR/USD: {eur_rate:.4f}")
    print(f"💰 Cash disponible: {portfolio['cash']:.2f}€")
    
    # DCA mensuel
    last_dca = portfolio.get("last_dca_date")
    current_month = datetime.now().strftime("%Y-%m")
    
    if last_dca is None or not last_dca.startswith(current_month):
        if last_dca is not None:
            portfolio["cash"] += MONTHLY_DCA
            print(f"📥 DCA mensuel ajouté: +{MONTHLY_DCA}€ → Cash: {portfolio['cash']:.2f}€")
        portfolio["last_dca_date"] = today
    
    # Télécharger les données
    print("\n📡 Téléchargement des données...")
    
    tickers = DATABASE + ["^VIX"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), 
                       end=end_date.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    
    if data.empty:
        print("❌ Erreur de téléchargement")
        return
    
    close = data['Close'].ffill().bfill()
    high = data['High'].ffill().bfill()
    low = data['Low'].ffill().bfill()
    
    vix = close["^VIX"] if "^VIX" in close.columns else pd.Series(index=close.index, data=15)
    prices = close.drop(columns=["^VIX"], errors="ignore")
    high_prices = high.drop(columns=["^VIX"], errors="ignore")
    low_prices = low.drop(columns=["^VIX"], errors="ignore")
    
    current_vix = float(vix.iloc[-1]) if not pd.isna(vix.iloc[-1]) else 15
    current_prices = prices.iloc[-1]
    idx = len(prices) - 1
    
    # Régime
    if current_vix >= VIX_ULTRA_DEFENSIVE:
        regime = "🔴 ULTRA-DÉFENSIF"
        max_positions = MAX_POSITIONS_ULTRA_DEFENSIVE
        defensive = True
    elif current_vix >= VIX_DEFENSIVE:
        regime = "🟡 DÉFENSIF"
        max_positions = MAX_POSITIONS_DEFENSIVE
        defensive = True
    else:
        regime = "🟢 NORMAL"
        max_positions = MAX_POSITIONS_NORMAL
        defensive = False
    
    print(f"\n📈 VIX: {current_vix:.1f} → {regime}")
    print(f"📊 Max positions: {max_positions}")
    
    # Indicateurs
    atr_pct = pd.DataFrame(index=prices.index)
    sma = pd.DataFrame(index=prices.index)
    high_60 = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        if ticker in high_prices.columns and ticker in low_prices.columns:
            tr1 = high_prices[ticker] - low_prices[ticker]
            tr2 = abs(high_prices[ticker] - prices[ticker].shift(1))
            tr3 = abs(low_prices[ticker] - prices[ticker].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_pct[ticker] = tr.rolling(ATR_PERIOD).mean() / prices[ticker]
        sma[ticker] = prices[ticker].rolling(SMA_PERIOD).mean()
        high_60[ticker] = prices[ticker].rolling(HIGH_LOOKBACK).max()
    
    # ============================================================
    # SIGNAUX
    # ============================================================
    
    signals = {"sell": [], "buy": [], "pyramid": []}
    
    print(f"\n{'='*70}")
    print(f"📂 POSITIONS ACTUELLES ({len(portfolio['positions'])})")
    print(f"{'='*70}")
    
    if not portfolio['positions']:
        print("\n   💵 Aucune position - 100% Cash")
    
    # Vérifier chaque position
    for ticker, pos in list(portfolio["positions"].items()):
        if ticker not in current_prices.index or pd.isna(current_prices[ticker]):
            print(f"⚠️ {ticker}: Prix non disponible")
            continue
        
        current_price_usd = float(current_prices[ticker])
        current_price_eur = usd_to_eur(current_price_usd, eur_rate)
        entry_price_eur = pos["entry_price_eur"]
        shares = pos["shares"]
        value_eur = current_price_eur * shares
        
        # Update peak
        peak_eur = pos.get("peak_price_eur", entry_price_eur)
        if current_price_eur > peak_eur:
            portfolio["positions"][ticker]["peak_price_eur"] = current_price_eur
            peak_eur = current_price_eur
        
        pnl_pct = (current_price_eur / entry_price_eur - 1) * 100
        pnl_eur = (current_price_eur - entry_price_eur) * shares
        
        pyramided = "🔺" if pos.get("pyramided", False) else ""
        print(f"\n{'📈' if pnl_pct > 0 else '📉'} {ticker}{pyramided}: {shares:.4f} actions")
        print(f"   Prix: {current_price_eur:.2f}€ | Entrée: {entry_price_eur:.2f}€")
        print(f"   Valeur: {value_eur:.2f}€ | PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        
        # ============================================================
        # v30.5 - VÉRIFIER PYRAMIDING
        # ============================================================
        
        should_pyramid, add_amount = check_pyramid_signal(pos, current_price_eur, high_prices, ticker, idx)
        
        if should_pyramid and portfolio["cash"] >= add_amount + COST_PER_TRADE:
            add_shares = (add_amount - usd_to_eur(COST_PER_TRADE, eur_rate)) / current_price_eur
            
            signals["pyramid"].append({
                "ticker": ticker,
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "add_shares": add_shares,
                "add_amount_eur": add_amount,
                "current_gain_pct": pnl_pct
            })
            print(f"   🔺 PYRAMIDING SIGNAL: +{add_shares:.4f} actions (+{add_amount:.0f}€)")
        
        # Vérifier trailing stop
        gain_ratio = current_price_eur / entry_price_eur
        sell_reason = None
        
        if gain_ratio >= TRAILING_STOP_ACTIVATION:
            trail_stop = peak_eur * (1 - TRAILING_STOP_PCT)
            if current_price_eur < trail_stop:
                sell_reason = f"TRAILING STOP (peak: {peak_eur:.2f}€)"
        elif gain_ratio >= MINI_TRAIL_ACTIVATION:
            mini_stop = peak_eur * (1 - MINI_TRAIL_PCT)
            if current_price_eur < mini_stop:
                sell_reason = f"MINI-TRAIL (peak: {peak_eur:.2f}€)"
        
        # Stop loss
        sl_pct = get_stop_loss_pct(ticker, defensive)
        stop_price_eur = entry_price_eur * (1 - sl_pct)
        
        if current_price_eur < stop_price_eur:
            sell_reason = f"STOP LOSS (-{sl_pct*100:.0f}%)"
        
        if sell_reason:
            signals["sell"].append({
                "ticker": ticker,
                "reason": sell_reason,
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "shares": shares,
                "value_eur": value_eur,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct
            })
    
    # ============================================================
    # CHERCHER NOUVEAUX SIGNAUX
    # ============================================================
    
    scores = score_hybride(prices, idx, current_vix)
    scores = scores.dropna().sort_values(ascending=False)
    
    def can_enter(ticker):
        if ticker not in prices.columns:
            return False
        price = current_prices[ticker]
        if pd.isna(price):
            return False
        price = float(price)
        
        if ticker in atr_pct.columns:
            atr_val = atr_pct[ticker].iloc[idx]
            if not pd.isna(atr_val) and atr_val >= get_atr_threshold(ticker):
                return False
        
        if ticker in sma.columns:
            sma_val = sma[ticker].iloc[idx]
            if not pd.isna(sma_val) and price <= float(sma_val):
                return False
        
        if ticker in high_60.columns:
            high_val = high_60[ticker].iloc[idx]
            if not pd.isna(high_val) and float(high_val) > 0:
                if (price / float(high_val) - 1) < -MAX_DRAWDOWN:
                    return False
        
        if idx >= 21:
            ret_1m = float(prices[ticker].iloc[idx]) / float(prices[ticker].iloc[idx - 21]) - 1
            if ret_1m < 0:
                return False
        
        return True
    
    valid_tickers = [t for t in scores.index if can_enter(t)]
    valid_scores = scores[valid_tickers]
    top_candidates = valid_scores.head(max_positions).index.tolist()
    
    # Vérifier rotations
    current_positions = list(portfolio["positions"].keys())
    
    for ticker in current_positions:
        if ticker in [s["ticker"] for s in signals["sell"]]:
            continue
        
        if ticker not in top_candidates:
            pos = portfolio["positions"][ticker]
            entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_date).days
            
            if days_held >= get_min_holding_days(ticker):
                for candidate in top_candidates:
                    if candidate not in current_positions:
                        current_score = pos.get("score", 0)
                        new_score = valid_scores.get(candidate, 0)
                        
                        if new_score > current_score * (1 + get_rotation_threshold(ticker)):
                            current_price_usd = float(current_prices[ticker])
                            current_price_eur = usd_to_eur(current_price_usd, eur_rate)
                            entry_price_eur = pos["entry_price_eur"]
                            shares = pos["shares"]
                            value_eur = current_price_eur * shares
                            pnl_eur = (current_price_eur - entry_price_eur) * shares
                            pnl_pct = (current_price_eur / entry_price_eur - 1) * 100
                            
                            signals["sell"].append({
                                "ticker": ticker,
                                "reason": f"ROTATION → {candidate}",
                                "price_usd": current_price_usd,
                                "price_eur": current_price_eur,
                                "shares": shares,
                                "value_eur": value_eur,
                                "pnl_eur": pnl_eur,
                                "pnl_pct": pnl_pct
                            })
                            break
    
    # ============================================================
    # CALCULER CASH DISPONIBLE
    # ============================================================
    
    # Cash après pyramiding
    cash_after_pyramid = portfolio["cash"]
    for pyr in signals["pyramid"]:
        cash_after_pyramid -= pyr["add_amount_eur"] + usd_to_eur(COST_PER_TRADE, eur_rate)
    
    # Cash après ventes
    cash_after_sells = cash_after_pyramid
    for sell in signals["sell"]:
        proceeds = sell["value_eur"] - usd_to_eur(COST_PER_TRADE, eur_rate)
        cash_after_sells += proceeds
    
    positions_after_sells = len(current_positions) - len(signals["sell"])
    available_slots = max_positions - positions_after_sells
    
    # ============================================================
    # v30.5 - ALLOCATION PONDÉRÉE POUR ACHATS
    # ============================================================
    
    if available_slots > 0 and cash_after_sells > 50:
        fees_total = available_slots * usd_to_eur(COST_PER_TRADE, eur_rate)
        cash_for_investing = cash_after_sells - fees_total
        
        # Collecter les candidats
        buy_candidates = []
        for candidate in top_candidates:
            if candidate in current_positions:
                continue
            if candidate in [s["ticker"] for s in signals["sell"]]:
                continue
            if len(buy_candidates) >= available_slots:
                break
            buy_candidates.append(candidate)
        
        # Allocation pondérée
        for rank, candidate in enumerate(buy_candidates, 1):
            price_usd = float(current_prices[candidate])
            price_eur = usd_to_eur(price_usd, eur_rate)
            
            # v30.5: Allocation pondérée par rang
            amount_to_invest = get_weighted_allocation(rank, len(buy_candidates), cash_for_investing)
            shares = amount_to_invest / price_eur
            
            sl_pct = get_stop_loss_pct(candidate, defensive)
            stop_price_eur = price_eur * (1 - sl_pct)
            
            signals["buy"].append({
                "ticker": candidate,
                "price_usd": price_usd,
                "price_eur": price_eur,
                "shares": shares,
                "amount_eur": amount_to_invest,
                "score": valid_scores[candidate],
                "stop_loss_eur": stop_price_eur,
                "stop_loss_pct": sl_pct * 100,
                "rank": rank,
                "allocation_pct": (amount_to_invest / cash_for_investing * 100) if cash_for_investing > 0 else 0
            })
    
    # ============================================================
    # AFFICHER SIGNAUX
    # ============================================================
    
    print(f"\n{'='*70}")
    print(f"🚨 SIGNAUX DU JOUR")
    print(f"{'='*70}")
    
    if not signals["sell"] and not signals["buy"] and not signals["pyramid"]:
        print("\n✅ Aucun signal aujourd'hui - HOLD")
    
    # Pyramiding
    if signals["pyramid"]:
        print(f"\n🔺 PYRAMIDING ({len(signals['pyramid'])})")
        print("-"*70)
        for pyr in signals["pyramid"]:
            print(f"""
   {pyr['ticker']} (déjà +{pyr['current_gain_pct']:.1f}%)
   ├─ Ajouter: {pyr['add_shares']:.4f} actions
   ├─ Montant: {pyr['add_amount_eur']:.2f}€ + 1€ frais
   └─ Prix: {pyr['price_eur']:.2f}€
""")
    
    # Ventes
    if signals["sell"]:
        print(f"\n🔴 VENTES ({len(signals['sell'])})")
        print("-"*70)
        for sell in signals["sell"]:
            print(f"""
   {sell['ticker']}
   ├─ Raison: {sell['reason']}
   ├─ Prix actuel: {sell['price_eur']:.2f}€
   ├─ Actions: {sell['shares']:.4f}
   ├─ Montant: {sell['value_eur']:.2f}€
   └─ PnL: {sell['pnl_eur']:+.2f}€ ({sell['pnl_pct']:+.1f}%)
""")
    
    # Achats
    if signals["buy"]:
        print(f"\n🟢 ACHATS ({len(signals['buy'])}) - Allocation pondérée")
        print("-"*70)
        for buy in signals["buy"]:
            print(f"""
   #{buy['rank']} {buy['ticker']} ({buy['allocation_pct']:.0f}% du cash)
   ┌─────────────────────────────────────────────
   │ 💶 MONTANT: {buy['amount_eur']:.2f}€ + 1€ frais
   │ 📊 ACTIONS: {buy['shares']:.4f}
   └─────────────────────────────────────────────
   ├─ Prix: {buy['price_eur']:.2f}€
   ├─ Score: {buy['score']:.3f}
   └─ Stop Loss: {buy['stop_loss_eur']:.2f}€ (-{buy['stop_loss_pct']:.0f}%)
""")
    
    # ============================================================
    # TOP 5 TENDANCES
    # ============================================================
    
    print(f"\n{'='*70}")
    print(f"🏆 TOP 5 TENDANCES")
    print(f"{'='*70}")
    
    top5 = valid_scores.head(5)
    current_positions = list(portfolio["positions"].keys())
    
    print(f"\n{'Rang':<6} {'Ticker':<8} {'Prix €':<12} {'Score':<10} {'Statut'}")
    print("-"*60)
    
    for i, (ticker, score) in enumerate(top5.items(), 1):
        price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
        
        if ticker in current_positions:
            statut = "📂 EN PORTEFEUILLE"
        elif ticker in [b["ticker"] for b in signals["buy"]]:
            statut = "🟢 ACHAT"
        else:
            statut = "👀 À surveiller"
        
        print(f"   {i:<4} {ticker:<8} {price_eur:>8.2f}€    {score:>6.3f}    {statut}")
    
    top5_data = []
    for ticker, score in top5.items():
        price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
        top5_data.append({
            "ticker": ticker,
            "price_eur": price_eur,
            "score": score,
            "in_portfolio": ticker in current_positions,
            "in_buy": ticker in [b["ticker"] for b in signals["buy"]]
        })
    
    # ============================================================
    # EXÉCUTER LES TRADES
    # ============================================================
    
    # Pyramiding
    for pyr in signals["pyramid"]:
        ticker = pyr["ticker"]
        pos = portfolio["positions"][ticker]
        
        old_shares = pos["shares"]
        old_entry = pos["entry_price_eur"]
        add_shares = pyr["add_shares"]
        new_price = pyr["price_eur"]
        
        # Nouveau prix moyen pondéré
        new_shares = old_shares + add_shares
        new_entry = (old_shares * old_entry + add_shares * new_price) / new_shares
        
        portfolio["positions"][ticker]["shares"] = new_shares
        portfolio["positions"][ticker]["entry_price_eur"] = new_entry
        portfolio["positions"][ticker]["pyramided"] = True
        
        portfolio["cash"] -= pyr["add_amount_eur"] + usd_to_eur(COST_PER_TRADE, eur_rate)
        
        log_trade(history, "PYRAMID", ticker, pyr["price_usd"], pyr["price_eur"],
                  add_shares, pyr["add_amount_eur"], eur_rate, reason="pyramid_add")
    
    # Ventes
    for sell in signals["sell"]:
        ticker = sell["ticker"]
        proceeds = sell["value_eur"] - usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] += proceeds
        
        log_trade(history, "SELL", ticker, sell["price_usd"], sell["price_eur"],
                  sell["shares"], sell["value_eur"], eur_rate,
                  reason=sell["reason"], pnl_eur=sell["pnl_eur"], pnl_pct=sell["pnl_pct"])
        
        del portfolio["positions"][ticker]
    
    # Achats
    for buy in signals["buy"]:
        ticker = buy["ticker"]
        cost = buy["amount_eur"] + usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] -= cost
        
        portfolio["positions"][ticker] = {
            "entry_price_eur": buy["price_eur"],
            "entry_price_usd": buy["price_usd"],
            "entry_date": today,
            "shares": buy["shares"],
            "initial_amount_eur": buy["amount_eur"],
            "amount_invested_eur": buy["amount_eur"],
            "score": buy["score"],
            "peak_price_eur": buy["price_eur"],
            "stop_loss_eur": buy["stop_loss_eur"],
            "rank": buy["rank"],
            "pyramided": False
        }
        
        log_trade(history, "BUY", ticker, buy["price_usd"], buy["price_eur"],
                  buy["shares"], buy["amount_eur"], eur_rate, reason=f"signal_rank{buy['rank']}")
    
    # ============================================================
    # RÉSUMÉ PORTFOLIO
    # ============================================================
    
    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ PORTFOLIO")
    print(f"{'='*70}")
    
    total_positions_value = 0
    for ticker, pos in portfolio["positions"].items():
        if ticker in current_prices.index and not pd.isna(current_prices[ticker]):
            current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
            value = current_price_eur * pos["shares"]
            total_positions_value += value
    
    total_value = portfolio["cash"] + total_positions_value
    
    start_date = datetime.strptime(portfolio["start_date"], "%Y-%m-%d")
    months_elapsed = (datetime.now().year - start_date.year) * 12 + (datetime.now().month - start_date.month)
    total_invested = portfolio["initial_capital"] + max(0, months_elapsed) * MONTHLY_DCA
    
    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1) * 100 if total_invested > 0 else 0
    
    print(f"""
   💵 Cash disponible:     {portfolio['cash']:.2f}€
   📈 Valeur positions:    {total_positions_value:.2f}€
   ────────────────────────────────────
   💰 VALEUR TOTALE:       {total_value:.2f}€
   📊 Total investi:       {total_invested:.2f}€
   {'📈' if total_pnl >= 0 else '📉'} PnL:                  {total_pnl:+.2f}€ ({total_pnl_pct:+.1f}%)
""")
    
    summary = history["summary"]
    if summary["total_trades"] > 0:
        print(f"   📜 HISTORIQUE")
        print(f"   Trades: {summary['total_trades']} | Pyramids: {summary.get('pyramids', 0)}")
        print(f"   Wins: {summary['winning_trades']} | Losses: {summary['losing_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1f}% | PnL réalisé: {summary['total_pnl_eur']:+.2f}€")
    
    save_portfolio(portfolio)
    save_trades_history(history)
    print(f"\n💾 Portfolio et historique sauvegardés")
    
    # ============================================================
    # TELEGRAM
    # ============================================================
    
    msg = f"📊 <b>APEX v30.5</b> - {today}\n"
    msg += f"{regime} | VIX: {current_vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n\n"
    
    if signals["sell"] or signals["buy"] or signals["pyramid"]:
        msg += f"🚨 <b>ACTIONS À FAIRE</b>\n\n"
        
        for pyr in signals["pyramid"]:
            msg += f"🔺 <b>PYRAMIDER {pyr['ticker']}</b>\n"
            msg += f"   Ajouter: {pyr['add_shares']:.4f} actions\n"
            msg += f"   Montant: <b>{pyr['add_amount_eur']:.2f}€</b> + 1€\n"
            msg += f"   Gain actuel: +{pyr['current_gain_pct']:.1f}%\n\n"
        
        for sell in signals["sell"]:
            msg += f"🔴 <b>VENDRE {sell['ticker']}</b>\n"
            msg += f"   Actions: {sell['shares']:.4f}\n"
            msg += f"   Montant: ~{sell['value_eur']:.2f}€\n"
            msg += f"   Raison: {sell['reason']}\n"
            msg += f"   PnL: {sell['pnl_eur']:+.2f}€ ({sell['pnl_pct']:+.1f}%)\n\n"
        
        for buy in signals["buy"]:
            msg += f"🟢 <b>ACHETER #{buy['rank']} {buy['ticker']}</b>\n"
            msg += f"   💶 Montant: <b>{buy['amount_eur']:.2f}€</b> ({buy['allocation_pct']:.0f}%)\n"
            msg += f"   📊 Actions: <b>{buy['shares']:.4f}</b>\n"
            msg += f"   Prix: {buy['price_eur']:.2f}€\n"
            msg += f"   Stop: {buy['stop_loss_eur']:.2f}€ (-{buy['stop_loss_pct']:.0f}%)\n\n"
    else:
        msg += f"✅ <b>Aucun signal - HOLD</b>\n\n"
    
    # Positions
    msg += f"📂 <b>MES POSITIONS</b>\n"
    for ticker, pos in portfolio["positions"].items():
        if ticker in current_prices.index and not pd.isna(current_prices[ticker]):
            current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
            entry_price_eur = pos["entry_price_eur"]
            shares = pos["shares"]
            pnl_pct = (current_price_eur / entry_price_eur - 1) * 100
            pnl_eur = (current_price_eur - entry_price_eur) * shares
            
            if ticker in valid_scores.index:
                rank = list(valid_scores.index).index(ticker) + 1
                score = valid_scores[ticker]
                rank_str = f"#{rank}"
            else:
                rank_str = "❌"
                score = 0
            
            pyramided = "🔺" if pos.get("pyramided", False) else ""
            emoji = "📈" if pnl_pct >= 0 else "📉"
            msg += f"{emoji} {ticker}{pyramided} ({rank_str}) @ {current_price_eur:.2f}€\n"
            msg += f"   PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%) | Score: {score:.3f}\n"
    msg += f"\n"
    
    msg += f"💰 <b>PORTFOLIO</b>\n"
    msg += f"Valeur: {total_value:.2f}€ ({total_pnl_pct:+.1f}%)\n"
    msg += f"Cash: {portfolio['cash']:.2f}€\n\n"
    
    msg += f"🏆 <b>TOP 5 TENDANCES</b>\n"
    for i, t in enumerate(top5_data, 1):
        if t["in_portfolio"]:
            marker = "📂"
        elif t["in_buy"]:
            marker = "🟢"
        else:
            marker = "👀"
        msg += f"{i}. {t['ticker']} @ {t['price_eur']:.2f}€ ({t['score']:.3f}) {marker}\n"
    
    send_telegram(msg)
    
    print(f"\n{'='*70}")
    print("✅ APEX v30.5 terminé")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
