"""
APEX v31 OPTIMISÉ - PRODUCTION
================================

Paramètres validés sur backtest 2015-2026 (10 ans):
- MFE Threshold: 15% (activer trailing dès +15%)
- Trailing: 5% (vendre si chute de 5% depuis le plus haut)
- Hard Stop: 18% UNIFORME (plus de stops par catégorie!)
- Pas de blacklist

Performance attendue:
- Win Rate: 73.7%
- ROI: +1,181% sur 10 ans
- CAGR: 51%/an
- Max Drawdown: -28.9%
- Sharpe: 2.10

Capital: 1,500€ initial + 100€/mois DCA
Tracking: portfolio.json + trades_history.json
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

# ============================================================
# NOUVEAUX PARAMÈTRES OPTIMISÉS (2015-2026)
# ============================================================

# Nombre de positions
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

# VIX thresholds
VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

# Calcul du score momentum
ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

# ⭐ PARAMÈTRES DE SORTIE OPTIMISÉS (validés 2015-2026)
HARD_STOP_PCT = 0.18            # -18% stop uniforme pour tous les tickers
MFE_THRESHOLD_PCT = 0.15        # Activer trailing à +15%
TRAILING_PCT = 0.05             # -5% du plus haut

# Ancien système de trailing (gardé pour compatibilité mais remplacé par MFE)
TRAILING_STOP_ACTIVATION = 1.40
TRAILING_STOP_PCT = 0.20
MINI_TRAIL_ACTIVATION = 1.25
MINI_TRAIL_PCT = 0.15

# Pyramiding (optionnel)
PYRAMID_GAIN_THRESHOLD = 0.15
PYRAMID_LOOKBACK = 20
PYRAMID_ADD_PCT = 0.50

# Rotation forcée (si score = 0 pendant X jours)
FORCE_ROTATION_DAYS = 10

# ============================================================
# DATABASE - 44 TICKERS (inchangé)
# ============================================================

DATABASE = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA",
    "AMD", "MU", "ASML", "TSM", "LRCX", "AMAT",
    "PLTR", "APP", "CRWD", "NET", "DDOG", "ZS",
    "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER",
    "MSTR", "MARA", "RIOT",
    "CEG",
    "LLY", "NVO", "UNH", "JNJ", "ABBV",
    "WMT", "COST", "PG", "KO",
    "XOM", "CVX",
    "QQQ", "SPY", "GLD", "SLV",
]

# Categories (pour info seulement, plus utilisé pour les stops)
ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML"}
TECH = {"APP", "TSLA", "NVDA", "PLTR", "DDOG"}

def get_category(ticker):
    """Retourne la catégorie du ticker (pour info seulement)"""
    if ticker in ULTRA_VOLATILE: return "ultra"
    elif ticker in CRYPTO: return "crypto"
    elif ticker in SEMI: return "semi"
    elif ticker in TECH: return "tech"
    return "other"

# ============================================================
# ⭐ NOUVEAU: STOP LOSS UNIFORME -18%
# ============================================================

def get_stop_loss_pct(ticker, defensive=False):
    """
    Retourne le % de stop loss.
    NOUVEAU: Stop uniforme à 18% pour tous les tickers.
    """
    base_stop = HARD_STOP_PCT  # 18% uniforme
    return base_stop * 0.85 if defensive else base_stop  # 15.3% si défensif

def calculate_stop_price(entry_price, stop_pct):
    """Calcule le prix du stop loss"""
    return entry_price * (1 - stop_pct)

# ============================================================
# ⭐ NOUVEAU: MFE TRAILING STOP
# ============================================================

def check_mfe_trailing_exit(pos, current_price, entry_price):
    """
    Vérifie si on doit sortir par MFE trailing.
    
    Logique:
    1. Calculer le gain actuel depuis l'entrée
    2. Si le gain max (MFE) a atteint +15%, le trailing est activé
    3. Si le prix chute de 5% depuis le plus haut, on vend
    
    Returns: (should_exit, reason, details)
    """
    peak_price = pos.get('peak_price_eur', entry_price)
    
    # Mise à jour du peak
    if current_price > peak_price:
        peak_price = current_price
        pos['peak_price_eur'] = peak_price
    
    # Calcul du MFE (gain max depuis entrée)
    mfe_pct = (peak_price / entry_price - 1)
    
    # Calcul du drawdown depuis le peak
    drawdown_from_peak = (current_price / peak_price - 1)
    
    # Calcul du gain actuel
    current_gain = (current_price / entry_price - 1)
    
    # Le trailing est-il activé? (MFE >= 15%)
    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT
    
    # Si trailing actif ET prix chute de 5% depuis le peak → VENDRE
    if trailing_active and drawdown_from_peak <= -TRAILING_PCT:
        return True, "MFE_TRAILING", {
            'mfe_pct': mfe_pct * 100,
            'peak_price': peak_price,
            'drawdown_pct': drawdown_from_peak * 100,
            'current_gain_pct': current_gain * 100
        }
    
    return False, None, {
        'trailing_active': trailing_active,
        'mfe_pct': mfe_pct * 100,
        'peak_price': peak_price,
        'drawdown_pct': drawdown_from_peak * 100
    }

def check_hard_stop_exit(current_price, entry_price, stop_price):
    """
    Vérifie si le hard stop (-18%) est touché.
    
    Returns: (should_exit, reason)
    """
    if current_price <= stop_price:
        loss_pct = (current_price / entry_price - 1) * 100
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

# ============================================================
# ALLOCATION PONDÉRÉE (inchangé)
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
# PORTFOLIO MANAGEMENT
# ============================================================

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": INITIAL_CAPITAL,
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {}
    }

def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=4)

def load_trades_history():
    """Charge l'historique des trades avec validation de structure"""
    default_history = {
        "trades": [],
        "summary": {
            "total_trades": 0,
            "buys": 0,
            "sells": 0,
            "pyramids": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl_eur": 0.0,
            "total_fees_eur": 0.0,
            "best_trade_eur": 0.0,
            "worst_trade_eur": 0.0,
            "win_rate": 0.0
        }
    }
    
    if os.path.exists(TRADES_HISTORY_FILE):
        try:
            with open(TRADES_HISTORY_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return default_history
                
                history = json.loads(content)
                
                # Validation et réparation de la structure
                if not isinstance(history, dict):
                    return default_history
                    
                if "trades" not in history:
                    history["trades"] = []
                
                if "summary" not in history:
                    history["summary"] = default_history["summary"]
                else:
                    # Compléter les champs summary manquants
                    for k, v in default_history["summary"].items():
                        if k not in history["summary"]:
                            history["summary"][k] = v
                            
                return history
        except Exception as e:
            print(f"⚠️ Erreur chargement historique ({e}): Réinitialisation.")
            return default_history
            
    return default_history

def save_trades_history(history):
    with open(TRADES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def log_trade(history, action, ticker, price_usd, price_eur, shares, amount_eur, 
              eur_rate, reason="", pnl_eur=None, pnl_pct=None):
    
    # Sécurité supplémentaire
    if "trades" not in history:
        history["trades"] = []
    
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
        "fee_eur": COST_PER_TRADE,
        "eur_usd_rate": round(eur_rate, 4),
        "reason": reason
    }
    
    if pnl_eur is not None:
        trade["pnl_eur"] = round(pnl_eur, 2)
        trade["pnl_pct"] = round(pnl_pct, 2)
    
    history["trades"].append(trade)
    
    # Update summary
    if "summary" not in history:
        history["summary"] = {
            "total_trades": 0, "buys": 0, "sells": 0, "pyramids": 0,
            "winning_trades": 0, "losing_trades": 0, "total_pnl_eur": 0.0,
            "total_fees_eur": 0.0, "best_trade_eur": 0.0, "worst_trade_eur": 0.0, "win_rate": 0.0
        }
        
    summary = history["summary"]
    summary["total_trades"] += 1
    summary["total_fees_eur"] += COST_PER_TRADE
    
    if action == "BUY":
        summary["buys"] += 1
    elif action == "SELL":
        summary["sells"] += 1
        if pnl_eur is not None:
            summary["total_pnl_eur"] += pnl_eur
            if pnl_eur > 0:
                summary["winning_trades"] += 1
            else:
                summary["losing_trades"] += 1
            
            if pnl_eur > summary.get("best_trade_eur", 0):
                summary["best_trade_eur"] = pnl_eur
            if pnl_eur < summary.get("worst_trade_eur", 0):
                summary["worst_trade_eur"] = pnl_eur
            
            total_closed = summary["winning_trades"] + summary["losing_trades"]
            if total_closed > 0:
                summary["win_rate"] = round(summary["winning_trades"] / total_closed * 100, 1)
    elif action == "PYRAMID":
        summary["pyramids"] += 1

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configuré")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

# ============================================================
# MARKET DATA & SCORING
# ============================================================

def get_market_data(tickers, days=100):
    """Télécharge les données de marché"""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    try:
        data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
        return data
    except Exception as e:
        print(f"Erreur download: {e}")
        return None

def calculate_momentum_score(close, high, atr_period=14, sma_period=20, high_lookback=60):
    """
    Calcule le score momentum d'un ticker.
    Score = (Prix - SMA) / ATR, normalisé par high 60j.
    """
    if len(close) < max(atr_period, sma_period, high_lookback):
        return np.nan
    
    # SMA
    sma = close.rolling(sma_period).mean()
    
    # ATR
    tr = high - close.shift(1)
    tr = tr.abs()
    atr = tr.rolling(atr_period).mean()
    
    # High 60j
    high_60 = high.rolling(high_lookback).max()
    
    # Score
    score = (close - sma) / atr
    score = score / (high_60 / close)  # Normalisation
    
    return score.iloc[-1] if not pd.isna(score.iloc[-1]) else np.nan

def get_vix():
    """Récupère le VIX actuel"""
    try:
        vix = yf.Ticker("^VIX")
        return vix.info.get('regularMarketPrice') or vix.info.get('previousClose') or 20
    except:
        return 20

def get_regime(vix):
    """Détermine le régime de marché basé sur VIX"""
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "🔴 ULTRA-DÉFENSIF", MAX_POSITIONS_ULTRA_DEFENSIVE
    elif vix >= VIX_DEFENSIVE:
        return "🟡 DÉFENSIF", MAX_POSITIONS_DEFENSIVE
    else:
        return "🟢 NORMAL", MAX_POSITIONS_NORMAL

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 APEX v31 OPTIMISÉ - PRODUCTION")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"⚙️  Paramètres: Hard Stop -18%, MFE Trailing +15%/-5%")
    
    # Load data
    portfolio = load_portfolio()
    history = load_trades_history()
    
    # Get market info
    eur_rate = get_eur_usd_rate()
    current_vix = get_vix()
    regime, max_positions = get_regime(current_vix)
    defensive = current_vix >= VIX_DEFENSIVE
    
    print(f"\n💱 EUR/USD: {eur_rate:.4f}")
    print(f"📊 VIX: {current_vix:.1f}")
    print(f"📈 Régime: {regime} (max {max_positions} positions)")
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # DCA mensuel
    last_dca = portfolio.get("last_dca_date")
    current_month = datetime.now().strftime("%Y-%m")
    
    if last_dca is None or not last_dca.startswith(current_month):
        portfolio["cash"] += MONTHLY_DCA
        portfolio["last_dca_date"] = today
        print(f"\n💰 DCA mensuel: +{MONTHLY_DCA}€")
    
    # Download market data
    print("\n📥 Téléchargement des données...")
    data = get_market_data(DATABASE)
    
    if data is None or data.empty:
        print("❌ Erreur: pas de données")
        send_telegram("❌ APEX v31: Erreur téléchargement données")
        return
    
    # Calculate scores
    scores = {}
    current_prices = {}
    
    for ticker in DATABASE:
        try:
            # Gestion multi-index ou simple index selon le nombre de tickers
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.get_level_values(0):
                    close = data[ticker]['Close'].dropna()
                    high = data[ticker]['High'].dropna()
                else:
                    continue
            else:
                 # Cas rare si 1 seul ticker téléchargé
                 close = data['Close'].dropna()
                 high = data['High'].dropna()

            if len(close) > 0:
                current_prices[ticker] = close.iloc[-1]
                score = calculate_momentum_score(close, high)
                if not np.isnan(score) and score > 0:
                    scores[ticker] = score
        except Exception as e:
            continue
    
    current_prices = pd.Series(current_prices)
    valid_scores = pd.Series(scores).sort_values(ascending=False)
    
    print(f"\n📊 {len(valid_scores)} tickers avec score > 0")
    
    # ============================================================
    # SIGNAUX
    # ============================================================
    
    signals = {"sell": [], "buy": [], "pyramid": [], "force_rotation": []}
    
    # ============================================================
    # 1. VÉRIFIER LES POSITIONS EXISTANTES
    # ============================================================
    
    print(f"\n{'='*70}")
    print("📂 VÉRIFICATION DES POSITIONS")
    print(f"{'='*70}")
    
    positions_to_remove = []
    
    for ticker, pos in portfolio["positions"].items():
        if ticker not in current_prices.index:
            continue
        
        current_price_usd = float(current_prices[ticker])
        current_price_eur = usd_to_eur(current_price_usd, eur_rate)
        entry_price_eur = pos["entry_price_eur"]
        shares = pos["shares"]
        
        # Mise à jour du peak
        if current_price_eur > pos.get('peak_price_eur', entry_price_eur):
            pos['peak_price_eur'] = current_price_eur
        
        stop_pct = get_stop_loss_pct(ticker, defensive)
        stop_price_eur = calculate_stop_price(entry_price_eur, stop_pct)
        pos['stop_loss_eur'] = stop_price_eur
        
        pnl_eur = (current_price_eur - entry_price_eur) * shares
        pnl_pct = (current_price_eur / entry_price_eur - 1) * 100
        
        # Current score
        current_score = valid_scores.get(ticker, 0)
        pos['score'] = current_score
        
        # Update rank
        if ticker in valid_scores.index:
            pos['rank'] = list(valid_scores.index).index(ticker) + 1
        else:
            pos['rank'] = 999
        
        print(f"\n🔹 {ticker}")
        print(f"   Prix: {current_price_eur:.2f}€ (entrée: {entry_price_eur:.2f}€)")
        print(f"   PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f"   Peak: {pos.get('peak_price_eur', entry_price_eur):.2f}€")
        print(f"   Score: {current_score:.3f} | Rank: #{pos['rank']}")
        
        should_sell = False
        sell_reason = ""
        
        # CHECK 1: Hard Stop -18%
        hit_hard_stop, hard_stop_reason = check_hard_stop_exit(
            current_price_eur, entry_price_eur, stop_price_eur
        )
        
        if hit_hard_stop:
            should_sell = True
            sell_reason = hard_stop_reason
            print(f"   ❌ HARD STOP touché! ({stop_price_eur:.2f}€)")
        
        # CHECK 2: MFE Trailing
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(
                pos, current_price_eur, entry_price_eur
            )
            
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason
                print(f"   📉 MFE TRAILING déclenché!")
                print(f"      MFE: +{mfe_details['mfe_pct']:.1f}%")
                print(f"      Drawdown: {mfe_details['drawdown_pct']:.1f}%")
            else:
                status = "ACTIF" if mfe_details['trailing_active'] else "INACTIF"
                print(f"   🎯 Trailing: {status} (MFE: +{mfe_details['mfe_pct']:.1f}%)")
        
        # CHECK 3: Force rotation (score = 0 pendant X jours)
        if not should_sell and current_score <= 0:
            days_zero = pos.get("days_zero_score", 0) + 1
            pos["days_zero_score"] = days_zero
            print(f"   ⚠️ Score ≤ 0 depuis {days_zero} jour(s)")
            
            if days_zero >= FORCE_ROTATION_DAYS:
                # Trouver un remplaçant
                for candidate in valid_scores.index:
                    if candidate not in portfolio["positions"]:
                        signals["force_rotation"].append({
                            "ticker": ticker,
                            "replacement": candidate,
                            "replacement_score": valid_scores[candidate],
                            "shares": shares,
                            "price_eur": current_price_eur,
                            "pnl_eur": pnl_eur,
                            "pnl_pct": pnl_pct,
                            "days_zero": days_zero
                        })
                        should_sell = True
                        sell_reason = f"FORCE_ROTATION_{days_zero}j"
                        break
        else:
            pos["days_zero_score"] = 0
        
        if should_sell:
            signals["sell"].append({
                "ticker": ticker,
                "shares": shares,
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "value_eur": current_price_eur * shares,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct,
                "reason": sell_reason
            })
            positions_to_remove.append(ticker)
    
    # ============================================================
    # 2. OPPORTUNITÉS D'ACHAT
    # ============================================================
    
    available_cash = portfolio["cash"]
    
    # Compter les positions après ventes
    future_positions = len(portfolio["positions"]) - len(positions_to_remove)
    slots_available = max_positions - future_positions
    
    if slots_available > 0 and available_cash > 50:
        print(f"\n{'='*70}")
        print(f"🛒 OPPORTUNITÉS D'ACHAT ({slots_available} slots disponibles)")
        print(f"{'='*70}")
        
        for ticker in valid_scores.index:
            if slots_available <= 0 or available_cash < 50:
                break
            
            if ticker in portfolio["positions"] and ticker not in positions_to_remove:
                continue
            
            # Force rotation achats
            is_replacement = False
            for rot in signals["force_rotation"]:
                if rot["replacement"] == ticker:
                    is_replacement = True
            
            # Si c'est un replacement, on l'accepte, sinon on vérifie le rang
            rank = list(valid_scores.index).index(ticker) + 1
            if rank > max_positions and not is_replacement:
                continue
            
            current_price_usd = float(current_prices[ticker])
            current_price_eur = usd_to_eur(current_price_usd, eur_rate)
            
            # Allocation pondérée
            allocation = get_weighted_allocation(rank, max_positions, available_cash)
            allocation = min(allocation, available_cash - 10)  # Garder 10€ min
            
            if allocation < 50:
                continue
            
            shares = allocation / current_price_eur
            stop_pct = get_stop_loss_pct(ticker, defensive)
            stop_price = calculate_stop_price(current_price_eur, stop_pct)
            
            signals["buy"].append({
                "ticker": ticker,
                "rank": rank,
                "score": valid_scores[ticker],
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "shares": shares,
                "amount_eur": allocation,
                "allocation_pct": allocation / available_cash * 100,
                "stop_loss_eur": stop_price,
                "stop_loss_pct": stop_pct * 100
            })
            
            available_cash -= allocation
            slots_available -= 1
            
            print(f"\n🟢 ACHETER #{rank}: {ticker}")
            print(f"   Score: {valid_scores[ticker]:.3f}")
            print(f"   Montant: {allocation:.2f}€")
            print(f"   Actions: {shares:.4f}")
            print(f"   Stop: {stop_price:.2f}€ (-{stop_pct*100:.0f}%)")
    
    # ============================================================
    # EXÉCUTION DES ORDRES
    # ============================================================
    
    # Ventes
    for sell in signals["sell"]:
        ticker = sell["ticker"]
        proceeds = sell["value_eur"] - usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] += proceeds
        
        log_trade(history, "SELL", ticker, sell["price_usd"], sell["price_eur"],
                  sell["shares"], sell["value_eur"], eur_rate,
                  reason=sell["reason"], pnl_eur=sell["pnl_eur"], pnl_pct=sell["pnl_pct"])
        
        if ticker in portfolio["positions"]:
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
            "pyramided": False,
            "days_zero_score": 0
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
        if ticker in current_prices.index:
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
    
    save_portfolio(portfolio)
    save_trades_history(history)
    print(f"\n💾 Portfolio et historique sauvegardés")
    
    # ============================================================
    # TELEGRAM
    # ============================================================
    
    msg = f"📊 <b>APEX v31 OPTIMISÉ</b> - {today}\n"
    msg += f"{regime} | VIX: {current_vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += f"⚙️ Stop: -18% | Trail: +15%/-5%\n\n"
    
    if signals["sell"] or signals["buy"] or signals["force_rotation"]:
        msg += f"🚨 <b>ACTIONS À FAIRE</b>\n\n"
        
        for rot in signals["force_rotation"]:
            msg += f"🔄 <b>ROTATION FORCÉE</b>\n"
            msg += f"   {rot['ticker']} → {rot['replacement']}\n"
            msg += f"   Score=0 depuis {rot['days_zero']}j\n\n"
        
        for sell in signals["sell"]:
            msg += f"🔴 <b>VENDRE {sell['ticker']}</b>\n"
            msg += f"   Actions: {sell['shares']:.4f}\n"
            msg += f"   Montant: ~{sell['value_eur']:.2f}€\n"
            msg += f"   Raison: {sell['reason']}\n"
            msg += f"   PnL: {sell['pnl_eur']:+.2f}€ ({sell['pnl_pct']:+.1f}%)\n\n"
        
        for buy in signals["buy"]:
            msg += f"🟢 <b>ACHETER #{buy['rank']} {buy['ticker']}</b>\n"
            msg += f"   💶 Montant: <b>{buy['amount_eur']:.2f}€</b>\n"
            msg += f"   📊 Actions: <b>{buy['shares']:.4f}</b>\n"
            msg += f"   Stop: {buy['stop_loss_eur']:.2f}€ (-18%)\n"
            msg += f"   MFE Trigger: {buy['price_eur']*1.15:.2f}€ (+15%)\n\n"
    else:
        msg += f"✅ <b>Aucun signal - HOLD</b>\n\n"
    
    # Positions
    msg += f"📂 <b>MES POSITIONS</b>\n"
    for ticker, pos in portfolio["positions"].items():
        if ticker in current_prices.index:
            current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
            entry_price_eur = pos["entry_price_eur"]
            pnl_pct = (current_price_eur / entry_price_eur - 1) * 100
            pnl_eur = (current_price_eur - entry_price_eur) * pos["shares"]
            mfe_pct = (pos.get('peak_price_eur', entry_price_eur) / entry_price_eur - 1) * 100
            
            trailing_status = "🟢ACTIF" if mfe_pct >= 15 else "⚪️"
            
            emoji = "📈" if pnl_pct >= 0 else "📉"
            msg += f"{emoji} {ticker} #{pos.get('rank', 'N/A')}\n"
            msg += f"   {pos['amount_invested_eur']:.0f}€ → {current_price_eur * pos['shares']:.0f}€\n"
            msg += f"   PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)\n"
            msg += f"   Trail: {trailing_status} MFE:+{mfe_pct:.1f}%\n"
    
    msg += f"\n💰 <b>TOTAL: {total_value:.2f}€</b> ({total_pnl_pct:+.1f}%)\n"
    
    # Top 5
    msg += f"\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, ticker in enumerate(valid_scores.head(5).index, 1):
        price = usd_to_eur(float(current_prices[ticker]), eur_rate)
        in_pf = "📂" if ticker in portfolio["positions"] else "👀"
        msg += f"{i}. {ticker} @ {price:.2f}€ ({valid_scores[ticker]:.3f}) {in_pf}\n"
    
    send_telegram(msg)
    
    print(f"\n{'='*70}")
    print("✅ APEX v31 OPTIMISÉ terminé")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
