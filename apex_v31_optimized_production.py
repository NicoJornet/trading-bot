"""
APEX v31.4 - HYBRID (PRODUCTION + SMART CASH)
=============================================
Base: APEX v31 Optimisé (Affichage riche, Momentum info)
Moteur: v31.3 (Arbitrage immédiat + Renforcement positions existantes)

Fonctionnalités:
- Affiche en détail les Scores, Rangs et Trailing de chaque position.
- Utilise le cash des ventes AVANT qu'elles soient exécutées pour réinvestir.
- RENFORCE les positions gagnantes si elles sont sous-allouées.
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
# PARAMÈTRES STRATÉGIQUES
# ============================================================

MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

# Paramètres de sortie
HARD_STOP_PCT = 0.18        # -18% sur prix d'entrée/PRU
MFE_THRESHOLD_PCT = 0.15    # Activation Trailing à +15%
TRAILING_PCT = 0.05         # Chute de 5% depuis le plus haut

FORCE_ROTATION_DAYS = 10

# ============================================================
# DATABASE
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

# ============================================================
# FONCTIONS LOGIQUES
# ============================================================

def get_stop_loss_pct(ticker, defensive=False):
    base_stop = HARD_STOP_PCT
    return base_stop * 0.85 if defensive else base_stop

def calculate_stop_price(entry_price, stop_pct):
    return entry_price * (1 - stop_pct)

def check_mfe_trailing_exit(pos, current_price, entry_price):
    """Logique Trailing MFE"""
    peak_price = pos.get('peak_price_eur', entry_price)
    
    # Mise à jour du peak
    if current_price > peak_price:
        peak_price = current_price
        pos['peak_price_eur'] = peak_price
    
    mfe_pct = (peak_price / entry_price - 1)
    drawdown_from_peak = (current_price / peak_price - 1)
    current_gain = (current_price / entry_price - 1)
    
    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT
    
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
    if current_price <= stop_price:
        loss_pct = (current_price / entry_price - 1) * 100
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

def get_weighted_allocation(rank, num_positions, total_capital):
    if num_positions == 1: return total_capital
    elif num_positions == 2: weights = {1: 0.60, 2: 0.40}
    elif num_positions == 3: weights = {1: 0.50, 2: 0.30, 3: 0.20}
    else:
        total_weight = sum(range(1, num_positions + 1))
        weights = {i: (num_positions - i + 1) / total_weight for i in range(1, num_positions + 1)}
    
    return total_capital * weights.get(rank, 1.0 / num_positions)

def get_eur_usd_rate():
    try:
        eur_usd = yf.Ticker("EURUSD=X")
        rate = eur_usd.info.get('regularMarketPrice') or eur_usd.info.get('previousClose')
        return rate if rate and rate > 0 else 1.08
    except: return 1.08

def usd_to_eur(amount_usd, rate=None):
    if rate is None: rate = get_eur_usd_rate()
    return amount_usd / rate

# ============================================================
# GESTION FICHIERS (ROBUSTE)
# ============================================================

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            try: return json.load(f)
            except: pass
    return {
        "currency": "EUR", "initial_capital": INITIAL_CAPITAL, "monthly_dca": MONTHLY_DCA,
        "cash": INITIAL_CAPITAL, "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None, "positions": {}
    }

def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=4)

def load_trades_history():
    """Charge l'historique avec structure sécurisée"""
    default = {"trades": [], "summary": {"total_trades": 0, "buys": 0, "sells": 0, "total_pnl_eur": 0.0}}
    if os.path.exists(TRADES_HISTORY_FILE):
        try:
            with open(TRADES_HISTORY_FILE, "r") as f:
                data = json.load(f)
                if "trades" not in data: data["trades"] = []
                if "summary" not in data: data["summary"] = default["summary"]
                return data
        except: pass
    return default

def save_trades_history(history):
    with open(TRADES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def log_trade(history, action, ticker, price_usd, price_eur, shares, amount_eur, 
              eur_rate, reason="", pnl_eur=None, pnl_pct=None):
    trade = {
        "id": len(history["trades"]) + 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "action": action, "ticker": ticker, "shares": round(shares, 4),
        "price_usd": round(price_usd, 2), "price_eur": round(price_eur, 2),
        "amount_eur": round(amount_eur, 2), "fee_eur": COST_PER_TRADE,
        "eur_usd_rate": round(eur_rate, 4), "reason": reason
    }
    if pnl_eur is not None:
        trade["pnl_eur"], trade["pnl_pct"] = round(pnl_eur, 2), round(pnl_pct, 2)
    
    history["trades"].append(trade)
    history["summary"]["total_trades"] += 1
    if action in ["BUY", "REINFORCE"]: history["summary"]["buys"] += 1
    elif action == "SELL": 
        history["summary"]["sells"] += 1
        if pnl_eur: history["summary"]["total_pnl_eur"] += pnl_eur

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except: pass

# ============================================================
# MARKET DATA & SCORING
# ============================================================

def get_market_data(tickers, days=100):
    end = datetime.now()
    start = end - timedelta(days=days)
    try: return yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    except: return None

def calculate_momentum_score(close, high):
    """Calcul du score APEX"""
    if len(close) < max(ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK): return np.nan
    sma = close.rolling(SMA_PERIOD).mean()
    tr = (high - close.shift(1)).abs()
    atr = tr.rolling(ATR_PERIOD).mean()
    high_60 = high.rolling(HIGH_LOOKBACK).max()
    score = ((close - sma) / atr) / (high_60 / close)
    return score.iloc[-1] if not pd.isna(score.iloc[-1]) else np.nan

def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        return vix.info.get('regularMarketPrice') or vix.info.get('previousClose') or 20
    except: return 20

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 APEX v31.4 - HYBRID (INFO + SMART CASH)")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load
    portfolio = load_portfolio()
    history = load_trades_history()
    eur_rate = get_eur_usd_rate()
    current_vix = get_vix()
    
    # Regime
    if current_vix >= VIX_ULTRA_DEFENSIVE: max_positions = MAX_POSITIONS_ULTRA_DEFENSIVE; regime="🔴 ULTRA-DEF"
    elif current_vix >= VIX_DEFENSIVE: max_positions = MAX_POSITIONS_DEFENSIVE; regime="🟡 DEF"
    else: max_positions = MAX_POSITIONS_NORMAL; regime="🟢 NORMAL"
    defensive = current_vix >= VIX_DEFENSIVE
    
    print(f"💱 EUR/USD: {eur_rate:.4f}")
    print(f"📊 VIX: {current_vix:.1f} | Régime: {regime}")
    
    # DCA
    today = datetime.now().strftime("%Y-%m-%d")
    if not portfolio.get("last_dca_date", "").startswith(datetime.now().strftime("%Y-%m")):
        portfolio["cash"] += MONTHLY_DCA
        portfolio["last_dca_date"] = today
        print(f"💰 DCA ajouté: +{MONTHLY_DCA}€")

    # Data
    print("\n📥 Téléchargement données...")
    data = get_market_data(DATABASE)
    if data is None or data.empty: return

    # Scoring
    scores, current_prices = {}, {}
    for ticker in DATABASE:
        try:
            if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]: df = data[ticker]
            elif ticker == data.columns[0]: df = data
            else: continue
            
            close, high = df['Close'].dropna(), df['High'].dropna()
            if len(close) > 0:
                current_prices[ticker] = close.iloc[-1]
                s = calculate_momentum_score(close, high)
                if s > 0: scores[ticker] = s
        except: continue
    
    valid_scores = pd.Series(scores).sort_values(ascending=False)
    print(f"📊 {len(valid_scores)} tickers avec score > 0")

    # ============================================================
    # 1. ANALYSE POSITIONS & PROJECTION CASH
    # ============================================================
    print(f"\n{'='*70}\n📂 VÉRIFICATION DES POSITIONS\n{'='*70}")
    
    signals = {"sell": [], "buy": [], "rotation": []}
    positions_to_remove = []
    
    # ⭐ CASH PROJETÉ = Cash Actuel + Ventes prévues
    projected_cash = portfolio["cash"]
    
    # Calcul valeur totale estimée pour allocation pondérée
    total_pf_value = projected_cash + sum(usd_to_eur(float(current_prices[t]), eur_rate)*p["shares"] for t,p in portfolio["positions"].items() if t in current_prices)
    
    for ticker, pos in portfolio["positions"].items():
        if ticker not in current_prices: continue
        
        # Données
        price_usd = float(current_prices[ticker])
        price_eur = usd_to_eur(price_usd, eur_rate)
        shares = pos["shares"]
        
        # Update Peak
        if price_eur > pos.get('peak_price_eur', 0): pos['peak_price_eur'] = price_eur
        
        # Stops
        stop_pct = get_stop_loss_pct(ticker, defensive)
        stop_price = calculate_stop_price(pos["entry_price_eur"], stop_pct)
        pos['stop_loss_eur'] = stop_price
        
        # Infos affichage (Retour de la richesse d'info v31)
        pnl_eur = (price_eur - pos["entry_price_eur"]) * shares
        pnl_pct = (price_eur / pos["entry_price_eur"] - 1) * 100
        curr_score = valid_scores.get(ticker, 0)
        pos['score'] = curr_score
        rank_display = f"#{list(valid_scores.index).index(ticker) + 1}" if ticker in valid_scores.index else "N/A"
        
        print(f"\n🔹 {ticker}")
        print(f"   Prix: {price_eur:.2f}€ (Entrée: {pos['entry_price_eur']:.2f}€)")
        print(f"   PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f"   Score: {curr_score:.3f} | Rank: {rank_display}")
        
        # Logic Vente
        sell, reason = False, ""
        
        hit_stop, r = check_hard_stop_exit(price_eur, pos["entry_price_eur"], stop_price)
        if hit_stop: 
            sell, reason = True, r
            print(f"   ❌ HARD STOP touché ({stop_price:.2f}€)")
        
        if not sell:
            hit_trail, r, det = check_mfe_trailing_exit(pos, price_eur, pos["entry_price_eur"])
            if hit_trail: 
                sell, reason = True, r
                print(f"   📉 MFE TRAILING déclenché (MFE +{det['mfe_pct']:.1f}%)")
            else: 
                status = "ACTIF" if det['trailing_active'] else "INACTIF"
                print(f"   🎯 Trailing: {status} (MFE: +{det['mfe_pct']:.1f}%)")

        # Rotation Forcée
        if not sell and curr_score <= 0:
            pos["days_zero_score"] = pos.get("days_zero_score", 0) + 1
            print(f"   ⚠️ Score ≤ 0 depuis {pos['days_zero_score']} jours")
            if pos["days_zero_score"] >= FORCE_ROTATION_DAYS:
                sell, reason = True, f"ROTATION_{pos['days_zero_score']}J"
        else: pos["days_zero_score"] = 0

        if sell:
            proceeds = (price_eur * shares) - usd_to_eur(COST_PER_TRADE, eur_rate)
            projected_cash += proceeds
            signals["sell"].append({
                "ticker": ticker, "shares": shares, "value_eur": price_eur*shares,
                "price_usd": price_usd, "price_eur": price_eur, "reason": reason,
                "pnl_eur": pnl_eur, "pnl_pct": pnl_pct
            })
            positions_to_remove.append(ticker)

    # ============================================================
    # 2. OPPORTUNITÉS D'ACHAT (ARBITRAGE + RENFORCEMENT)
    # ============================================================
    
    # On regarde les positions qui vont rester
    future_positions = [t for t in portfolio["positions"] if t not in positions_to_remove]
    slots_available = max_positions - len(future_positions)
    
    print(f"\n{'='*70}")
    print(f"💵 CASH ACTUEL: {portfolio['cash']:.2f}€")
    print(f"🔮 CASH PROJETÉ: {projected_cash:.2f}€ (Après ventes)")
    print(f"📦 SLOTS DISPO: {slots_available}")
    print(f"{'='*70}")

    if projected_cash > 50:
        for ticker in valid_scores.index:
            if projected_cash < 50: break
            
            # Est-ce un renforcement d'une position existante ?
            is_reinforce = (ticker in future_positions)
            
            # Si c'est un nouvel achat, on a besoin d'un slot libre
            if not is_reinforce and slots_available <= 0: continue
            
            rank = list(valid_scores.index).index(ticker) + 1
            if rank > max_positions: continue # Focus Top
            
            # Allocation Cible
            target_alloc = get_weighted_allocation(rank, max_positions, total_pf_value)
            
            # Combien on a déjà ?
            current_invested = 0
            if is_reinforce:
                current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
                current_invested = portfolio["positions"][ticker]["shares"] * current_price_eur
            
            # Besoin
            amount_needed = target_alloc - current_invested
            
            # On investit seulement si le besoin est significatif (>50€)
            if amount_needed > 50:
                amount_to_invest = min(amount_needed, projected_cash - 10)
                
                if amount_to_invest > 50:
                    price_usd = float(current_prices[ticker])
                    price_eur = usd_to_eur(price_usd, eur_rate)
                    shares = amount_to_invest / price_eur
                    
                    action_type = "REINFORCE" if is_reinforce else "BUY"
                    
                    print(f"🟢 {action_type} #{rank} {ticker}")
                    print(f"   Score: {valid_scores[ticker]:.3f}")
                    print(f"   Montant: {amount_to_invest:.2f}€")
                    
                    signals["buy"].append({
                        "ticker": ticker, "action": action_type,
                        "amount_eur": amount_to_invest, "shares": shares,
                        "price_usd": price_usd, "price_eur": price_eur,
                        "score": valid_scores[ticker], "rank": rank,
                        "stop_loss_eur": calculate_stop_price(price_eur, get_stop_loss_pct(ticker, defensive))
                    })
                    
                    projected_cash -= amount_to_invest
                    if not is_reinforce: slots_available -= 1

    # ============================================================
    # 3. EXÉCUTION
    # ============================================================
    print(f"\n{'='*70}\n⚡ EXÉCUTION\n{'='*70}")
    
    # Ventes
    for s in signals["sell"]:
        proceeds = s["value_eur"] - usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] += proceeds
        del portfolio["positions"][s["ticker"]]
        log_trade(history, "SELL", s["ticker"], s["price_usd"], s["price_eur"], s["shares"], s["value_eur"], eur_rate, s["reason"], s["pnl_eur"], s["pnl_pct"])
        print(f"✅ VENDU {s['ticker']} (+{proceeds:.2f}€)")

    # Achats / Renforcements
    for b in signals["buy"]:
        cost = b["amount_eur"] + usd_to_eur(COST_PER_TRADE, eur_rate)
        
        # Vérif ultime cash
        if portfolio["cash"] >= cost:
            portfolio["cash"] -= cost
            
            if b["ticker"] in portfolio["positions"]:
                # CALCUL DU PRU (PRIX MOYEN PONDÉRÉ)
                old_pos = portfolio["positions"][b["ticker"]]
                total_shares = old_pos["shares"] + b["shares"]
                # Valeur comptable totale
                total_cost_eur = (old_pos["entry_price_eur"] * old_pos["shares"]) + b["amount_eur"]
                avg_price_eur = total_cost_eur / total_shares
                
                # Mise à jour
                old_pos["entry_price_eur"] = avg_price_eur
                old_pos["shares"] = total_shares
                old_pos["amount_invested_eur"] += b["amount_eur"]
                # On garde l'ancien peak s'il est plus haut, sinon on prend le nouveau prix
                old_pos["peak_price_eur"] = max(old_pos.get("peak_price_eur", 0), b["price_eur"])
                
                log_trade(history, "REINFORCE", b["ticker"], b["price_usd"], b["price_eur"], b["shares"], b["amount_eur"], eur_rate, f"Rank #{b['rank']}")
                print(f"✅ RENFORCÉ {b['ticker']} (Nouveau PRU: {avg_price_eur:.2f}€)")
            else:
                # NOUVEAU
                portfolio["positions"][b["ticker"]] = {
                    "entry_price_eur": b["price_eur"], "entry_price_usd": b["price_usd"],
                    "entry_date": today, "shares": b["shares"],
                    "amount_invested_eur": b["amount_eur"], "initial_amount_eur": b["amount_eur"],
                    "score": b["score"], "peak_price_eur": b["price_eur"],
                    "stop_loss_eur": b["stop_loss_eur"], "rank": b["rank"], "days_zero_score": 0
                }
                log_trade(history, "BUY", b["ticker"], b["price_usd"], b["price_eur"], b["shares"], b["amount_eur"], eur_rate, f"Rank #{b['rank']}")
                print(f"✅ ACHETÉ {b['ticker']}")
        else:
            print(f"⚠️ FONDS INSUFFISANTS POUR {b['ticker']} (Manque {cost - portfolio['cash']:.2f}€)")

    # ============================================================
    # 4. RÉSUMÉ & TELEGRAM
    # ============================================================
    print(f"\n{'='*70}\n📊 RÉSUMÉ FINAL\n{'='*70}")
    
    total_val = portfolio["cash"] + sum(usd_to_eur(float(current_prices[t]), eur_rate)*p["shares"] for t,p in portfolio["positions"].items() if t in current_prices)
    
    # Save
    save_portfolio(portfolio)
    save_trades_history(history)
    print(f"💾 Sauvegardé. Valeur Totale: {total_val:.2f}€")

    # Top 5 Momentum (Ta fonctionnalité demandée)
    print(f"\n🏆 TOP 5 MOMENTUM")
    for i, ticker in enumerate(valid_scores.head(5).index, 1):
        price = usd_to_eur(float(current_prices[ticker]), eur_rate)
        in_pf = "📂" if ticker in portfolio["positions"] else "👀"
        print(f"{i}. {ticker} ({valid_scores[ticker]:.3f}) {in_pf}")

    # Telegram Msg
    msg = f"📊 <b>APEX v31.4 HYBRID</b>\nVal: {total_val:.0f}€ | Cash: {portfolio['cash']:.0f}€\n\n"
    if signals['sell']: msg += "🔴 <b>VENTES:</b>\n" + "\n".join([f"- {s['ticker']} ({s['reason']})" for s in signals['sell']]) + "\n\n"
    if signals['buy']: msg += "🟢 <b>MOUVEMENTS:</b>\n" + "\n".join([f"- {b['action']} {b['ticker']} ({b['amount_eur']:.0f}€)" for b in signals['buy']]) + "\n\n"
    
    msg += "🏆 <b>TOP 3:</b>\n"
    for i, ticker in enumerate(valid_scores.head(3).index, 1):
        msg += f"{i}. {ticker} ({valid_scores[ticker]:.3f})\n"

    if not signals['sell'] and not signals['buy']: msg += "\n😴 Aucun mouvement."
    
    send_telegram(msg)

if __name__ == "__main__":
    main()
