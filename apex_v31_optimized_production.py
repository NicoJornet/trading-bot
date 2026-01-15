"""
APEX v31.2 - CASH OPTIMISÉ & ARBITRAGE IMMÉDIAT
================================================
Modifications:
1. Cash Flow Unifié: Les achats utilisent (Cash Actuel + Cash des Ventes prévues).
2. Fix: Correction du bug 'trades_history' si le fichier est vide.
3. Arbitrage: Le capital libéré est immédiatement réinvesti sur le Top Rank.
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

# Paramètres de sortie
HARD_STOP_PCT = 0.18
MFE_THRESHOLD_PCT = 0.15
TRAILING_PCT = 0.05

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
# FONCTIONS UTILITAIRES
# ============================================================

def get_stop_loss_pct(ticker, defensive=False):
    base_stop = HARD_STOP_PCT
    return base_stop * 0.85 if defensive else base_stop

def calculate_stop_price(entry_price, stop_pct):
    return entry_price * (1 - stop_pct)

def check_mfe_trailing_exit(pos, current_price, entry_price):
    peak_price = pos.get('peak_price_eur', entry_price)
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

def eur_to_usd(amount_eur, rate=None):
    if rate is None: rate = get_eur_usd_rate()
    return amount_eur * rate

# ============================================================
# GESTION FICHIERS ROBUSTE
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
    if action == "BUY": history["summary"]["buys"] += 1
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
# CORE LOGIC
# ============================================================

def get_market_data(tickers, days=100):
    end = datetime.now()
    start = end - timedelta(days=days)
    try: return yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    except: return None

def calculate_momentum_score(close, high, atr_period=14, sma_period=20, high_lookback=60):
    if len(close) < max(atr_period, sma_period, high_lookback): return np.nan
    sma = close.rolling(sma_period).mean()
    tr = (high - close.shift(1)).abs()
    atr = tr.rolling(atr_period).mean()
    high_60 = high.rolling(high_lookback).max()
    score = ((close - sma) / atr) / (high_60 / close)
    return score.iloc[-1]

def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        return vix.info.get('regularMarketPrice') or vix.info.get('previousClose') or 20
    except: return 20

def main():
    print("=" * 70)
    print("🚀 APEX v31.2 - CASH OPTIMISÉ")
    print("=" * 70)
    
    portfolio = load_portfolio()
    history = load_trades_history()
    eur_rate = get_eur_usd_rate()
    current_vix = get_vix()
    
    # Gestion Régime
    if current_vix >= VIX_ULTRA_DEFENSIVE: max_positions = MAX_POSITIONS_ULTRA_DEFENSIVE; regime="🔴 ULTRA-DEF"
    elif current_vix >= VIX_DEFENSIVE: max_positions = MAX_POSITIONS_DEFENSIVE; regime="🟡 DEF"
    else: max_positions = MAX_POSITIONS_NORMAL; regime="🟢 NORMAL"
    defensive = current_vix >= VIX_DEFENSIVE
    
    print(f"📊 VIX: {current_vix:.1f} | Régime: {regime} (Max {max_positions} slots)")
    
    # DCA
    today = datetime.now().strftime("%Y-%m-%d")
    if not portfolio.get("last_dca_date", "").startswith(datetime.now().strftime("%Y-%m")):
        portfolio["cash"] += MONTHLY_DCA
        portfolio["last_dca_date"] = today
        print(f"💰 DCA ajouté: +{MONTHLY_DCA}€")

    # Data
    print("📥 Téléchargement données...")
    data = get_market_data(DATABASE)
    if data is None or data.empty: return

    # Scoring
    scores, current_prices = {}, {}
    for ticker in DATABASE:
        try:
            # Gestion structure yfinance
            if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                df = data[ticker]
            elif ticker == data.columns[0]: # Cas ticker unique
                df = data
            else: continue
                
            close, high = df['Close'].dropna(), df['High'].dropna()
            if len(close) > 0:
                current_prices[ticker] = close.iloc[-1]
                s = calculate_momentum_score(close, high)
                if s > 0: scores[ticker] = s
        except: continue
    
    valid_scores = pd.Series(scores).sort_values(ascending=False)
    print(f"📊 {len(valid_scores)} tickers éligibles")

    # --------------------------------------------------------
    # 1. ANALYSE DES VENTES (Et calcul du cash prévisionnel)
    # --------------------------------------------------------
    print(f"\n{'='*30}\n📂 ANALYSE POSITIONS\n{'='*30}")
    
    signals = {"sell": [], "buy": [], "rotation": []}
    positions_to_remove = []
    
    # ⭐ CLÉ DU FIX: On projette le cash disponible APRES les ventes
    projected_cash = portfolio["cash"] 
    
    for ticker, pos in portfolio["positions"].items():
        if ticker not in current_prices: continue
        
        # Données
        price_usd = float(current_prices[ticker])
        price_eur = usd_to_eur(price_usd, eur_rate)
        shares = pos["shares"]
        
        # Update Peak
        if price_eur > pos.get('peak_price_eur', 0): pos['peak_price_eur'] = price_eur
        
        # Calculs Stops
        stop_pct = get_stop_loss_pct(ticker, defensive)
        stop_price = calculate_stop_price(pos["entry_price_eur"], stop_pct)
        pos['stop_loss_eur'] = stop_price
        
        # Check Exits
        sell, reason = False, ""
        
        # Hard Stop
        hit_stop, r = check_hard_stop_exit(price_eur, pos["entry_price_eur"], stop_price)
        if hit_stop: sell, reason = True, r
        
        # Trailing
        if not sell:
            hit_trail, r, det = check_mfe_trailing_exit(pos, price_eur, pos["entry_price_eur"])
            if hit_trail: sell, reason = True, r
            else: print(f"🔹 {ticker}: Trailing {'ACTIF' if det['trailing_active'] else 'ATTENTE'} (MFE +{det['mfe_pct']:.1f}%)")
        
        # Rotation Forcée
        curr_score = valid_scores.get(ticker, 0)
        pos['score'] = curr_score
        if not sell and curr_score <= 0:
            pos["days_zero_score"] = pos.get("days_zero_score", 0) + 1
            if pos["days_zero_score"] >= FORCE_ROTATION_DAYS:
                sell, reason = True, f"ROTATION_{pos['days_zero_score']}J"
        else: pos["days_zero_score"] = 0

        # Logique de Vente
        if sell:
            print(f"❌ VENTE {ticker}: {reason}")
            # Estimation du cash récupéré (moins frais)
            proceeds = (price_eur * shares) - usd_to_eur(COST_PER_TRADE, eur_rate)
            projected_cash += proceeds
            
            signals["sell"].append({
                "ticker": ticker, "shares": shares, "price_usd": price_usd, "price_eur": price_eur,
                "value_eur": price_eur * shares, "reason": reason,
                "pnl_eur": (price_eur - pos["entry_price_eur"]) * shares,
                "pnl_pct": (price_eur / pos["entry_price_eur"] - 1) * 100
            })
            positions_to_remove.append(ticker)

    # --------------------------------------------------------
    # 2. OPPORTUNITÉS D'ACHAT (Avec Cash Projeté)
    # --------------------------------------------------------
    future_pos_count = len(portfolio["positions"]) - len(positions_to_remove)
    slots = max_positions - future_pos_count
    
    print(f"\n💵 Cash Actuel: {portfolio['cash']:.2f}€")
    print(f"🔮 Cash Projeté (après ventes): {projected_cash:.2f}€")
    
    if slots > 0 and projected_cash > 50:
        print(f"\n{'='*30}\n🛒 SHOPPING (Slots: {slots})\n{'='*30}")
        
        for ticker in valid_scores.index:
            if slots <= 0 or projected_cash < 50: break
            if ticker in portfolio["positions"] and ticker not in positions_to_remove: continue
            
            # Rang
            rank = list(valid_scores.index).index(ticker) + 1
            if rank > max_positions: continue # On reste concentré sur le Top
            
            # Allocation sur le cash PROJETÉ
            alloc = get_weighted_allocation(rank, max_positions, projected_cash)
            alloc = min(alloc, projected_cash - 10) # Buffer securité
            
            if alloc > 50:
                price_usd = float(current_prices[ticker])
                price_eur = usd_to_eur(price_usd, eur_rate)
                shares = alloc / price_eur
                
                print(f"🟢 ACHAT {ticker} (#{rank}) -> {alloc:.2f}€ ({shares:.4f} shares)")
                
                signals["buy"].append({
                    "ticker": ticker, "rank": rank, "score": valid_scores[ticker],
                    "price_usd": price_usd, "price_eur": price_eur,
                    "shares": shares, "amount_eur": alloc,
                    "stop_loss_eur": calculate_stop_price(price_eur, get_stop_loss_pct(ticker, defensive))
                })
                
                projected_cash -= alloc
                slots -= 1

    # --------------------------------------------------------
    # 3. EXÉCUTION
    # --------------------------------------------------------
    print(f"\n{'='*30}\n⚡ EXÉCUTION\n{'='*30}")
    
    # Exec Ventes
    for s in signals["sell"]:
        proceeds = s["value_eur"] - usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] += proceeds
        del portfolio["positions"][s["ticker"]]
        log_trade(history, "SELL", s["ticker"], s["price_usd"], s["price_eur"], s["shares"], s["value_eur"], eur_rate, s["reason"], s["pnl_eur"], s["pnl_pct"])
        print(f"✅ VENDU {s['ticker']} (+{proceeds:.2f}€ cash)")

    # Exec Achats
    for b in signals["buy"]:
        cost = b["amount_eur"] + usd_to_eur(COST_PER_TRADE, eur_rate)
        # Vérification ultime (au cas où estimation projected_cash était off)
        if portfolio["cash"] >= cost:
            portfolio["cash"] -= cost
            portfolio["positions"][b["ticker"]] = {
                "entry_price_eur": b["price_eur"], "entry_price_usd": b["price_usd"],
                "entry_date": today, "shares": b["shares"],
                "amount_invested_eur": b["amount_eur"], "initial_amount_eur": b["amount_eur"],
                "score": b["score"], "peak_price_eur": b["price_eur"],
                "stop_loss_eur": b["stop_loss_eur"], "rank": b["rank"],
                "days_zero_score": 0
            }
            log_trade(history, "BUY", b["ticker"], b["price_usd"], b["price_eur"], b["shares"], b["amount_eur"], eur_rate, f"Rank #{b['rank']}")
            print(f"✅ ACHETÉ {b['ticker']} (-{cost:.2f}€ cash)")
        else:
            print(f"⚠️ FONDS INSUFFISANTS POUR {b['ticker']} (Manque {cost - portfolio['cash']:.2f}€)")

    # --------------------------------------------------------
    # 4. FINALISATION
    # --------------------------------------------------------
    total_val = portfolio["cash"] + sum(usd_to_eur(float(current_prices[t]), eur_rate)*p["shares"] for t,p in portfolio["positions"].items() if t in current_prices)
    
    save_portfolio(portfolio)
    save_trades_history(history)
    
    # Telegram
    msg = f"📊 <b>APEX v31.2</b>\nVal: {total_val:.0f}€ | Cash: {portfolio['cash']:.0f}€\n\n"
    if signals['sell']: msg += "🔴 VENTES:\n" + "\n".join([f"- {s['ticker']} ({s['reason']})" for s in signals['sell']]) + "\n\n"
    if signals['buy']: msg += "🟢 ACHATS:\n" + "\n".join([f"- {b['ticker']} ({b['amount_eur']:.0f}€)" for b in signals['buy']])
    if not signals['sell'] and not signals['buy']: msg += "😴 Aucun mouvement."
    send_telegram(msg)
    
    print("\n🏁 Terminé.")

if __name__ == "__main__":
    main()
