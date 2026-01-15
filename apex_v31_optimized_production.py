"""
APEX v31.4 - HYBRID (PRODUCTION + SMART CASH + SOFT MOMENTUM TRIM)
==================================================================
Base: APEX v31 Optimisé (Affichage riche, Momentum info)
Moteur: v31.3 (Arbitrage immédiat + Renforcement positions existantes)

+ AJOUT (Soft Momentum Rebalance)
- Utilise le cash des ventes AVANT exécution pour planifier les achats (projected_cash)
- TRIM (vente partielle) uniquement sur la position #1 si surpondérée
  Preset soft momentum:
    TARGET_TOLERANCE = 0.12 (12%)
    MIN_TRADE_EUR = 150
    TRIM uniquement #1
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
# SOFT MOMENTUM TRIM (rebalance partiel)
# ============================================================
REBALANCE_TRIM = True
TRIM_ONLY_RANK1 = True
TARGET_TOLERANCE = 0.12   # 12% de tolérance (ex: cible 50% => trim si > 56%)
MIN_TRADE_EUR = 150       # trim uniquement si l'excès dépasse 150€
CASH_BUFFER_EUR = 10      # garde un petit buffer de cash

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

def get_target_weight(rank, num_positions):
    """Poids cible (en %) selon ranking"""
    if num_positions == 1:
        return 1.00
    if num_positions == 2:
        return 0.60 if rank == 1 else 0.40
    if num_positions == 3:
        if rank == 1: return 0.50
        if rank == 2: return 0.30
        return 0.20
    # fallback dégressif
    total = sum(range(1, num_positions + 1))
    return (num_positions - rank + 1) / total

def get_eur_usd_rate():
    try:
        eur_usd = yf.Ticker("EURUSD=X")
        rate = eur_usd.info.get('regularMarketPrice') or eur_usd.info.get('previousClose')
        return rate if rate and rate > 0 else 1.08
    except:
        return 1.08

def usd_to_eur(amount_usd, rate=None):
    if rate is None:
        rate = get_eur_usd_rate()
    return amount_usd / rate

# ============================================================
# GESTION FICHIERS (ROBUSTE)
# ============================================================

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            try:
                data = json.load(f)
                # sécurité sur champs potentiellement None
                if data.get("cash") is None:
                    data["cash"] = INITIAL_CAPITAL
                if data.get("positions") is None:
                    data["positions"] = {}
                return data
            except:
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
    """Charge l'historique avec structure sécurisée"""
    default = {"trades": [], "summary": {"total_trades": 0, "buys": 0, "sells": 0, "total_pnl_eur": 0.0}}
    if os.path.exists(TRADES_HISTORY_FILE):
        try:
            with open(TRADES_HISTORY_FILE, "r") as f:
                data = json.load(f)
                if "trades" not in data or data["trades"] is None:
                    data["trades"] = []
                if "summary" not in data or data["summary"] is None:
                    data["summary"] = default["summary"]
                return data
        except:
            pass
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
        "action": action,
        "ticker": ticker,
        "shares": round(float(shares), 6),
        "price_usd": round(float(price_usd), 2),
        "price_eur": round(float(price_eur), 2),
        "amount_eur": round(float(amount_eur), 2),
        "fee_eur": float(COST_PER_TRADE),
        "eur_usd_rate": round(float(eur_rate), 4),
        "reason": reason
    }
    if pnl_eur is not None:
        trade["pnl_eur"], trade["pnl_pct"] = round(float(pnl_eur), 2), round(float(pnl_pct), 2)

    history["trades"].append(trade)
    history["summary"]["total_trades"] += 1
    if action in ["BUY", "REINFORCE"]:
        history["summary"]["buys"] += 1
    elif action == "SELL":
        history["summary"]["sells"] += 1
        if pnl_eur:
            history["summary"]["total_pnl_eur"] += float(pnl_eur)

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

# ============================================================
# MARKET DATA & SCORING
# ============================================================

def get_market_data(tickers, days=100):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        return yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    except:
        return None

def calculate_momentum_score(close, high):
    """Calcul du score APEX (tel que dans ton code)"""
    if len(close) < max(ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK):
        return np.nan
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
    except:
        return 20

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 APEX v31.4 - HYBRID (INFO + SMART CASH + SOFT MOMENTUM TRIM)")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load
    portfolio = load_portfolio()
    history = load_trades_history()
    eur_rate = get_eur_usd_rate()
    current_vix = get_vix()

    # Regime
    if current_vix >= VIX_ULTRA_DEFENSIVE:
        max_positions = MAX_POSITIONS_ULTRA_DEFENSIVE
        regime = "🔴 ULTRA-DEF"
    elif current_vix >= VIX_DEFENSIVE:
        max_positions = MAX_POSITIONS_DEFENSIVE
        regime = "🟡 DEF"
    else:
        max_positions = MAX_POSITIONS_NORMAL
        regime = "🟢 NORMAL"
    defensive = current_vix >= VIX_DEFENSIVE

    print(f"💱 EUR/USD: {eur_rate:.4f}")
    print(f"📊 VIX: {current_vix:.1f} | Régime: {regime}")

    # =========================
    # DCA (FIX BUG NoneType)
    # =========================
    today = datetime.now().strftime("%Y-%m-%d")
    last_dca = portfolio.get("last_dca_date") or ""   # ✅ FIX: None -> ""
    if not str(last_dca).startswith(datetime.now().strftime("%Y-%m")):
        portfolio["cash"] = float(portfolio.get("cash") or 0.0) + float(MONTHLY_DCA)
        portfolio["last_dca_date"] = today
        print(f"💰 DCA ajouté: +{MONTHLY_DCA}€")

    # Data
    print("\n📥 Téléchargement données...")
    data = get_market_data(DATABASE)
    if data is None or data.empty:
        print("⚠️ Données indisponibles.")
        return

    # Scoring
    scores, current_prices = {}, {}
    for ticker in DATABASE:
        try:
            if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                df = data[ticker]
            elif not isinstance(data.columns, pd.MultiIndex):
                df = data
            else:
                continue

            close, high = df['Close'].dropna(), df['High'].dropna()
            if len(close) > 0:
                current_prices[ticker] = close.iloc[-1]
                s = calculate_momentum_score(close, high)
                if s > 0:
                    scores[ticker] = s
        except:
            continue

    valid_scores = pd.Series(scores).sort_values(ascending=False)
    print(f"📊 {len(valid_scores)} tickers avec score > 0")

    # ============================================================
    # 1) ANALYSE POSITIONS & PROJECTION CASH (cash + ventes prévues)
    # ============================================================
    print(f"\n{'='*70}\n📂 VÉRIFICATION DES POSITIONS\n{'='*70}")

    signals = {"sell": [], "trim": [], "buy": []}
    positions_to_remove = []

    projected_cash = float(portfolio.get("cash") or 0.0)

    # Valeur totale estimée
    total_pf_value = projected_cash + sum(
        usd_to_eur(float(current_prices[t]), eur_rate) * float(p.get("shares") or 0.0)
        for t, p in (portfolio.get("positions") or {}).items()
        if t in current_prices
    )

    for ticker, pos in list((portfolio.get("positions") or {}).items()):
        if ticker not in current_prices:
            continue

        # Données
        price_usd = float(current_prices[ticker])
        price_eur = usd_to_eur(price_usd, eur_rate)
        shares = float(pos.get("shares") or 0.0)

        if shares <= 0:
            continue

        # Update Peak
        if price_eur > float(pos.get('peak_price_eur') or 0.0):
            pos['peak_price_eur'] = price_eur

        # Stops
        entry_eur = float(pos.get("entry_price_eur") or price_eur)
        stop_pct = get_stop_loss_pct(ticker, defensive)
        stop_price = calculate_stop_price(entry_eur, stop_pct)
        pos['stop_loss_eur'] = stop_price

        # Infos affichage
        pnl_eur = (price_eur - entry_eur) * shares
        pnl_pct = (price_eur / entry_eur - 1) * 100 if entry_eur > 0 else 0.0
        curr_score = float(valid_scores.get(ticker, 0))
        pos['score'] = curr_score
        rank_display = f"#{list(valid_scores.index).index(ticker) + 1}" if ticker in valid_scores.index else "N/A"

        print(f"\n🔹 {ticker}")
        print(f"   Prix: {price_eur:.2f}€ (Entrée: {entry_eur:.2f}€)")
        print(f"   PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f"   Score: {curr_score:.3f} | Rank: {rank_display}")

        # Logic Vente
        sell, reason = False, ""

        hit_stop, r = check_hard_stop_exit(price_eur, entry_eur, stop_price)
        if hit_stop:
            sell, reason = True, r
            print(f"   ❌ HARD STOP touché ({stop_price:.2f}€)")

        if not sell:
            hit_trail, r, det = check_mfe_trailing_exit(pos, price_eur, entry_eur)
            if hit_trail:
                sell, reason = True, r
                print(f"   📉 MFE TRAILING déclenché (MFE +{det['mfe_pct']:.1f}%)")
            else:
                status = "ACTIF" if det['trailing_active'] else "INACTIF"
                print(f"   🎯 Trailing: {status} (MFE: +{det['mfe_pct']:.1f}%)")

        # Rotation Forcée
        if not sell and curr_score <= 0:
            pos["days_zero_score"] = int(pos.get("days_zero_score") or 0) + 1
            print(f"   ⚠️ Score ≤ 0 depuis {pos['days_zero_score']} jours")
            if pos["days_zero_score"] >= FORCE_ROTATION_DAYS:
                sell, reason = True, f"ROTATION_{pos['days_zero_score']}J"
        else:
            pos["days_zero_score"] = 0

        if sell:
            proceeds = (price_eur * shares) - usd_to_eur(COST_PER_TRADE, eur_rate)
            projected_cash += proceeds
            signals["sell"].append({
                "ticker": ticker, "shares": shares, "value_eur": price_eur * shares,
                "price_usd": price_usd, "price_eur": price_eur, "reason": reason,
                "pnl_eur": pnl_eur, "pnl_pct": pnl_pct
            })
            positions_to_remove.append(ticker)

    # ============================================================
    # 2) SOFT TRIM (#1 uniquement) + OPPORTUNITÉS D'ACHAT
    # ============================================================

    positions_dict = portfolio.get("positions") or {}
    future_positions = [t for t in positions_dict if t not in positions_to_remove]
    slots_available = max_positions - len(future_positions)

    print(f"\n{'='*70}")
    print(f"💵 CASH ACTUEL: {float(portfolio.get('cash') or 0.0):.2f}€")
    print(f"🔮 CASH PROJETÉ (après SELLS): {projected_cash:.2f}€")
    print(f"📦 SLOTS DISPO: {slots_available}")
    print(f"{'='*70}")

    future_total_value = projected_cash + sum(
        usd_to_eur(float(current_prices[t]), eur_rate) * float(positions_dict[t].get("shares") or 0.0)
        for t in future_positions if t in current_prices
    )

    # --- TRIM: uniquement sur le rang #1 ---
    if REBALANCE_TRIM and len(valid_scores) > 0 and len(future_positions) > 0:
        target_universe = list(valid_scores.head(max_positions).index)
        rank1_ticker = target_universe[0] if len(target_universe) > 0 else None

        if rank1_ticker and (rank1_ticker in future_positions) and (rank1_ticker in current_prices):
            w = get_target_weight(1, max_positions)
            target_value = w * future_total_value

            price_eur = usd_to_eur(float(current_prices[rank1_ticker]), eur_rate)
            pos = positions_dict[rank1_ticker]
            current_value = float(pos.get("shares") or 0.0) * price_eur

            if current_value > target_value * (1 + TARGET_TOLERANCE):
                excess = current_value - target_value
                if excess >= MIN_TRADE_EUR:
                    shares_to_sell = excess / price_eur
                    shares_to_sell = min(shares_to_sell, float(pos.get("shares") or 0.0))

                    value_eur = shares_to_sell * price_eur
                    proceeds = value_eur - usd_to_eur(COST_PER_TRADE, eur_rate)

                    signals["trim"].append({
                        "ticker": rank1_ticker,
                        "shares": shares_to_sell,
                        "price_usd": float(current_prices[rank1_ticker]),
                        "price_eur": price_eur,
                        "value_eur": value_eur,
                        "reason": f"TRIM_RANK1_TO_{int(w*100)}%_TOL{int(TARGET_TOLERANCE*100)}"
                    })

                    projected_cash += proceeds

                    print(f"\n✂️ TRIM PLANIFIÉ (#1) {rank1_ticker}")
                    print(f"   Valeur actuelle: {current_value:.2f}€")
                    print(f"   Cible: {target_value:.2f}€ (poids {int(w*100)}%)")
                    print(f"   Excès: {excess:.2f}€ | Vente: {value_eur:.2f}€ (min {MIN_TRADE_EUR}€)")
                    print(f"   Cash projeté (après TRIM): {projected_cash:.2f}€")

    # total_pf_value après trims planifiés
    total_pf_value = projected_cash + sum(
        usd_to_eur(float(current_prices[t]), eur_rate) * float(positions_dict[t].get("shares") or 0.0)
        for t in future_positions if t in current_prices
    )

    slots_available = max_positions - len(future_positions)

    print(f"\n{'='*70}")
    print(f"🔮 CASH PROJETÉ (après SELLS + TRIM): {projected_cash:.2f}€")
    print(f"📦 SLOTS DISPO: {slots_available}")
    print(f"{'='*70}")

    # --- ACHATS / RENFORCEMENTS ---
    if projected_cash > 50 and len(valid_scores) > 0:
        for ticker in valid_scores.index:
            if projected_cash < (50 + CASH_BUFFER_EUR):
                break

            is_reinforce = (ticker in future_positions)

            if not is_reinforce and slots_available <= 0:
                continue

            rank = list(valid_scores.index).index(ticker) + 1
            if rank > max_positions:
                continue

            target_alloc = get_weighted_allocation(rank, max_positions, total_pf_value)

            current_invested = 0.0
            if is_reinforce and ticker in current_prices:
                current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
                current_invested = float(positions_dict[ticker].get("shares") or 0.0) * current_price_eur

            amount_needed = target_alloc - current_invested

            if amount_needed > 50:
                amount_to_invest = min(amount_needed, projected_cash - CASH_BUFFER_EUR)

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
                        "score": float(valid_scores[ticker]), "rank": rank,
                        "stop_loss_eur": calculate_stop_price(price_eur, get_stop_loss_pct(ticker, defensive))
                    })

                    projected_cash -= amount_to_invest
                    if not is_reinforce:
                        slots_available -= 1

    # ============================================================
    # 3) EXÉCUTION (SELLS -> TRIMS -> BUYS)
    # ============================================================
    print(f"\n{'='*70}\n⚡ EXÉCUTION\n{'='*70}")

    # SELLS complets
    for s in signals["sell"]:
        proceeds = float(s["value_eur"]) - usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] = float(portfolio.get("cash") or 0.0) + proceeds
        if s["ticker"] in positions_dict:
            del positions_dict[s["ticker"]]
        log_trade(
            history, "SELL", s["ticker"], s["price_usd"], s["price_eur"],
            s["shares"], s["value_eur"], eur_rate, s["reason"], s["pnl_eur"], s["pnl_pct"]
        )
        print(f"✅ VENDU {s['ticker']} (+{proceeds:.2f}€)")

    # TRIMS partiels
    for s in signals.get("trim", []):
        ticker = s["ticker"]
        if ticker not in positions_dict:
            continue
        pos = positions_dict[ticker]

        shares_to_sell = float(s["shares"])
        if shares_to_sell <= 0:
            continue

        price_eur = float(s["price_eur"])
        price_usd = float(s["price_usd"])
        value_eur = float(s["value_eur"])

        proceeds = value_eur - usd_to_eur(COST_PER_TRADE, eur_rate)
        portfolio["cash"] = float(portfolio.get("cash") or 0.0) + proceeds

        entry = float(pos.get("entry_price_eur") or price_eur)
        pnl_eur = (price_eur - entry) * shares_to_sell
        pnl_pct = (price_eur / entry - 1) * 100 if entry > 0 else 0.0

        pos["shares"] = float(pos.get("shares") or 0.0) - shares_to_sell

        if "amount_invested_eur" in pos and pos.get("amount_invested_eur") is not None:
            pos["amount_invested_eur"] = max(0.0, float(pos["amount_invested_eur"]) - entry * shares_to_sell)

        if float(pos.get("shares") or 0.0) <= 1e-8:
            del positions_dict[ticker]

        log_trade(
            history, "SELL", ticker, price_usd, price_eur, shares_to_sell, value_eur,
            eur_rate, s["reason"], pnl_eur, pnl_pct
        )

        print(f"✂️ TRIM {ticker} (-{shares_to_sell:.6f} sh) (+{proceeds:.2f}€)")

    # BUYS / RENFORCE
    for b in signals["buy"]:
        cost = float(b["amount_eur"]) + usd_to_eur(COST_PER_TRADE, eur_rate)

        if float(portfolio.get("cash") or 0.0) >= cost:
            portfolio["cash"] = float(portfolio.get("cash") or 0.0) - cost

            if b["ticker"] in positions_dict:
                old_pos = positions_dict[b["ticker"]]
                old_shares = float(old_pos.get("shares") or 0.0)
                new_shares = float(b["shares"])
                total_shares = old_shares + new_shares

                total_cost_eur = (float(old_pos.get("entry_price_eur") or 0.0) * old_shares) + float(b["amount_eur"])
                avg_price_eur = total_cost_eur / total_shares if total_shares > 0 else float(b["price_eur"])

                old_pos["entry_price_eur"] = avg_price_eur
                old_pos["shares"] = total_shares
                old_pos["amount_invested_eur"] = float(old_pos.get("amount_invested_eur") or 0.0) + float(b["amount_eur"])
                old_pos["peak_price_eur"] = max(float(old_pos.get("peak_price_eur") or 0.0), float(b["price_eur"]))

                log_trade(
                    history, "REINFORCE", b["ticker"], b["price_usd"], b["price_eur"],
                    b["shares"], b["amount_eur"], eur_rate, f"Rank #{b['rank']}"
                )
                print(f"✅ RENFORCÉ {b['ticker']} (Nouveau PRU: {avg_price_eur:.2f}€)")
            else:
                positions_dict[b["ticker"]] = {
                    "entry_price_eur": float(b["price_eur"]),
                    "entry_price_usd": float(b["price_usd"]),
                    "entry_date": today,
                    "shares": float(b["shares"]),
                    "amount_invested_eur": float(b["amount_eur"]),
                    "initial_amount_eur": float(b["amount_eur"]),
                    "score": float(b["score"]),
                    "peak_price_eur": float(b["price_eur"]),
                    "stop_loss_eur": float(b["stop_loss_eur"]),
                    "rank": int(b["rank"]),
                    "days_zero_score": 0
                }
                log_trade(
                    history, "BUY", b["ticker"], b["price_usd"], b["price_eur"],
                    b["shares"], b["amount_eur"], eur_rate, f"Rank #{b['rank']}"
                )
                print(f"✅ ACHETÉ {b['ticker']}")
        else:
            print(f"⚠️ FONDS INSUFFISANTS POUR {b['ticker']} (Manque {cost - float(portfolio.get('cash') or 0.0):.2f}€)")

    # ============================================================
    # 4) RÉSUMÉ & TELEGRAM
    # ============================================================
    print(f"\n{'='*70}\n📊 RÉSUMÉ FINAL\n{'='*70}")

    portfolio["positions"] = positions_dict  # ensure saved
    total_val = float(portfolio.get("cash") or 0.0) + sum(
        usd_to_eur(float(current_prices[t]), eur_rate) * float(p.get("shares") or 0.0)
        for t, p in positions_dict.items()
        if t in current_prices
    )

    save_portfolio(portfolio)
    save_trades_history(history)
    print(f"💾 Sauvegardé. Valeur Totale: {total_val:.2f}€")

    print(f"\n🏆 TOP 5 MOMENTUM")
    for i, ticker in enumerate(valid_scores.head(5).index, 1):
        price = usd_to_eur(float(current_prices[ticker]), eur_rate)
        in_pf = "📂" if ticker in positions_dict else "👀"
        print(f"{i}. {ticker} ({valid_scores[ticker]:.3f}) {in_pf} | {price:.2f}€")

    msg = f"📊 <b>APEX v31.4 HYBRID</b>\nVal: {total_val:.0f}€ | Cash: {float(portfolio.get('cash') or 0.0):.0f}€\n\n"
    if signals['sell']:
        msg += "🔴 <b>VENTES:</b>\n" + "\n".join([f"- {s['ticker']} ({s['reason']})" for s in signals['sell']]) + "\n\n"
    if signals.get('trim'):
        msg += "✂️ <b>TRIMS:</b>\n" + "\n".join([f"- {s['ticker']} ({s['reason']})" for s in signals['trim']]) + "\n\n"
    if signals['buy']:
        msg += "🟢 <b>MOUVEMENTS:</b>\n" + "\n".join([f"- {b['action']} {b['ticker']} ({b['amount_eur']:.0f}€)" for b in signals['buy']]) + "\n\n"

    msg += "🏆 <b>TOP 3:</b>\n"
    for i, ticker in enumerate(valid_scores.head(3).index, 1):
        msg += f"{i}. {ticker} ({valid_scores[ticker]:.3f})\n"

    if not signals['sell'] and not signals.get('trim') and not signals['buy']:
        msg += "\n😴 Aucun mouvement."

    send_telegram(msg)

if __name__ == "__main__":
    main()
