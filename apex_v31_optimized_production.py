"""
APEX v31 OPTIMISÉ - PRODUCTION
===============================
Paramètres validés sur backtest 2015-2026 (10 ans):
- MFE Threshold: 15% (activer trailing dès +15%)
- Trailing: 5% (vendre si chute de 5% depuis le plus haut)
- Hard Stop: 18% UNIFORME (plus de stops par catégorie!)
- Pas de blacklist

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
COST_PER_TRADE = 1.0  # ✅ frais en EUR
PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# PARAMÈTRES OPTIMISÉS
# ============================================================
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

HARD_STOP_PCT = 0.18
MFE_THRESHOLD_PCT = 0.15
TRAILING_PCT = 0.05

FORCE_ROTATION_DAYS = 10

# ============================================================
# DATABASE - 44 TICKERS
# ============================================================
DATABASE = [
    "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA",
    "AMD", "MU", "ASML", "TSM", "LRCX", "AMAT",
    "PLTR", "APP", "CRWD", "NET", "DDOG", "ZS",
    "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER",
    "MSTR", "MARA", "RIOT", "CEG",
    "LLY", "NVO", "UNH", "JNJ", "ABBV",
    "WMT", "COST", "PG", "KO",
    "XOM", "CVX",
    "QQQ", "SPY", "GLD", "SLV",
]

# Categories (info)
ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML"}
TECH = {"APP", "TSLA", "NVDA", "PLTR", "DDOG"}

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

# ============================================================
# STOP LOSS UNIFORME -18%
# ============================================================
def get_stop_loss_pct(ticker, defensive=False):
    base_stop = HARD_STOP_PCT
    return base_stop * 0.85 if defensive else base_stop

def calculate_stop_price(entry_price, stop_pct):
    return entry_price * (1 - stop_pct)

# ============================================================
# MFE TRAILING STOP
# ============================================================
def check_mfe_trailing_exit(pos, current_price, entry_price):
    peak_price = pos.get("peak_price_eur", entry_price)
    if current_price > peak_price:
        peak_price = current_price
        pos["peak_price_eur"] = peak_price

    mfe_pct = (peak_price / entry_price - 1)
    drawdown_from_peak = (current_price / peak_price - 1)
    current_gain = (current_price / entry_price - 1)

    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT

    if trailing_active and drawdown_from_peak <= -TRAILING_PCT:
        return True, "MFE_TRAILING", {
            "mfe_pct": mfe_pct * 100,
            "peak_price": peak_price,
            "drawdown_pct": drawdown_from_peak * 100,
            "current_gain_pct": current_gain * 100
        }

    return False, None, {
        "trailing_active": trailing_active,
        "mfe_pct": mfe_pct * 100,
        "peak_price": peak_price,
        "drawdown_pct": drawdown_from_peak * 100
    }

def check_hard_stop_exit(current_price, entry_price, stop_price):
    if current_price <= stop_price:
        loss_pct = (current_price / entry_price - 1) * 100
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

# ============================================================
# ALLOCATION PONDÉRÉE
# ============================================================
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

# ============================================================
# CONVERSION EUR/USD (plus robuste)
# ============================================================
def get_eur_usd_rate():
    try:
        fx = yf.download("EURUSD=X", period="7d", interval="1d", progress=False, auto_adjust=True)
        if not fx.empty:
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 0:
                return rate
    except Exception:
        pass
    return 1.08

def usd_to_eur(amount_usd, rate=None):
    rate = rate or get_eur_usd_rate()
    return amount_usd / rate

def eur_to_usd(amount_eur, rate=None):
    rate = rate or get_eur_usd_rate()
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
                if not isinstance(history, dict):
                    return default_history
                if "trades" not in history:
                    history["trades"] = []
                if "summary" not in history:
                    history["summary"] = default_history["summary"]
                else:
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

def log_trade(history, action, ticker, price_usd, price_eur, shares, amount_eur, eur_rate,
              reason="", pnl_eur=None, pnl_pct=None):
    if "trades" not in history:
        history["trades"] = []

    trade = {
        "id": len(history["trades"]) + 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "ticker": ticker,
        "shares": round(float(shares), 6),
        "price_usd": round(float(price_usd), 4),
        "price_eur": round(float(price_eur), 4),
        "amount_eur": round(float(amount_eur), 2),
        "fee_eur": COST_PER_TRADE,
        "eur_usd_rate": round(float(eur_rate), 6),
        "reason": reason
    }
    if pnl_eur is not None:
        trade["pnl_eur"] = round(float(pnl_eur), 2)
        trade["pnl_pct"] = round(float(pnl_pct), 2)

    history["trades"].append(trade)

    if "summary" not in history:
        history["summary"] = {
            "total_trades": 0, "buys": 0, "sells": 0, "pyramids": 0,
            "winning_trades": 0, "losing_trades": 0, "total_pnl_eur": 0.0,
            "total_fees_eur": 0.0, "best_trade_eur": 0.0, "worst_trade_eur": 0.0,
            "win_rate": 0.0
        }

    summary = history["summary"]
    summary["total_trades"] += 1
    summary["total_fees_eur"] += COST_PER_TRADE

    if action == "BUY":
        summary["buys"] += 1
    elif action == "SELL":
        summary["sells"] += 1
        if pnl_eur is not None:
            summary["total_pnl_eur"] += float(pnl_eur)
            if pnl_eur > 0:
                summary["winning_trades"] += 1
            else:
                summary["losing_trades"] += 1
            summary["best_trade_eur"] = max(summary.get("best_trade_eur", 0.0), float(pnl_eur))
            summary["worst_trade_eur"] = min(summary.get("worst_trade_eur", 0.0), float(pnl_eur))
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
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

# ============================================================
# MARKET DATA & SCORING
# ============================================================
def get_market_data(tickers, days=160):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True
        )
        return data
    except Exception as e:
        print(f"Erreur download: {e}")
        return None

def calculate_momentum_score(close, high, low, volume=None,
                             atr_period=14, sma_period=20, high_lookback=60):
    """
    Score momentum 0-10.
    Pondération : 45% distance SMA20, 35% retour 10j, 15% pénalité high 60j, 5% volume relatif.
    """
    needed = max(atr_period, sma_period, high_lookback, 20) + 15
    if len(close) < needed:
        return np.nan

    sma20 = close.rolling(sma_period).mean()

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    atr_last = atr.iloc[-1]
    if pd.isna(atr_last) or atr_last <= 0:
        return np.nan

    dist_sma20 = (close.iloc[-1] - sma20.iloc[-1]) / atr_last
    norm_dist_sma20 = min(max(dist_sma20, 0), 3.0) / 3.0

    retour_10j = close.pct_change(10).iloc[-1]
    norm_retour_10j = min(max(retour_10j, 0), 0.4) / 0.4

    high60 = high.rolling(high_lookback).max().iloc[-1]
    dist_high60 = (high60 - close.iloc[-1]) / atr_last
    norm_penalite = min(max(dist_high60, 0), 5.0) / 5.0
    score_penalite = 1 - norm_penalite

    # Volume relatif (optionnel)
    norm_volume = 0.0
    if volume is not None and len(volume.dropna()) >= 20:
        v = float(volume.iloc[-1])
        v_ma = float(volume.rolling(20).mean().iloc[-1])
        if v_ma > 0:
            volume_rel = v / v_ma
            norm_volume = min(max(volume_rel - 1, 0), 2.0) / 2.0

    score = (
        0.45 * norm_dist_sma20
        + 0.35 * norm_retour_10j
        + 0.15 * score_penalite
        + 0.05 * norm_volume
    ) * 10

    return float(score) if not pd.isna(score) else np.nan

def get_vix():
    try:
        v = yf.download("^VIX", period="7d", interval="1d", progress=False, auto_adjust=True)
        if not v.empty:
            return float(v["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0

def get_regime(vix):
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
    print("⚙️ Paramètres: Hard Stop -18%, MFE Trailing +15%/-5%")

    portfolio = load_portfolio()
    history = load_trades_history()

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
    if last_dca is None or not str(last_dca).startswith(current_month):
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

    # Scores
    scores = {}
    current_prices = {}

    for ticker in DATABASE:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                tdf = data[ticker].dropna()
            else:
                tdf = data.dropna()

            close = tdf["Close"].dropna()
            high = tdf["High"].dropna()
            low = tdf["Low"].dropna()
            volume = tdf["Volume"].dropna() if "Volume" in tdf.columns else None

            if len(close) == 0:
                continue

            current_prices[ticker] = float(close.iloc[-1])
            score = calculate_momentum_score(close, high, low, volume=volume)
            if not np.isnan(score) and score > 0:
                scores[ticker] = score

        except Exception:
            continue

    current_prices = pd.Series(current_prices, dtype=float)
    valid_scores = pd.Series(scores, dtype=float).sort_values(ascending=False)
    print(f"\n📊 {len(valid_scores)} tickers avec score > 0")

    signals = {"sell": [], "buy": [], "pyramid": [], "force_rotation": []}

    # ============================================================
    # 1. CHECK POSITIONS
    # ============================================================
    print(f"\n{'='*70}")
    print("📂 VÉRIFICATION DES POSITIONS")
    print(f"{'='*70}")

    positions_to_remove = []

    for ticker, pos in list(portfolio["positions"].items()):
        if ticker not in current_prices.index:
            continue

        current_price_usd = float(current_prices[ticker])
        current_price_eur = usd_to_eur(current_price_usd, eur_rate)

        entry_price_eur = float(pos["entry_price_eur"])
        shares = float(pos["shares"])

        if current_price_eur > pos.get("peak_price_eur", entry_price_eur):
            pos["peak_price_eur"] = current_price_eur

        stop_pct = get_stop_loss_pct(ticker, defensive)
        stop_price_eur = calculate_stop_price(entry_price_eur, stop_pct)
        pos["stop_loss_eur"] = stop_price_eur

        pnl_eur = (current_price_eur - entry_price_eur) * shares
        pnl_pct = (current_price_eur / entry_price_eur - 1) * 100

        current_score = float(valid_scores.get(ticker, 0.0))
        pos["score"] = current_score

        if ticker in valid_scores.index:
            pos["rank"] = list(valid_scores.index).index(ticker) + 1
        else:
            pos["rank"] = 999

        print(f"\n🔹 {ticker}")
        print(f" Prix: {current_price_eur:.2f}€ (entrée: {entry_price_eur:.2f}€)")
        print(f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f" Peak: {pos.get('peak_price_eur', entry_price_eur):.2f}€")
        print(f" Score: {current_score:.3f} | Rank: #{pos['rank']}")

        should_sell = False
        sell_reason = ""

        # ✅ CHECK 1: Hard Stop
        hit_hard_stop, hard_stop_reason = check_hard_stop_exit(
            current_price_eur, entry_price_eur, stop_price_eur
        )
        if hit_hard_stop:
            should_sell = True
            sell_reason = hard_stop_reason
            print(f" ❌ HARD STOP touché! ({stop_price_eur:.2f}€)")

        # CHECK 2: MFE trailing
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(
                pos, current_price_eur, entry_price_eur
            )
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason
                print(" 📉 MFE TRAILING déclenché!")
                print(f" MFE: +{mfe_details['mfe_pct']:.1f}%")
                print(f" Drawdown: {mfe_details['drawdown_pct']:.1f}%")
            else:
                status = "ACTIF" if mfe_details["trailing_active"] else "INACTIF"
                print(f" 🎯 Trailing: {status} (MFE: +{mfe_details['mfe_pct']:.1f}%)")

        # CHECK 3: Force rotation score<=0
        if not should_sell and current_score <= 0:
            days_zero = int(pos.get("days_zero_score", 0)) + 1
            pos["days_zero_score"] = days_zero
            print(f" ⚠️ Score ≤ 0 depuis {days_zero} jour(s)")

            if days_zero >= FORCE_ROTATION_DAYS:
                for candidate in valid_scores.index:
                    if candidate not in portfolio["positions"]:
                        signals["force_rotation"].append({
                            "ticker": ticker,
                            "replacement": candidate,
                            "replacement_score": float(valid_scores[candidate]),
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
    # 2. BUY OPPORTUNITIES
    # ============================================================
    available_cash = float(portfolio["cash"])
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

            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    tdf = data[ticker].dropna()
                else:
                    tdf = data.dropna()
            except Exception:
                continue

            close = tdf["Close"].dropna()
            high = tdf["High"].dropna()
            low = tdf["Low"].dropna()

            if len(close) < 60 or len(high) < 60:
                continue

            # Freshness
            last_date = close.index[-1]
            last_high_date_20 = high.tail(20).idxmax()
            days_since_high_20 = (last_date - last_high_date_20).days
            retour_20j = close.pct_change(20).iloc[-1]
            freshness_ok = (days_since_high_20 <= 8) or (retour_20j >= 0.05)

            # Anti-chasse (ATR14)
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            if pd.isna(atr14) or atr14 <= 0:
                continue
            sma20 = close.rolling(20).mean().iloc[-1]
            dist_sma20 = (close.iloc[-1] - sma20) / atr14
            anti_chasse_ok = dist_sma20 <= 2.0

            if not freshness_ok or not anti_chasse_ok:
                continue

            is_replacement = any(rot["replacement"] == ticker for rot in signals["force_rotation"])

            rank = list(valid_scores.index).index(ticker) + 1
            if rank > max_positions and not is_replacement:
                continue

            current_price_usd = float(current_prices.get(ticker, np.nan))
            if np.isnan(current_price_usd) or current_price_usd <= 0:
                continue
            current_price_eur = usd_to_eur(current_price_usd, eur_rate)

            allocation = get_weighted_allocation(rank, max_positions, available_cash)
            allocation = min(allocation, max(0.0, available_cash - 10.0))

            if allocation < 50:
                continue

            shares = allocation / current_price_eur

            stop_pct = get_stop_loss_pct(ticker, defensive)
            stop_price = calculate_stop_price(current_price_eur, stop_pct)

            signals["buy"].append({
                "ticker": ticker,
                "rank": rank,
                "score": float(valid_scores[ticker]),
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "shares": shares,
                "amount_eur": allocation,
                "allocation_pct": allocation / available_cash * 100 if available_cash > 0 else 0,
                "stop_loss_eur": stop_price,
                "stop_loss_pct": stop_pct * 100
            })

            available_cash -= allocation
            slots_available -= 1

            print(f"\n🟢 ACHETER #{rank}: {ticker}")
            print(f" Score: {valid_scores[ticker]:.3f}")
            print(f" Montant: {allocation:.2f}€")
            print(f" Actions: {shares:.4f}")
            print(f" Stop: {stop_price:.2f}€ (-{stop_pct*100:.0f}%)")

    # ============================================================
    # EXECUTION ORDERS
    # ============================================================
    # Ventes
    for sell in signals["sell"]:
        ticker = sell["ticker"]
        proceeds = max(0.0, sell["value_eur"] - COST_PER_TRADE)  # ✅ frais en EUR
        portfolio["cash"] = float(portfolio["cash"]) + proceeds

        log_trade(
            history, "SELL", ticker,
            sell["price_usd"], sell["price_eur"],
            sell["shares"], sell["value_eur"],
            eur_rate,
            reason=sell["reason"],
            pnl_eur=sell["pnl_eur"],
            pnl_pct=sell["pnl_pct"]
        )

        if ticker in portfolio["positions"]:
            del portfolio["positions"][ticker]

    # Achats
    for buy in signals["buy"]:
        ticker = buy["ticker"]
        cost = buy["amount_eur"] + COST_PER_TRADE  # ✅ frais en EUR
        if float(portfolio["cash"]) < cost:
            continue
        portfolio["cash"] = float(portfolio["cash"]) - cost

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

        log_trade(
            history, "BUY", ticker,
            buy["price_usd"], buy["price_eur"],
            buy["shares"], buy["amount_eur"],
            eur_rate,
            reason=f"signal_rank{buy['rank']}"
        )

    # ============================================================
    # PORTFOLIO SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ PORTFOLIO")
    print(f"{'='*70}")

    total_positions_value = 0.0
    for ticker, pos in portfolio["positions"].items():
        if ticker in current_prices.index:
            current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
            total_positions_value += current_price_eur * float(pos["shares"])

    total_value = float(portfolio["cash"]) + total_positions_value
    start_date = datetime.strptime(portfolio["start_date"], "%Y-%m-%d")
    months_elapsed = (datetime.now().year - start_date.year) * 12 + (datetime.now().month - start_date.month)
    total_invested = float(portfolio["initial_capital"]) + max(0, months_elapsed) * float(MONTHLY_DCA)

    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1) * 100 if total_invested > 0 else 0.0

    print(
        f" 💵 Cash disponible: {float(portfolio['cash']):.2f}€\n"
        f"📈 Valeur positions: {total_positions_value:.2f}€\n"
        f"────────────────────────────────────\n"
        f"💰 VALEUR TOTALE: {total_value:.2f}€\n"
        f"📊 Total investi: {total_invested:.2f}€\n"
        f"{'📈' if total_pnl >= 0 else '📉'} PnL: {total_pnl:+.2f}€ ({total_pnl_pct:+.1f}%)\n"
    )

    save_portfolio(portfolio)
    save_trades_history(history)

    print("\n💾 Portfolio et historique sauvegardés")

    # ============================================================
    # TELEGRAM MESSAGE
    # ============================================================
    msg = f"📊 <b>APEX v31 OPTIMISÉ</b> - {today}\n"
    msg += f"{regime} | VIX: {current_vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | Trail: +15%/-5%\n\n"

    if signals["sell"] or signals["buy"] or signals["force_rotation"]:
        msg += "🚨 <b>ACTIONS À FAIRE</b>\n\n"

        for rot in signals["force_rotation"]:
            msg += "🔄 <b>ROTATION FORCÉE</b>\n"
            msg += f" {rot['ticker']} → {rot['replacement']}\n"
            msg += f" Score=0 depuis {rot['days_zero']}j\n\n"

        for sell in signals["sell"]:
            msg += f"🔴 <b>VENDRE {sell['ticker']}</b>\n"
            msg += f" Actions: {sell['shares']:.4f}\n"
            msg += f" Montant: ~{sell['value_eur']:.2f}€\n"
            msg += f" Raison: {sell['reason']}\n"
            msg += f" PnL: {sell['pnl_eur']:+.2f}€ ({sell['pnl_pct']:+.1f}%)\n\n"

        for buy in signals["buy"]:
            msg += f"🟢 <b>ACHETER #{buy['rank']} {buy['ticker']}</b>\n"
            msg += f" 💶 Montant: <b>{buy['amount_eur']:.2f}€</b>\n"
            msg += f" 📊 Actions: <b>{buy['shares']:.4f}</b>\n"
            msg += f" Stop: {buy['stop_loss_eur']:.2f}€ (-18%)\n"
            msg += f" MFE Trigger: {buy['price_eur']*1.15:.2f}€ (+15%)\n\n"
    else:
        msg += "✅ <b>Aucun signal - HOLD</b>\n\n"

    msg += "📂 <b>MES POSITIONS</b>\n"
    for ticker, pos in portfolio["positions"].items():
        if ticker in current_prices.index:
            current_price_eur = usd_to_eur(float(current_prices[ticker]), eur_rate)
            entry_price_eur = float(pos["entry_price_eur"])
            pnl_pct = (current_price_eur / entry_price_eur - 1) * 100
            pnl_eur = (current_price_eur - entry_price_eur) * float(pos["shares"])
            mfe_pct = (float(pos.get("peak_price_eur", entry_price_eur)) / entry_price_eur - 1) * 100
            trailing_status = "🟢ACTIF" if mfe_pct >= 15 else "⚪️"
            emoji = "📈" if pnl_pct >= 0 else "📉"
            msg += f"{emoji} {ticker} #{pos.get('rank', 'N/A')}\n"
            msg += f" {float(pos['amount_invested_eur']):.0f}€ → {current_price_eur * float(pos['shares']):.0f}€\n"
            msg += f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)\n"
            msg += f" Trail: {trailing_status} MFE:+{mfe_pct:.1f}%\n"

    msg += f"\n💰 <b>TOTAL: {total_value:.2f}€</b> ({total_pnl_pct:+.1f}%)\n"

    msg += "\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, ticker in enumerate(valid_scores.head(5).index, 1):
        if ticker in current_prices.index:
            price = usd_to_eur(float(current_prices[ticker]), eur_rate)
            in_pf = "📂" if ticker in portfolio["positions"] else "👀"
            msg += f"{i}. {ticker} @ {price:.2f}€ ({valid_scores[ticker]:.3f}) {in_pf}\n"

    send_telegram(msg)

    print(f"\n{'='*70}")
    print("✅ APEX v31 OPTIMISÉ terminé")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
