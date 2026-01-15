"""
APEX v32 BASELINE WINNER — PRODUCTION (SAFE @ 7h Paris)
======================================================

Objectif:
- Baseline winner (sans filtres freshness/anti-chasse)
- Stops: Hard stop -18% (ou -15.3% en mode défensif VIX>=25)
- MFE Trailing: activé dès +15% MFE, sortie si -5% depuis le peak
- Rotation forcée: uniquement si score<=0 pendant X jours ET trailing INACTIF

Sécurité:
- Par défaut: SIGNAL ONLY (ne modifie pas portfolio.json / trades_history.json)
- Pour exécuter: --execute ou variable d'env APEX_EXECUTE=1

Scheduling:
- Conçu pour tourner via GitHub Actions (cron UTC) mais envoyer à 7h Paris
- Time guard (Europe/Paris) pour n'envoyer qu'à 07:00 locale si activé
"""

import os
import json
import argparse
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# =========================
# CONFIG
# =========================
INITIAL_CAPITAL = 1500
MONTHLY_DCA = 100
COST_PER_TRADE = 1.0  # frais fixes EUR (optionnel)

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Positions & regime
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

# Momentum score
ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

# Stops baseline winner
HARD_STOP_PCT = 0.18
MFE_THRESHOLD_PCT = 0.15
TRAILING_PCT = 0.05

FORCE_ROTATION_DAYS = 10  # rotation si score<=0 X jours (mais SAFE)

# Universe
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

ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML", "NVDA", "TSM", "SMCI"}
TECH = {"APP", "TSLA", "NVDA", "PLTR", "DDOG"}


# =========================
# UTILS
# =========================
def get_category(ticker: str) -> str:
    if ticker in ULTRA_VOLATILE:
        return "ultra"
    if ticker in CRYPTO:
        return "crypto"
    if ticker in SEMI:
        return "semi"
    if ticker in TECH:
        return "tech"
    return "other"


def safe_float(x, default=np.nan):
    try:
        if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray)):
            # empêcher "truth value ambiguous" & conversions bizarres
            if hasattr(x, "iloc"):
                x = x.iloc[-1]
            else:
                x = np.array(x).reshape(-1)[-1]
        return float(x)
    except Exception:
        return default


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                txt = f.read().strip()
            if not txt:
                return default
            return json.loads(txt)
        except Exception:
            return default
    return default


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# =========================
# FX / VIX
# =========================
def get_eur_usd_rate():
    # Dernier close daily EURUSD
    try:
        fx = yf.download("EURUSD=X", period="14d", interval="1d", progress=False, auto_adjust=True)
        if fx is not None and not fx.empty:
            rate = safe_float(fx["Close"].dropna().iloc[-1], default=np.nan)
            if not np.isnan(rate) and rate > 0:
                return rate
    except Exception:
        pass
    return 1.08


def usd_to_eur(amount_usd, rate):
    return float(amount_usd) / float(rate)


def get_vix_last_close():
    try:
        v = yf.download("^VIX", period="14d", interval="1d", progress=False, auto_adjust=True)
        if v is not None and not v.empty:
            return safe_float(v["Close"].dropna().iloc[-1], default=20.0)
    except Exception:
        pass
    return 20.0


def get_regime(vix: float):
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "🔴 ULTRA-DÉFENSIF", MAX_POSITIONS_ULTRA_DEFENSIVE
    if vix >= VIX_DEFENSIVE:
        return "🟡 DÉFENSIF", MAX_POSITIONS_DEFENSIVE
    return "🟢 NORMAL", MAX_POSITIONS_NORMAL


# =========================
# Stops
# =========================
def get_stop_loss_pct(defensive: bool) -> float:
    # Même stop, juste un peu plus serré en mode défensif
    return HARD_STOP_PCT * 0.85 if defensive else HARD_STOP_PCT


def calculate_stop_price(entry_price, stop_pct):
    return float(entry_price) * (1 - float(stop_pct))


def check_hard_stop_exit(current_price, entry_price, stop_price):
    if float(current_price) <= float(stop_price):
        loss_pct = (float(current_price) / float(entry_price) - 1) * 100
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None


def check_mfe_trailing_exit(pos_dict, current_price, entry_price):
    """
    Trailing activé si MFE >= +15%.
    Sortie si drawdown depuis peak <= -5%.
    """
    entry_price = float(entry_price)
    current_price = float(current_price)

    peak = safe_float(pos_dict.get("peak_price_eur", entry_price), default=entry_price)
    if current_price > peak:
        peak = current_price
        pos_dict["peak_price_eur"] = peak

    mfe_pct = peak / entry_price - 1
    drawdown = current_price / peak - 1
    current_gain = current_price / entry_price - 1

    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT

    details = {
        "trailing_active": bool(trailing_active),
        "mfe_pct": mfe_pct * 100,
        "peak_price": peak,
        "drawdown_pct": drawdown * 100,
        "current_gain_pct": current_gain * 100,
    }

    if trailing_active and drawdown <= -TRAILING_PCT:
        return True, "MFE_TRAILING", details

    return False, None, details


# =========================
# Momentum score (UNIQUE)
# =========================
def calculate_momentum_score(close, high, low, volume=None,
                             atr_period=14, sma_period=20, high_lookback=60):
    """
    Score 0-10
    - 45% dist SMA20 (en ATR)
    - 35% retour 10j
    - 15% pénalité dist high 60j (en ATR)
    - 5% volume relatif
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

    norm_volume = 0.0
    if volume is not None and len(volume.dropna()) >= 20:
        v = safe_float(volume.iloc[-1], default=np.nan)
        v_ma = safe_float(volume.rolling(20).mean().iloc[-1], default=np.nan)
        if not np.isnan(v) and not np.isnan(v_ma) and v_ma > 0:
            volume_rel = v / v_ma
            norm_volume = min(max(volume_rel - 1, 0), 2.0) / 2.0

    score = (
        0.45 * norm_dist_sma20
        + 0.35 * norm_retour_10j
        + 0.15 * score_penalite
        + 0.05 * norm_volume
    ) * 10

    return float(score) if not pd.isna(score) else np.nan


# =========================
# Portfolio + trades
# =========================
def load_portfolio():
    default = {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": float(INITIAL_CAPITAL),
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {}
    }
    return load_json(PORTFOLIO_FILE, default)


def load_trades_history():
    default = {
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
    return load_json(TRADES_HISTORY_FILE, default)


def log_trade(history, action, ticker, price_usd, price_eur, shares, amount_eur, eur_rate,
              trade_date, reason="", pnl_eur=None, pnl_pct=None):
    if "trades" not in history:
        history["trades"] = []
    if "summary" not in history:
        history["summary"] = {}

    trade = {
        "id": len(history["trades"]) + 1,
        "date": str(trade_date),
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "ticker": ticker,
        "shares": round(float(shares), 6),
        "price_usd": round(float(price_usd), 4),
        "price_eur": round(float(price_eur), 4),
        "amount_eur": round(float(amount_eur), 2),
        "fee_eur": float(COST_PER_TRADE),
        "eur_usd_rate": round(float(eur_rate), 6),
        "reason": reason
    }
    if pnl_eur is not None:
        trade["pnl_eur"] = round(float(pnl_eur), 2)
        trade["pnl_pct"] = round(float(pnl_pct), 2)

    history["trades"].append(trade)

    # summary update
    s = history["summary"]
    s.setdefault("total_trades", 0)
    s.setdefault("buys", 0)
    s.setdefault("sells", 0)
    s.setdefault("pyramids", 0)
    s.setdefault("winning_trades", 0)
    s.setdefault("losing_trades", 0)
    s.setdefault("total_pnl_eur", 0.0)
    s.setdefault("total_fees_eur", 0.0)
    s.setdefault("best_trade_eur", 0.0)
    s.setdefault("worst_trade_eur", 0.0)
    s.setdefault("win_rate", 0.0)

    s["total_trades"] += 1
    s["total_fees_eur"] += float(COST_PER_TRADE)

    if action == "BUY":
        s["buys"] += 1
    elif action == "SELL":
        s["sells"] += 1
        if pnl_eur is not None:
            s["total_pnl_eur"] += float(pnl_eur)
            if float(pnl_eur) > 0:
                s["winning_trades"] += 1
            else:
                s["losing_trades"] += 1
            s["best_trade_eur"] = max(float(s.get("best_trade_eur", 0.0)), float(pnl_eur))
            s["worst_trade_eur"] = min(float(s.get("worst_trade_eur", 0.0)), float(pnl_eur))
            closed = s["winning_trades"] + s["losing_trades"]
            if closed > 0:
                s["win_rate"] = round(s["winning_trades"] / closed * 100, 1)


# =========================
# Telegram
# =========================
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configuré (token/chat_id manquants)")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False


# =========================
# Market data
# =========================
def get_market_data(tickers, days=220):
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


def find_asof_date(data) -> date:
    """
    Détermine la date "as-of" (dernier jour dispo) à partir d'un ticker stable.
    """
    for t in ["SPY", "QQQ", DATABASE[0]]:
        try:
            if isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0):
                idx = data[t]["Close"].dropna().index
            else:
                idx = data["Close"].dropna().index
            if len(idx) > 0:
                return idx[-1].date()
        except Exception:
            continue
    return datetime.now().date()


# =========================
# Allocation
# =========================
def get_weighted_allocation(rank, num_positions, total_capital):
    total_capital = float(total_capital)
    if num_positions <= 1:
        return total_capital
    if num_positions == 2:
        weights = {1: 0.60, 2: 0.40}
    elif num_positions == 3:
        weights = {1: 0.50, 2: 0.30, 3: 0.20}
    else:
        total_weight = sum(range(1, num_positions + 1))
        weights = {i: (num_positions - i + 1) / total_weight for i in range(1, num_positions + 1)}
    return total_capital * weights.get(rank, 1.0 / num_positions)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--execute", action="store_true", help="Applique réellement les trades (modifie JSON).")
    parser.add_argument("--force_send", action="store_true", help="Ignore le guard 7h Paris et envoie quand même.")
    parser.add_argument("--guard_7h_paris", action="store_true", help="N'envoie que si heure locale Europe/Paris == 07:00.")
    args, _unknown = parser.parse_known_args()  # IMPORTANT: ignore -f kernel.json etc.

    # Execution gate
    env_execute = os.environ.get("APEX_EXECUTE", "").strip() == "1"
    EXECUTE = bool(args.execute or env_execute)

    # 7h Paris guard
    tz = ZoneInfo("Europe/Paris")
    now_paris = datetime.now(tz)
    if args.guard_7h_paris and not args.force_send:
        if not (now_paris.hour == 7):
            print(f"⏭️ Guard 7h Paris actif: il est {now_paris.strftime('%H:%M')} à Paris, pas 07:xx. Exit.")
            return

    print("=" * 70)
    print("🚀 APEX v32 BASELINE WINNER — PRODUCTION (SAFE)")
    print("=" * 70)
    print(f"🕖 Paris now: {now_paris.strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {'✅ EXECUTE' if EXECUTE else '🟦 SIGNAL ONLY (no write)'}")
    print("Stops: HardStop -18% | MFE +15% then trail -5%")
    print("Note: à 7h Paris, les signaux sont calculés sur le DERNIER close disponible.\n")

    # Load state
    portfolio = load_portfolio()
    history = load_trades_history()

    # Market info
    eur_rate = get_eur_usd_rate()
    vix = get_vix_last_close()
    regime, max_positions = get_regime(vix)
    defensive = vix >= VIX_DEFENSIVE

    # Download prices
    print("📥 Téléchargement données...")
    data = get_market_data(DATABASE)
    if data is None or data.empty:
        send_telegram("❌ APEX v32: Erreur téléchargement données (yfinance).")
        return

    asof = find_asof_date(data)  # dernier jour dispo (close)
    asof_str = str(asof)

    print(f"📌 As-of date (close): {asof_str}")
    print(f"💱 EUR/USD: {eur_rate:.4f} | 📊 VIX: {vix:.1f} | 📈 Régime: {regime} (max {max_positions})")

    # DCA: basé sur asof (le mois du dernier close), pas "now"
    last_dca = portfolio.get("last_dca_date")
    asof_month = asof_str[:7]
    if last_dca is None or not str(last_dca).startswith(asof_month):
        if EXECUTE:
            portfolio["cash"] = float(portfolio.get("cash", 0.0)) + float(MONTHLY_DCA)
            portfolio["last_dca_date"] = asof_str
        print(f"💰 DCA mensuel prévu pour {asof_month}: +{MONTHLY_DCA}€ ({'appliqué' if EXECUTE else 'signal'})")

    # Compute scores + current prices (as-of close)
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

            current_prices[ticker] = safe_float(close.iloc[-1], default=np.nan)
            score = calculate_momentum_score(close, high, low, volume=volume)
            if not np.isnan(score) and score > 0:
                scores[ticker] = float(score)

        except Exception:
            continue

    current_prices = pd.Series(current_prices, dtype=float).dropna()
    valid_scores = pd.Series(scores, dtype=float).sort_values(ascending=False)

    print(f"📊 {len(valid_scores)} tickers avec score > 0\n")

    signals = {"sell": [], "buy": [], "force_rotation": []}
    positions_to_remove = []

    # =========================
    # 1) CHECK POSITIONS
    # =========================
    print("=" * 70)
    print("📂 CHECK POSITIONS")
    print("=" * 70)

    for ticker, pos in list(portfolio.get("positions", {}).items()):
        if ticker not in current_prices.index:
            continue

        price_usd = float(current_prices[ticker])
        price_eur = usd_to_eur(price_usd, eur_rate)

        entry_eur = float(pos["entry_price_eur"])
        shares = float(pos["shares"])

        # stop
        stop_pct = get_stop_loss_pct(defensive)
        stop_price = calculate_stop_price(entry_eur, stop_pct)
        pos["stop_loss_eur"] = stop_price

        pnl_eur = (price_eur - entry_eur) * shares
        pnl_pct = (price_eur / entry_eur - 1) * 100

        # score + rank
        score_now = float(valid_scores.get(ticker, 0.0))
        pos["score"] = score_now
        if ticker in valid_scores.index:
            pos["rank"] = int(list(valid_scores.index).index(ticker) + 1)
        else:
            pos["rank"] = 999

        # trailing check (also updates peak in pos)
        hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(pos, price_eur, entry_eur)
        trailing_active = bool(mfe_details.get("trailing_active", False))
        pos["trailing_active"] = trailing_active
        pos["mfe_pct"] = float(mfe_details.get("mfe_pct", 0.0))

        print(f"\n🔹 {ticker}  | Rank #{pos['rank']} | Score {score_now:.3f}")
        print(f"  Prix: {price_eur:.2f}€ (entrée {entry_eur:.2f}€) | PnL {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f"  Peak: {float(pos.get('peak_price_eur', entry_eur)):.2f}€ | MFE {pos['mfe_pct']:+.1f}% | Trailing {'ACTIF' if trailing_active else 'INACTIF'}")
        print(f"  Stop: {stop_price:.2f}€ (-{stop_pct*100:.1f}%)")

        should_sell = False
        sell_reason = ""

        # Hard stop
        hit_hs, hs_reason = check_hard_stop_exit(price_eur, entry_eur, stop_price)
        if hit_hs:
            should_sell = True
            sell_reason = hs_reason

        # MFE trailing exit
        if not should_sell and hit_mfe:
            should_sell = True
            sell_reason = mfe_reason

        # Force rotation SAFE:
        # - seulement si trailing INACTIF
        # - et rank "mauvais" (évite de sortir trop tôt un leader en consolidation)
        if not should_sell:
            if score_now <= 0:
                days_zero = int(pos.get("days_zero_score", 0)) + 1
                pos["days_zero_score"] = days_zero
            else:
                pos["days_zero_score"] = 0

            rank_now = int(pos.get("rank", 999))
            if (pos.get("days_zero_score", 0) >= FORCE_ROTATION_DAYS
                and (not trailing_active)
                and rank_now > max_positions * 2):
                # find replacement
                for candidate in valid_scores.index:
                    if candidate not in portfolio["positions"]:
                        signals["force_rotation"].append({
                            "ticker": ticker,
                            "replacement": candidate,
                            "days_zero": int(pos["days_zero_score"]),
                        })
                        should_sell = True
                        sell_reason = f"FORCE_ROTATION_{pos['days_zero_score']}j"
                        break

        if should_sell:
            signals["sell"].append({
                "ticker": ticker,
                "shares": shares,
                "price_usd": price_usd,
                "price_eur": price_eur,
                "value_eur": price_eur * shares,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct,
                "reason": sell_reason,
            })
            positions_to_remove.append(ticker)
            print(f"  🔴 SIGNAL SELL: {sell_reason}")

    # =========================
    # 2) BUY OPPORTUNITIES
    # =========================
    available_cash = float(portfolio.get("cash", 0.0))
    future_positions = len(portfolio.get("positions", {})) - len(positions_to_remove)
    slots = max_positions - future_positions

    if slots > 0 and available_cash > 50:
        print("\n" + "=" * 70)
        print(f"🛒 BUY OPPORTUNITIES — slots={slots} | cash={available_cash:.2f}€")
        print("=" * 70)

        for ticker in valid_scores.index:
            if slots <= 0 or available_cash < 50:
                break

            if ticker in portfolio["positions"] and ticker not in positions_to_remove:
                continue

            rank = int(list(valid_scores.index).index(ticker) + 1)
            if rank > max_positions:
                continue

            price_usd = safe_float(current_prices.get(ticker, np.nan), default=np.nan)
            if np.isnan(price_usd) or price_usd <= 0:
                continue
            price_eur = usd_to_eur(price_usd, eur_rate)

            allocation = get_weighted_allocation(rank, max_positions, available_cash)
            allocation = min(allocation, max(0.0, available_cash - 10.0))
            if allocation < 50:
                continue

            shares = allocation / price_eur

            stop_pct = get_stop_loss_pct(defensive)
            stop_price = calculate_stop_price(price_eur, stop_pct)

            signals["buy"].append({
                "ticker": ticker,
                "rank": rank,
                "score": float(valid_scores[ticker]),
                "price_usd": price_usd,
                "price_eur": price_eur,
                "shares": shares,
                "amount_eur": allocation,
                "stop_loss_eur": stop_price,
                "stop_loss_pct": stop_pct * 100,
            })

            available_cash -= allocation
            slots -= 1

            print(f"  🟢 BUY #{rank} {ticker} | {allocation:.2f}€ | stop {stop_price:.2f}€")

    # =========================
    # EXECUTE (optional)
    # =========================
    if EXECUTE:
        # SELL
        for s in signals["sell"]:
            proceeds = max(0.0, float(s["value_eur"]) - float(COST_PER_TRADE))
            portfolio["cash"] = float(portfolio.get("cash", 0.0)) + proceeds

            log_trade(
                history, "SELL", s["ticker"],
                s["price_usd"], s["price_eur"], s["shares"], s["value_eur"], eur_rate,
                trade_date=asof_str,
                reason=s["reason"],
                pnl_eur=s["pnl_eur"], pnl_pct=s["pnl_pct"]
            )
            if s["ticker"] in portfolio["positions"]:
                del portfolio["positions"][s["ticker"]]

        # BUY
        for b in signals["buy"]:
            cost = float(b["amount_eur"]) + float(COST_PER_TRADE)
            if float(portfolio.get("cash", 0.0)) < cost:
                continue
            portfolio["cash"] = float(portfolio.get("cash", 0.0)) - cost

            portfolio["positions"][b["ticker"]] = {
                "entry_price_eur": b["price_eur"],
                "entry_price_usd": b["price_usd"],
                "entry_date": asof_str,
                "shares": b["shares"],
                "initial_amount_eur": b["amount_eur"],
                "amount_invested_eur": b["amount_eur"],
                "score": b["score"],
                "peak_price_eur": b["price_eur"],
                "stop_loss_eur": b["stop_loss_eur"],
                "rank": b["rank"],
                "pyramided": False,
                "days_zero_score": 0,
            }

            log_trade(
                history, "BUY", b["ticker"],
                b["price_usd"], b["price_eur"], b["shares"], b["amount_eur"], eur_rate,
                trade_date=asof_str,
                reason=f"signal_rank{b['rank']}"
            )

        portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_json(PORTFOLIO_FILE, portfolio)
        save_json(TRADES_HISTORY_FILE, history)

    # =========================
    # SUMMARY
    # =========================
    # portfolio valuation as-of (approx)
    pos_value = 0.0
    for t, p in portfolio.get("positions", {}).items():
        if t in current_prices.index:
            px_eur = usd_to_eur(float(current_prices[t]), eur_rate)
            pos_value += px_eur * float(p["shares"])
    total_value = float(portfolio.get("cash", 0.0)) + pos_value

    # invested (approx since start_date)
    try:
        sd = datetime.strptime(portfolio["start_date"], "%Y-%m-%d").date()
        months = (asof.year - sd.year) * 12 + (asof.month - sd.month)
        months = max(0, months)
    except Exception:
        months = 0
    invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL)) + months * float(MONTHLY_DCA)
    pnl = total_value - invested
    pnl_pct = (total_value / invested - 1) * 100 if invested > 0 else 0.0

    # =========================
    # TELEGRAM MESSAGE
    # =========================
    mode_txt = "✅ EXECUTE" if EXECUTE else "🟦 SIGNAL ONLY (plan — exécution manuelle)"
    msg = f"📊 <b>APEX v32 BASELINE WINNER</b>\n"
    msg += f"🗓️ As-of (close): <b>{asof_str}</b>\n"
    msg += f"{mode_txt}\n"
    msg += f"{regime} | VIX: {vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | Trail: +15%/-5%\n\n"

    if signals["sell"] or signals["buy"] or signals["force_rotation"]:
        msg += "🚨 <b>PLAN DU JOUR</b>\n\n"

        for rot in signals["force_rotation"]:
            msg += "🔄 <b>ROTATION FORCÉE</b>\n"
            msg += f" {rot['ticker']} → {rot['replacement']} (score<=0 {rot['days_zero']}j)\n\n"

        for s in signals["sell"]:
            msg += f"🔴 <b>SELL {s['ticker']}</b>\n"
            msg += f" Raison: {s['reason']}\n"
            msg += f" PnL: {s['pnl_eur']:+.2f}€ ({s['pnl_pct']:+.1f}%)\n\n"

        for b in signals["buy"]:
            msg += f"🟢 <b>BUY #{b['rank']} {b['ticker']}</b>\n"
            msg += f" Montant: <b>{b['amount_eur']:.2f}€</b>\n"
            msg += f" Stop: {b['stop_loss_eur']:.2f}€\n"
            msg += f" Trigger MFE: {b['price_eur']*1.15:.2f}€\n\n"
    else:
        msg += "✅ <b>Aucun signal</b> — HOLD\n\n"

    msg += "📂 <b>POSITIONS</b>\n"
    for t, p in portfolio.get("positions", {}).items():
        if t in current_prices.index:
            px_eur = usd_to_eur(float(current_prices[t]), eur_rate)
            entry = float(p["entry_price_eur"])
            sh = float(p["shares"])
            pe = (px_eur - entry) * sh
            pp = (px_eur / entry - 1) * 100
            mfe = float(p.get("mfe_pct", 0.0))
            trail = "🟢ACTIF" if float(mfe) >= 15 else "⚪️"
            msg += f"{'📈' if pp >= 0 else '📉'} {t} #{p.get('rank','?')}\n"
            msg += f" PnL: {pe:+.2f}€ ({pp:+.1f}%) | Trail: {trail} MFE:+{mfe:.1f}%\n"

    msg += f"\n💰 <b>TOTAL: {total_value:.2f}€</b> ({pnl_pct:+.1f}%)\n"

    msg += "\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, t in enumerate(valid_scores.head(5).index, 1):
        if t in current_prices.index:
            px_eur = usd_to_eur(float(current_prices[t]), eur_rate)
            in_pf = "📂" if t in portfolio.get("positions", {}) else "👀"
            msg += f"{i}. {t} @ {px_eur:.2f}€ ({valid_scores[t]:.3f}) {in_pf}\n"

    send_telegram(msg)

    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()
