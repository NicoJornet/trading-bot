"""
APEX v32.2 BASELINE WINNER - PRODUCTION (Telegram + Rotation C + Allocation fix)
==============================================================================
Baseline gagnant (sans freshness / anti-chasse), avec:
- Hard Stop -18% (défensif: stop réduit)
- MFE Trailing: +15% puis -5% depuis peak
- Rotation "C": (score faible OU rank trop mauvais) pendant N jours
  + Rotation safe: pas de rotation si trailing actif

Capital: 1500€ initial + 100€/mois DCA
Tracking: portfolio.json + trades_history.json
Telegram: message à chaque run (scheduler à part)
"""

import os
import json
import requests
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 1500
MONTHLY_DCA = 100
COST_PER_TRADE = 1.0  # frais en EUR par ordre

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Positions (régimes VIX)
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

# Indicateurs / score
ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

# Stops
HARD_STOP_PCT = 0.18
MFE_THRESHOLD_PCT = 0.15
TRAILING_PCT = 0.05

# Rotation C
ROTATION_DAYS = 10
BAD_SCORE_THRESHOLD = 1.0      # score "faible"
BAD_RANK_THRESHOLD = 15        # rank "trop mauvais"
ROTATION_SAFE_NO_IF_TRAILING = True

# Exécution/achat
CASH_BUFFER_EUR = 10.0   # on garde toujours un petit buffer cash
MIN_ORDER_EUR = 50.0

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

# (info)
ULTRA_VOLATILE = {"SMCI", "RKLB"}
CRYPTO = {"MSTR", "MARA", "RIOT"}
SEMI = {"AMD", "LRCX", "MU", "AMAT", "ASML"}
TECH = {"APP", "TSLA", "NVDA", "PLTR", "DDOG"}

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


# ============================================================
# FX / VIX / Regime
# ============================================================
def get_eur_usd_rate() -> float:
    """EURUSD close recent, fallback si Yahoo fail."""
    try:
        fx = yf.download("EURUSD=X", period="7d", interval="1d", progress=False, auto_adjust=True)
        if not fx.empty and "Close" in fx.columns:
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 0:
                return rate
    except Exception:
        pass
    return 1.08

def usd_to_eur(amount_usd: float, rate: float) -> float:
    return float(amount_usd) / float(rate)

def eur_to_usd(amount_eur: float, rate: float) -> float:
    return float(amount_eur) * float(rate)

def get_vix() -> float:
    try:
        v = yf.download("^VIX", period="7d", interval="1d", progress=False, auto_adjust=True)
        if not v.empty and "Close" in v.columns:
            return float(v["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0

def get_regime(vix: float):
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "🔴 ULTRA-DÉFENSIF", MAX_POSITIONS_ULTRA_DEFENSIVE
    if vix >= VIX_DEFENSIVE:
        return "🟡 DÉFENSIF", MAX_POSITIONS_DEFENSIVE
    return "🟢 NORMAL", MAX_POSITIONS_NORMAL


# ============================================================
# Stops / Trailing
# ============================================================
def get_stop_loss_pct(defensive: bool) -> float:
    base = HARD_STOP_PCT
    return base * 0.85 if defensive else base

def calculate_stop_price(entry_price_eur: float, stop_pct: float) -> float:
    return float(entry_price_eur) * (1.0 - float(stop_pct))

def check_hard_stop_exit(current_price_eur: float, entry_price_eur: float, stop_price_eur: float):
    if float(current_price_eur) <= float(stop_price_eur):
        loss_pct = (float(current_price_eur) / float(entry_price_eur) - 1.0) * 100.0
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

def check_mfe_trailing_exit(pos: dict, current_price_eur: float, entry_price_eur: float):
    peak = float(pos.get("peak_price_eur", entry_price_eur))
    if current_price_eur > peak:
        peak = float(current_price_eur)
        pos["peak_price_eur"] = peak

    mfe_pct = (peak / float(entry_price_eur) - 1.0)
    drawdown_from_peak = (float(current_price_eur) / peak - 1.0) if peak > 0 else 0.0
    current_gain = (float(current_price_eur) / float(entry_price_eur) - 1.0)

    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT

    details = {
        "trailing_active": trailing_active,
        "mfe_pct": mfe_pct * 100.0,
        "peak_price": peak,
        "drawdown_pct": drawdown_from_peak * 100.0,
        "current_gain_pct": current_gain * 100.0
    }

    if trailing_active and drawdown_from_peak <= -TRAILING_PCT:
        return True, "MFE_TRAILING", details

    return False, None, details


# ============================================================
# Portfolio I/O
# ============================================================
def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # default
    return {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": float(INITIAL_CAPITAL),
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {}
    }

def save_portfolio(portfolio: dict):
    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=4)

def load_trades_history() -> dict:
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
                raw = f.read().strip()
                if not raw:
                    return default_history
                history = json.loads(raw)
                if not isinstance(history, dict):
                    return default_history
                history.setdefault("trades", [])
                history.setdefault("summary", default_history["summary"])
                # backfill summary keys
                for k, v in default_history["summary"].items():
                    history["summary"].setdefault(k, v)
                return history
        except Exception:
            return default_history

    return default_history

def save_trades_history(history: dict):
    with open(TRADES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def log_trade(history: dict, action: str, ticker: str,
              price_usd: float, price_eur: float,
              shares: float, amount_eur: float, eur_rate: float,
              reason: str = "", pnl_eur: float = None, pnl_pct: float = None):
    trade = {
        "id": len(history.get("trades", [])) + 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
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
    if pnl_eur is not None and pnl_pct is not None:
        trade["pnl_eur"] = round(float(pnl_eur), 2)
        trade["pnl_pct"] = round(float(pnl_pct), 2)

    history.setdefault("trades", []).append(trade)

    summary = history.setdefault("summary", {})
    # ensure keys
    for k in ["total_trades","buys","sells","pyramids","winning_trades","losing_trades",
              "total_pnl_eur","total_fees_eur","best_trade_eur","worst_trade_eur","win_rate"]:
        summary.setdefault(k, 0 if "eur" not in k and "rate" not in k else 0.0)

    summary["total_trades"] += 1
    summary["total_fees_eur"] += float(COST_PER_TRADE)

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
            summary["best_trade_eur"] = max(float(summary.get("best_trade_eur", 0.0)), float(pnl_eur))
            summary["worst_trade_eur"] = min(float(summary.get("worst_trade_eur", 0.0)), float(pnl_eur))
            total_closed = int(summary.get("winning_trades", 0)) + int(summary.get("losing_trades", 0))
            if total_closed > 0:
                summary["win_rate"] = round(int(summary["winning_trades"]) / total_closed * 100.0, 1)


# ============================================================
# Telegram
# ============================================================
def send_telegram(message: str) -> bool:
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
# Data download & momentum score
# ============================================================
def get_market_data(tickers, days=200):
    end = datetime.now()
    start = end - timedelta(days=int(days))
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

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(int(period)).mean()

def calculate_momentum_score(close, high, low, volume=None):
    """
    Score momentum 0-10.
    Pondération : 45% distance SMA20, 35% retour 10j, 15% pénalité high 60j, 5% volume relatif.
    NOTE: clamp => score rarement <= 0, d'où rotation "score faible" plutôt que "score<=0".
    """
    needed = max(ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK, 20) + 15
    if len(close) < needed:
        return np.nan

    sma20 = close.rolling(SMA_PERIOD).mean()
    atr = calculate_atr(high, low, close, ATR_PERIOD)

    atr_last = atr.iloc[-1]
    if pd.isna(atr_last) or float(atr_last) <= 0:
        return np.nan

    dist_sma20 = (close.iloc[-1] - sma20.iloc[-1]) / atr_last
    norm_dist_sma20 = min(max(float(dist_sma20), 0.0), 3.0) / 3.0

    retour_10j = close.pct_change(10).iloc[-1]
    norm_retour_10j = min(max(float(retour_10j), 0.0), 0.4) / 0.4

    high60 = high.rolling(HIGH_LOOKBACK).max().iloc[-1]
    dist_high60 = (high60 - close.iloc[-1]) / atr_last
    norm_penalite = min(max(float(dist_high60), 0.0), 5.0) / 5.0
    score_penalite = 1.0 - norm_penalite

    norm_volume = 0.0
    if volume is not None and len(volume.dropna()) >= 20:
        v = float(volume.iloc[-1])
        v_ma = float(volume.rolling(20).mean().iloc[-1])
        if v_ma > 0:
            volume_rel = v / v_ma
            norm_volume = min(max(volume_rel - 1.0, 0.0), 2.0) / 2.0

    score = (
        0.45 * norm_dist_sma20
        + 0.35 * norm_retour_10j
        + 0.15 * score_penalite
        + 0.05 * norm_volume
    ) * 10.0

    return float(score) if not pd.isna(score) else np.nan


# ============================================================
# Allocation logic (fix)
# ============================================================
def allocation_weights(n_slots: int):
    if n_slots <= 1:
        return [1.0]
    if n_slots == 2:
        return [0.60, 0.40]
    if n_slots == 3:
        return [0.50, 0.30, 0.20]
    # fallback equal
    return [1.0 / n_slots] * n_slots


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🚀 APEX v32.2 BASELINE WINNER - PRODUCTION")
    print("=" * 70)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    print(f"📅 {now.strftime('%Y-%m-%d %H:%M')}")
    print("⚙️ Paramètres: Hard Stop -18% (defensif: réduit), MFE +15%/-5%")
    print(f"⚙️ Rotation C: score<{BAD_SCORE_THRESHOLD} OU rank>{BAD_RANK_THRESHOLD} pendant {ROTATION_DAYS}j")
    if ROTATION_SAFE_NO_IF_TRAILING:
        print("⚙️ Rotation safe: pas de rotation si trailing actif")

    portfolio = load_portfolio()
    history = load_trades_history()

    eur_rate = get_eur_usd_rate()
    vix = get_vix()
    regime, max_positions = get_regime(vix)
    defensive = vix >= VIX_DEFENSIVE

    print(f"\n💱 EUR/USD: {eur_rate:.4f}")
    print(f"📊 VIX: {vix:.1f}")
    print(f"📈 Régime: {regime} (max {max_positions} positions)")

    # DCA mensuel
    last_dca = portfolio.get("last_dca_date")
    current_month = now.strftime("%Y-%m")
    if last_dca is None or not str(last_dca).startswith(current_month):
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + float(MONTHLY_DCA)
        portfolio["last_dca_date"] = today
        print(f"\n💰 DCA mensuel: +{MONTHLY_DCA}€")

    # Download market data
    print("\n📥 Téléchargement des données...")
    data = get_market_data(DATABASE, days=220)
    if data is None or data.empty:
        print("❌ Erreur: pas de données")
        send_telegram("❌ APEX v32.2: Erreur téléchargement données")
        return

    # Build current prices + scores
    current_prices_usd = {}
    scores = {}

    for ticker in DATABASE:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                tdf = data[ticker].dropna()
            else:
                tdf = data.dropna()

            if tdf.empty or "Close" not in tdf.columns:
                continue

            close = tdf["Close"].dropna()
            high = tdf["High"].dropna()
            low = tdf["Low"].dropna()
            volume = tdf["Volume"].dropna() if "Volume" in tdf.columns else None

            if len(close) < 60:
                continue

            current_prices_usd[ticker] = float(close.iloc[-1])
            sc = calculate_momentum_score(close, high, low, volume=volume)
            if not np.isnan(sc) and sc > 0:
                scores[ticker] = float(sc)

        except Exception:
            continue

    current_prices_usd = pd.Series(current_prices_usd, dtype=float)
    valid_scores = pd.Series(scores, dtype=float).sort_values(ascending=False)

    print(f"\n📊 {len(valid_scores)} tickers avec score > 0")

    signals = {"sell": [], "buy": [], "force_rotation": []}

    # ============================================================
    # 1) CHECK POSITIONS -> generate SELL (hard stop / trailing / rotation C)
    # ============================================================
    print(f"\n{'='*70}")
    print("📂 VÉRIFICATION DES POSITIONS")
    print(f"{'='*70}")

    positions_to_remove = []

    for ticker, pos in list(portfolio.get("positions", {}).items()):
        if ticker not in current_prices_usd.index:
            continue

        current_price_usd = float(current_prices_usd[ticker])
        current_price_eur = usd_to_eur(current_price_usd, eur_rate)

        entry_price_eur = float(pos.get("entry_price_eur", current_price_eur))
        shares = float(pos.get("shares", 0.0))
        invested = float(pos.get("amount_invested_eur", pos.get("initial_amount_eur", 0.0)))

        # update peak
        if current_price_eur > float(pos.get("peak_price_eur", entry_price_eur)):
            pos["peak_price_eur"] = float(current_price_eur)

        # score & rank
        current_score = float(valid_scores.get(ticker, 0.0))
        if ticker in valid_scores.index:
            current_rank = list(valid_scores.index).index(ticker) + 1
        else:
            current_rank = 999

        pos["score"] = current_score
        pos["rank"] = current_rank

        # stop
        stop_pct = get_stop_loss_pct(defensive)
        stop_price_eur = calculate_stop_price(entry_price_eur, stop_pct)
        pos["stop_loss_eur"] = stop_price_eur

        value_now = current_price_eur * shares
        pnl_eur = value_now - invested
        pnl_pct = (value_now / invested - 1.0) * 100.0 if invested > 0 else 0.0

        print(f"\n🔹 {ticker}")
        print(f" Prix: {current_price_eur:.2f}€ (entrée: {entry_price_eur:.2f}€)")
        print(f" Détenu: {invested:.2f}€ → {value_now:.2f}€")
        print(f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f" Score: {current_score:.3f} | Rank: #{current_rank}")

        should_sell = False
        sell_reason = ""

        # 1) Hard stop
        hit_hs, hs_reason = check_hard_stop_exit(current_price_eur, entry_price_eur, stop_price_eur)
        if hit_hs:
            should_sell = True
            sell_reason = hs_reason
            print(f" ❌ HARD STOP touché ({stop_price_eur:.2f}€)")

        # 2) MFE trailing
        trailing_active = False
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(pos, current_price_eur, entry_price_eur)
            trailing_active = bool(mfe_details.get("trailing_active", False))
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason
                print(" 📉 MFE TRAILING déclenché!")
                print(f" MFE: +{mfe_details['mfe_pct']:.1f}% | Drawdown: {mfe_details['drawdown_pct']:.1f}%")
            else:
                status = "ACTIF" if trailing_active else "INACTIF"
                print(f" 🎯 Trailing: {status} (MFE: +{mfe_details['mfe_pct']:.1f}%)")

        # 3) Rotation C (score faible OU rank mauvais) pendant ROTATION_DAYS
        if not should_sell:
            # rotation safe: pas si trailing actif
            if ROTATION_SAFE_NO_IF_TRAILING and trailing_active:
                pos["days_bad_rotation"] = 0
                print(" 🔒 Rotation safe: trailing actif -> pas de rotation")
            else:
                bad_score = current_score < BAD_SCORE_THRESHOLD
                bad_rank = current_rank > BAD_RANK_THRESHOLD
                if bad_score or bad_rank:
                    days_bad = int(pos.get("days_bad_rotation", 0)) + 1
                    pos["days_bad_rotation"] = days_bad
                    why = []
                    if bad_score:
                        why.append(f"score<{BAD_SCORE_THRESHOLD}")
                    if bad_rank:
                        why.append(f"rank>{BAD_RANK_THRESHOLD}")
                    print(f" ⚠️ Rotation watch: {' & '.join(why)} depuis {days_bad} jour(s)")

                    if days_bad >= ROTATION_DAYS:
                        # find replacement = best candidate not already held
                        for candidate in valid_scores.index:
                            if candidate not in portfolio["positions"] and candidate not in positions_to_remove:
                                signals["force_rotation"].append({
                                    "ticker": ticker,
                                    "replacement": candidate,
                                    "replacement_score": float(valid_scores[candidate]),
                                    "shares": shares,
                                    "price_eur": current_price_eur,
                                    "value_eur": value_now,
                                    "pnl_eur": pnl_eur,
                                    "pnl_pct": pnl_pct,
                                    "days_bad": days_bad
                                })
                                should_sell = True
                                sell_reason = f"FORCE_ROTATION_{days_bad}j"
                                break
                else:
                    pos["days_bad_rotation"] = 0

        if should_sell:
            signals["sell"].append({
                "ticker": ticker,
                "shares": shares,
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "value_eur": value_now,
                "invested_eur": invested,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct,
                "reason": sell_reason
            })
            positions_to_remove.append(ticker)

    # ============================================================
    # 2) EXECUTE SELLS first (so cash is correct for buys)
    # ============================================================
    for sell in signals["sell"]:
        ticker = sell["ticker"]
        proceeds = max(0.0, float(sell["value_eur"]) - float(COST_PER_TRADE))
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + proceeds

        log_trade(
            history, "SELL", ticker,
            sell["price_usd"], sell["price_eur"],
            sell["shares"], sell["value_eur"],
            eur_rate,
            reason=sell["reason"],
            pnl_eur=sell["pnl_eur"],
            pnl_pct=sell["pnl_pct"]
        )

        if ticker in portfolio.get("positions", {}):
            del portfolio["positions"][ticker]

    # ============================================================
    # 3) BUY OPPORTUNITIES (allocation fixed)
    # ============================================================
    # recompute slots after sells
    n_positions = len(portfolio.get("positions", {}))
    slots_available = max_positions - n_positions
    cash_available = float(portfolio.get("cash", 0.0))

    # buffer
    cash_to_invest = max(0.0, cash_available - CASH_BUFFER_EUR)

    if slots_available > 0 and cash_to_invest >= MIN_ORDER_EUR:
        print(f"\n{'='*70}")
        print(f"🛒 OPPORTUNITÉS D'ACHAT ({slots_available} slots disponibles)")
        print(f"{'='*70}")

        weights = allocation_weights(slots_available)
        planned_buys = []

        # pick candidates in order of score
        for candidate in valid_scores.index:
            if len(planned_buys) >= slots_available:
                break
            if candidate in portfolio["positions"]:
                continue

            # rank is momentum rank in full universe, but buy-slot order is planned_buys index
            price_usd = float(current_prices_usd.get(candidate, np.nan))
            if np.isnan(price_usd) or price_usd <= 0:
                continue
            price_eur = usd_to_eur(price_usd, eur_rate)

            planned_buys.append({
                "ticker": candidate,
                "score": float(valid_scores[candidate]),
                "price_usd": price_usd,
                "price_eur": price_eur,
                "momentum_rank": list(valid_scores.index).index(candidate) + 1
            })

        # allocate based on slots (not momentum rank)
        for i, buy_pick in enumerate(planned_buys):
            if cash_to_invest < MIN_ORDER_EUR:
                break

            w = weights[i] if i < len(weights) else (1.0 / max(1, len(planned_buys)))
            amount = cash_to_invest * w

            # safety: do not exceed remaining investable
            amount = min(amount, cash_to_invest)
            if amount < MIN_ORDER_EUR:
                continue

            # shares
            shares = amount / float(buy_pick["price_eur"])

            stop_pct = get_stop_loss_pct(defensive)
            stop_price = calculate_stop_price(buy_pick["price_eur"], stop_pct)

            signals["buy"].append({
                "ticker": buy_pick["ticker"],
                "rank": i + 1,  # slot rank
                "momentum_rank": buy_pick["momentum_rank"],
                "score": buy_pick["score"],
                "price_usd": buy_pick["price_usd"],
                "price_eur": buy_pick["price_eur"],
                "shares": shares,
                "amount_eur": amount,
                "stop_loss_eur": stop_price,
                "stop_loss_pct": stop_pct * 100.0
            })

            cash_to_invest -= amount

            print(f"\n🟢 ACHETER slot#{i+1}: {buy_pick['ticker']}")
            print(f" Score: {buy_pick['score']:.3f} | MomentumRank: #{buy_pick['momentum_rank']}")
            print(f" Montant: {amount:.2f}€ | Actions: {shares:.4f} | Prix: {buy_pick['price_eur']:.2f}€")
            print(f" Stop: {stop_price:.2f}€ (-{stop_pct*100:.1f}%)")

    # Execute buys
    for buy in signals["buy"]:
        ticker = buy["ticker"]
        cost = float(buy["amount_eur"]) + float(COST_PER_TRADE)
        if float(portfolio.get("cash", 0.0)) < cost:
            continue

        portfolio["cash"] = float(portfolio.get("cash", 0.0)) - cost

        portfolio.setdefault("positions", {})[ticker] = {
            "entry_price_eur": float(buy["price_eur"]),
            "entry_price_usd": float(buy["price_usd"]),
            "entry_date": today,
            "shares": float(buy["shares"]),
            # ✅ amounts for Telegram clarity
            "initial_amount_eur": float(buy["amount_eur"]),
            "amount_invested_eur": float(buy["amount_eur"]),
            "score": float(buy["score"]),
            "peak_price_eur": float(buy["price_eur"]),
            "stop_loss_eur": float(buy["stop_loss_eur"]),
            "rank": int(buy["momentum_rank"]),  # store momentum rank in portfolio
            "pyramided": False,
            "days_bad_rotation": 0
        }

        log_trade(
            history, "BUY", ticker,
            buy["price_usd"], buy["price_eur"],
            buy["shares"], buy["amount_eur"],
            eur_rate,
            reason=f"signal_slot{buy['rank']}_momentum#{buy['momentum_rank']}"
        )

    # ============================================================
    # Portfolio summary
    # ============================================================
    total_positions_value = 0.0
    for ticker, pos in portfolio.get("positions", {}).items():
        if ticker in current_prices_usd.index:
            px_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
            total_positions_value += px_eur * float(pos.get("shares", 0.0))

    total_value = float(portfolio.get("cash", 0.0)) + total_positions_value
    start_date = datetime.strptime(portfolio["start_date"], "%Y-%m-%d")
    months_elapsed = (now.year - start_date.year) * 12 + (now.month - start_date.month)
    total_invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL)) + max(0, months_elapsed) * float(MONTHLY_DCA)

    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0

    save_portfolio(portfolio)
    save_trades_history(history)

    # ============================================================
    # Telegram message
    # ============================================================
    msg = f"📊 <b>APEX v32.2 BASELINE WINNER</b> - {today}\n"
    msg += f"{regime} | VIX: {vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | MFE: +15%/-5%\n\n"

    if signals["sell"] or signals["buy"] or signals["force_rotation"]:
        msg += "🚨 <b>ACTIONS</b>\n\n"

        for rot in signals["force_rotation"]:
            msg += "🔄 <b>ROTATION</b>\n"
            msg += f" {rot['ticker']} → {rot['replacement']}\n"
            msg += f" Motif: score<{BAD_SCORE_THRESHOLD} OU rank>{BAD_RANK_THRESHOLD} ({rot['days_bad']}j)\n\n"

        for sell in signals["sell"]:
            msg += f"🔴 <b>VENDRE {sell['ticker']}</b>\n"
            msg += f" Actions: {sell['shares']:.4f}\n"
            msg += f" Montant: ~{sell['value_eur']:.2f}€\n"
            msg += f" Raison: {sell['reason']}\n"
            msg += f" PnL: {sell['pnl_eur']:+.2f}€ ({sell['pnl_pct']:+.1f}%)\n\n"

        # cash restant estimé après exécution (déjà exécuté dans le code)
        cash_now = float(portfolio.get("cash", 0.0))

        for buy in signals["buy"]:
            msg += f"🟢 <b>ACHETER slot#{buy['rank']} {buy['ticker']}</b>\n"
            msg += f" 💶 Montant: <b>{buy['amount_eur']:.2f}€</b>\n"
            msg += f" 📊 Actions: <b>{buy['shares']:.4f}</b>\n"
            msg += f" 💵 Prix: {buy['price_eur']:.2f}€\n"
            msg += f" Stop: {buy['stop_loss_eur']:.2f}€ (-{buy['stop_loss_pct']:.1f}%)\n"
            msg += f" MFE trigger: {buy['price_eur']*1.15:.2f}€ (+15%)\n\n"

        msg += f"💵 <b>CASH restant (après exécution): {cash_now:.2f}€</b>\n\n"
    else:
        msg += "✅ <b>Aucun signal - HOLD</b>\n\n"

    # positions detail: Détenu -> Valeur
    msg += "📂 <b>POSITIONS</b>\n"
    for ticker, pos in portfolio.get("positions", {}).items():
        if ticker in current_prices_usd.index:
            px_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
            entry_eur = float(pos.get("entry_price_eur", px_eur))
            shares = float(pos.get("shares", 0.0))

            invested = float(pos.get("amount_invested_eur", pos.get("initial_amount_eur", 0.0)))
            value_now = px_eur * shares

            pnl_eur = value_now - invested
            pnl_pct = (value_now / invested - 1.0) * 100.0 if invested > 0 else 0.0

            peak = float(pos.get("peak_price_eur", entry_eur))
            mfe_pct = (peak / entry_eur - 1.0) * 100.0 if entry_eur > 0 else 0.0
            trailing_status = "🟢ACTIF" if mfe_pct >= 15.0 else "⚪️"
            emoji = "📈" if pnl_eur >= 0 else "📉"
            rank = pos.get("rank", "?")

            msg += f"{emoji} {ticker} #{rank}\n"
            msg += f" Détenu: <b>{invested:.0f}€</b> → <b>{value_now:.0f}€</b>\n"
            msg += f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%) | Trail: {trailing_status} MFE:+{mfe_pct:.1f}%\n"

    msg += f"\n💰 <b>TOTAL: {total_value:.2f}€</b> ({total_pnl_pct:+.1f}%)\n"

    # top 5 momentum
    msg += "\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, t in enumerate(valid_scores.head(5).index, 1):
        if t in current_prices_usd.index:
            price_eur = usd_to_eur(float(current_prices_usd[t]), eur_rate)
            in_pf = "📂" if t in portfolio.get("positions", {}) else "👀"
            msg += f"{i}. {t} @ {price_eur:.2f}€ ({valid_scores[t]:.3f}) {in_pf}\n"

    send_telegram(msg)

    print(f"\n{'='*70}")
    print("✅ APEX v32.2 terminé")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
