"""
APEX v32 BASELINE WINNER - PRODUCTION (Fix + Rotation C + 7h message-ready)
==========================================================================
Baseline gagnant (backtest 2015-2025):
- Pas de filtres freshness/anti-chasse
- Hard Stop: -18% uniforme (défensif: -15.3%)
- MFE Trailing: +15% puis sortie si -5% depuis le peak

Ajouts (robustesse prod):
- Rotation C (plus fiable que "score=0"):
    * Si rank devient mauvais (au-delà d'un cutoff) pendant X runs consécutifs
    * Rotation safe: PAS de rotation si trailing actif (MFE>=15%)
- Nettoyage portfolio.json si il contient des clés parasites (trades/summary)
- Sync manuel optionnel via variables d'environnement:
    * APEX_CASH_OVERRIDE_EUR="482.26"
    * APEX_REMOVE_TICKERS="LRCX,TSLA"

Usage:
  python apex_v32_baseline_winner.py

GitHub Actions:
  Planifie-le à 7h heure FR (voir workflow YAML plus bas).
"""

import os
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# =========================
# CONFIG
# =========================
INITIAL_CAPITAL = 1500.0
MONTHLY_DCA = 100.0
COST_PER_TRADE_EUR = 1.0

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Regimes (VIX)
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25.0
VIX_ULTRA_DEFENSIVE = 35.0

# Indicators
ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

# Exits
HARD_STOP_PCT = 0.18
MFE_THRESHOLD_PCT = 0.15
TRAILING_PCT = 0.05

# Rotation C (rank-based)
ROTATION_RANK_CUTOFF = 12     # au-delà de ce rank, on considère "mauvais"
ROTATION_BAD_RANK_DAYS = 5    # nb de runs consécutifs avec rank > cutoff avant rotation
ROTATION_SAFE_IF_TRAILING_ACTIVE = True

# Force rotation legacy (score<=0)
FORCE_ROTATION_DAYS = 10

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


# =========================
# Helpers: FX
# =========================
def get_eur_usd_rate() -> float:
    """
    Rate = USD per 1 EUR (ex: 1.10).
    If unavailable, returns fallback.
    """
    try:
        fx = yf.download("EURUSD=X", period="10d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(fx, pd.DataFrame) and not fx.empty and "Close" in fx.columns:
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 0:
                return rate
    except Exception:
        pass
    return 1.08


def usd_to_eur(amount_usd: float, eurusd: float) -> float:
    # eurusd = USD per EUR
    return float(amount_usd) / float(eurusd)


# =========================
# Portfolio / History IO
# =========================
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def load_portfolio() -> dict:
    base = {
        "currency": "EUR",
        "initial_capital": float(INITIAL_CAPITAL),
        "monthly_dca": float(MONTHLY_DCA),
        "cash": float(INITIAL_CAPITAL),
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {}
    }
    if not os.path.exists(PORTFOLIO_FILE):
        return base

    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return base
    except Exception:
        return base

    # Nettoyage si tu as mélangé portfolio + trades
    for k in ["trades", "summary"]:
        if k in data:
            data.pop(k, None)

    # Merge defaults
    for k, v in base.items():
        if k not in data:
            data[k] = v

    if "positions" not in data or not isinstance(data["positions"], dict):
        data["positions"] = {}

    # Normalisation
    data["cash"] = _safe_float(data.get("cash", base["cash"]), base["cash"])
    data["initial_capital"] = _safe_float(data.get("initial_capital", base["initial_capital"]), base["initial_capital"])
    data["monthly_dca"] = _safe_float(data.get("monthly_dca", base["monthly_dca"]), base["monthly_dca"])

    # Normaliser positions
    for t, pos in list(data["positions"].items()):
        if not isinstance(pos, dict):
            data["positions"].pop(t, None)
            continue
        pos["entry_price_eur"] = _safe_float(pos.get("entry_price_eur", 0.0))
        pos["shares"] = _safe_float(pos.get("shares", 0.0))
        pos["peak_price_eur"] = _safe_float(pos.get("peak_price_eur", pos["entry_price_eur"]))
        pos["days_zero_score"] = int(pos.get("days_zero_score", 0) or 0)
        pos["bad_rank_days"] = int(pos.get("bad_rank_days", 0) or 0)

    return data


def save_portfolio(pf: dict) -> None:
    pf["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)


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
    if not os.path.exists(TRADES_HISTORY_FILE):
        return default_history

    try:
        with open(TRADES_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default_history
            history = json.loads(content)
            if not isinstance(history, dict):
                return default_history
            if "trades" not in history or not isinstance(history["trades"], list):
                history["trades"] = []
            if "summary" not in history or not isinstance(history["summary"], dict):
                history["summary"] = default_history["summary"]
            else:
                for k, v in default_history["summary"].items():
                    if k not in history["summary"]:
                        history["summary"][k] = v
            return history
    except Exception:
        return default_history


def save_trades_history(history: dict) -> None:
    with open(TRADES_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def log_trade(history: dict, action: str, ticker: str,
              price_usd: float, price_eur: float,
              shares: float, amount_eur: float, eurusd: float,
              reason: str = "", pnl_eur: float | None = None, pnl_pct: float | None = None) -> None:
    if "trades" not in history:
        history["trades"] = []

    trade = {
        "id": len(history["trades"]) + 1,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "action": action,
        "ticker": ticker,
        "shares": round(_safe_float(shares), 6),
        "price_usd": round(_safe_float(price_usd), 4),
        "price_eur": round(_safe_float(price_eur), 4),
        "amount_eur": round(_safe_float(amount_eur), 2),
        "fee_eur": float(COST_PER_TRADE_EUR),
        "eur_usd_rate": round(_safe_float(eurusd), 6),
        "reason": reason
    }
    if pnl_eur is not None:
        trade["pnl_eur"] = round(_safe_float(pnl_eur), 2)
        trade["pnl_pct"] = round(_safe_float(pnl_pct), 2)

    history["trades"].append(trade)

    if "summary" not in history or not isinstance(history["summary"], dict):
        history["summary"] = {
            "total_trades": 0, "buys": 0, "sells": 0, "pyramids": 0,
            "winning_trades": 0, "losing_trades": 0, "total_pnl_eur": 0.0,
            "total_fees_eur": 0.0, "best_trade_eur": 0.0, "worst_trade_eur": 0.0,
            "win_rate": 0.0
        }

    s = history["summary"]
    s["total_trades"] += 1
    s["total_fees_eur"] += float(COST_PER_TRADE_EUR)

    if action == "BUY":
        s["buys"] += 1
    elif action == "SELL":
        s["sells"] += 1
        if pnl_eur is not None:
            pnl_eur_f = _safe_float(pnl_eur)
            s["total_pnl_eur"] += pnl_eur_f
            if pnl_eur_f > 0:
                s["winning_trades"] += 1
            else:
                s["losing_trades"] += 1
            s["best_trade_eur"] = max(_safe_float(s.get("best_trade_eur", 0.0)), pnl_eur_f)
            s["worst_trade_eur"] = min(_safe_float(s.get("worst_trade_eur", 0.0)), pnl_eur_f)
            closed = s["winning_trades"] + s["losing_trades"]
            if closed > 0:
                s["win_rate"] = round(s["winning_trades"] / closed * 100.0, 1)
    elif action == "PYRAMID":
        s["pyramids"] += 1


# =========================
# Telegram
# =========================
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


# =========================
# Market Data
# =========================
def get_market_data(tickers: list[str], days: int = 200) -> pd.DataFrame | None:
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        print(f"Erreur download: {e}")
    return None


def get_vix() -> float:
    try:
        v = yf.download("^VIX", period="10d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(v, pd.DataFrame) and not v.empty and "Close" in v.columns:
            return float(v["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0


def get_regime(vix: float) -> tuple[str, int]:
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "🔴 ULTRA-DÉFENSIF", MAX_POSITIONS_ULTRA_DEFENSIVE
    if vix >= VIX_DEFENSIVE:
        return "🟡 DÉFENSIF", MAX_POSITIONS_DEFENSIVE
    return "🟢 NORMAL", MAX_POSITIONS_NORMAL


# =========================
# Scoring
# =========================
def calculate_momentum_score(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series | None = None,
                             atr_period: int = 14, sma_period: int = 20, high_lookback: int = 60) -> float:
    needed = max(atr_period, sma_period, high_lookback, 20) + 15
    if len(close) < needed:
        return float("nan")

    sma20 = close.rolling(sma_period).mean()

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    atr_last = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float("nan")
    if math.isnan(atr_last) or atr_last <= 0:
        return float("nan")

    dist_sma20 = (float(close.iloc[-1]) - float(sma20.iloc[-1])) / atr_last
    norm_dist_sma20 = min(max(dist_sma20, 0.0), 3.0) / 3.0

    retour_10j = float(close.pct_change(10).iloc[-1])
    norm_retour_10j = min(max(retour_10j, 0.0), 0.4) / 0.4

    high60 = float(high.rolling(high_lookback).max().iloc[-1])
    dist_high60 = (high60 - float(close.iloc[-1])) / atr_last
    norm_penalite = min(max(dist_high60, 0.0), 5.0) / 5.0
    score_penalite = 1.0 - norm_penalite

    norm_volume = 0.0
    if volume is not None and len(volume.dropna()) >= 20:
        v = float(volume.iloc[-1])
        v_ma = float(volume.rolling(20).mean().iloc[-1])
        if v_ma > 0:
            volume_rel = v / v_ma
            norm_volume = min(max(volume_rel - 1.0, 0.0), 2.0) / 2.0

    score = (0.45 * norm_dist_sma20 + 0.35 * norm_retour_10j + 0.15 * score_penalite + 0.05 * norm_volume) * 10.0
    return float(score) if not math.isnan(score) else float("nan")


# =========================
# Exits
# =========================
def get_stop_loss_pct(defensive: bool) -> float:
    return HARD_STOP_PCT * 0.85 if defensive else HARD_STOP_PCT


def calculate_stop_price(entry_price_eur: float, stop_pct: float) -> float:
    return float(entry_price_eur) * (1.0 - float(stop_pct))


def check_hard_stop_exit(current_price_eur: float, entry_price_eur: float, stop_price_eur: float) -> tuple[bool, str | None]:
    if current_price_eur <= stop_price_eur:
        loss_pct = (current_price_eur / entry_price_eur - 1.0) * 100.0
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None


def check_mfe_trailing_exit(pos: dict, current_price_eur: float, entry_price_eur: float) -> tuple[bool, str | None, dict]:
    peak_price = _safe_float(pos.get("peak_price_eur", entry_price_eur), entry_price_eur)
    if current_price_eur > peak_price:
        peak_price = current_price_eur
        pos["peak_price_eur"] = peak_price

    mfe_pct = (peak_price / entry_price_eur - 1.0)
    drawdown_from_peak = (current_price_eur / peak_price - 1.0)
    current_gain = (current_price_eur / entry_price_eur - 1.0)

    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT

    if trailing_active and drawdown_from_peak <= -TRAILING_PCT:
        return True, "MFE_TRAILING", {
            "trailing_active": True,
            "mfe_pct": mfe_pct * 100.0,
            "peak_price": peak_price,
            "drawdown_pct": drawdown_from_peak * 100.0,
            "current_gain_pct": current_gain * 100.0
        }

    return False, None, {
        "trailing_active": trailing_active,
        "mfe_pct": mfe_pct * 100.0,
        "peak_price": peak_price,
        "drawdown_pct": drawdown_from_peak * 100.0,
        "current_gain_pct": current_gain * 100.0
    }


# =========================
# Allocation
# =========================
def get_weighted_allocation(rank: int, num_positions: int, total_cash: float) -> float:
    if num_positions <= 1:
        return total_cash
    if num_positions == 2:
        weights = {1: 0.60, 2: 0.40}
    elif num_positions == 3:
        weights = {1: 0.50, 2: 0.30, 3: 0.20}
    else:
        total_weight = sum(range(1, num_positions + 1))
        weights = {i: (num_positions - i + 1) / total_weight for i in range(1, num_positions + 1)}
    return float(total_cash) * float(weights.get(rank, 1.0 / num_positions))


# =========================
# Manual sync helpers
# =========================
def apply_manual_overrides(pf: dict) -> None:
    """
    Optionnel: permet de "sync" sans éditer portfolio.json.
    - APEX_CASH_OVERRIDE_EUR="482.26"
    - APEX_REMOVE_TICKERS="LRCX,TSLA"
    """
    cash_override = os.environ.get("APEX_CASH_OVERRIDE_EUR", "").strip()
    if cash_override:
        pf["cash"] = _safe_float(cash_override, pf.get("cash", INITIAL_CAPITAL))

    rm = os.environ.get("APEX_REMOVE_TICKERS", "").strip()
    if rm:
        for t in [x.strip().upper() for x in rm.split(",") if x.strip()]:
            if t in pf.get("positions", {}):
                pf["positions"].pop(t, None)


# =========================
# MAIN
# =========================
def main() -> None:
    print("=" * 70)
    print("🚀 APEX v32 BASELINE WINNER - PRODUCTION")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("⚙️ Stop: -18% | MFE: +15% then -5% from peak")
    print("⚙️ Baseline winner (pas de filtres freshness/anti-chasse)")
    print("⚙️ Rotation C: rank mauvais X jours, rotation safe si trailing actif")

    pf = load_portfolio()
    hist = load_trades_history()

    # optional manual sync
    apply_manual_overrides(pf)

    eurusd = get_eur_usd_rate()
    vix = get_vix()
    regime, max_pos = get_regime(vix)
    defensive = vix >= VIX_DEFENSIVE

    print(f"\n💱 EUR/USD: {eurusd:.4f}")
    print(f"📊 VIX: {vix:.1f}")
    print(f"📈 Régime: {regime} (max {max_pos} positions)")
    today = datetime.now().strftime("%Y-%m-%d")

    # DCA monthly
    last_dca = pf.get("last_dca_date")
    current_month = datetime.now().strftime("%Y-%m")
    if last_dca is None or not str(last_dca).startswith(current_month):
        pf["cash"] = _safe_float(pf.get("cash", 0.0)) + MONTHLY_DCA
        pf["last_dca_date"] = today
        print(f"\n💰 DCA mensuel: +{MONTHLY_DCA:.0f}€")

    # Market data
    print("\n📥 Téléchargement des données...")
    data = get_market_data(DATABASE, days=220)
    if data is None or data.empty:
        print("❌ Erreur: pas de données")
        send_telegram("❌ APEX v32: Erreur téléchargement données")
        return

    # Scores + prices
    scores: dict[str, float] = {}
    prices_usd: dict[str, float] = {}

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

            if len(close) < (max(ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK) + 20):
                continue

            prices_usd[ticker] = float(close.iloc[-1])

            sc = calculate_momentum_score(close, high, low, volume=volume,
                                          atr_period=ATR_PERIOD, sma_period=SMA_PERIOD, high_lookback=HIGH_LOOKBACK)
            if not math.isnan(sc) and sc > 0:
                scores[ticker] = float(sc)
        except Exception:
            continue

    if not scores:
        print("⚠️ Aucun score valide aujourd’hui.")
        send_telegram(f"⚠️ APEX v32 - {today}: aucun score valide (data issue).")
        return

    ranked = pd.Series(scores).sort_values(ascending=False)
    print(f"\n📊 {len(ranked)} tickers avec score > 0")

    # Signals
    signals = {"sell": [], "buy": [], "force_rotation": []}
    positions_to_remove: list[str] = []

    # =========================
    # 1) CHECK POSITIONS
    # =========================
    print("\n" + "=" * 70)
    print("📂 VÉRIFICATION DES POSITIONS")
    print("=" * 70)

    for ticker, pos in list(pf.get("positions", {}).items()):
        if ticker not in prices_usd:
            continue

        current_price_usd = float(prices_usd[ticker])
        current_price_eur = usd_to_eur(current_price_usd, eurusd)

        entry_price_eur = _safe_float(pos.get("entry_price_eur", 0.0))
        shares = _safe_float(pos.get("shares", 0.0))
        if entry_price_eur <= 0 or shares <= 0:
            continue

        # stop
        stop_pct = get_stop_loss_pct(defensive)
        stop_price_eur = calculate_stop_price(entry_price_eur, stop_pct)
        pos["stop_loss_eur"] = stop_price_eur

        pnl_eur = (current_price_eur - entry_price_eur) * shares
        pnl_pct = (current_price_eur / entry_price_eur - 1.0) * 100.0

        # rank / score
        current_score = float(ranked.get(ticker, 0.0))
        pos["score"] = current_score
        if ticker in ranked.index:
            pos["rank"] = int(list(ranked.index).index(ticker) + 1)
        else:
            pos["rank"] = 999

        print(f"\n🔹 {ticker}")
        print(f" Prix: {current_price_eur:.2f}€ (entrée: {entry_price_eur:.2f}€)")
        print(f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f" Score: {current_score:.3f} | Rank: #{pos['rank']}")

        should_sell = False
        sell_reason = ""

        # (A) Hard stop
        hit_hs, hs_reason = check_hard_stop_exit(current_price_eur, entry_price_eur, stop_price_eur)
        if hit_hs:
            should_sell = True
            sell_reason = hs_reason or "HARD_STOP"
            print(f" ❌ HARD STOP touché ({stop_price_eur:.2f}€)")

        # (B) MFE trailing
        trailing_active_now = False
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(pos, current_price_eur, entry_price_eur)
            trailing_active_now = bool(mfe_details.get("trailing_active", False))
            status = "ACTIF" if trailing_active_now else "INACTIF"
            print(f" 🎯 Trailing: {status} (MFE: +{mfe_details['mfe_pct']:.1f}%)")
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason or "MFE_TRAILING"
                print(f" 📉 MFE TRAILING déclenché (DD: {mfe_details['drawdown_pct']:.1f}%)")

        # (C) Rotation C (rank-based) — mais rotation safe si trailing actif
        if not should_sell:
            rank = int(pos.get("rank", 999) or 999)
            bad_rank_days = int(pos.get("bad_rank_days", 0) or 0)

            is_bad_rank = rank > ROTATION_RANK_CUTOFF
            if is_bad_rank:
                bad_rank_days += 1
                pos["bad_rank_days"] = bad_rank_days
                print(f" 🔁 Rotation C: rank>{ROTATION_RANK_CUTOFF} depuis {bad_rank_days} run(s)")
            else:
                pos["bad_rank_days"] = 0

            if is_bad_rank and bad_rank_days >= ROTATION_BAD_RANK_DAYS:
                if ROTATION_SAFE_IF_TRAILING_ACTIVE and trailing_active_now:
                    # rotation safe: on ne rotate pas si trailing actif
                    print(" 🛡️ Rotation safe: trailing actif → PAS de rotation")
                else:
                    # trouver un remplaçant parmi le top (hors portefeuille)
                    replacement = None
                    for cand in ranked.index:
                        if cand not in pf["positions"] and cand not in positions_to_remove:
                            replacement = cand
                            break
                    if replacement:
                        signals["force_rotation"].append({
                            "ticker": ticker,
                            "replacement": replacement,
                            "replacement_score": float(ranked[replacement]),
                            "reason": f"ROTATION_C_rank>{ROTATION_RANK_CUTOFF}_{bad_rank_days}runs"
                        })
                        should_sell = True
                        sell_reason = f"ROTATION_C_{bad_rank_days}runs"

        # Legacy: score<=0 for X days (rare)
        if not should_sell and current_score <= 0.0:
            dz = int(pos.get("days_zero_score", 0) or 0) + 1
            pos["days_zero_score"] = dz
            if dz >= FORCE_ROTATION_DAYS:
                should_sell = True
                sell_reason = f"FORCE_ROTATION_score0_{dz}d"
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

    # =========================
    # 2) BUY
    # =========================
    available_cash = _safe_float(pf.get("cash", 0.0))
    future_positions = len(pf.get("positions", {})) - len(positions_to_remove)
    slots_available = int(max_pos - future_positions)

    if slots_available > 0 and available_cash > 50.0:
        print("\n" + "=" * 70)
        print(f"🛒 OPPORTUNITÉS D'ACHAT ({slots_available} slot(s))")
        print("=" * 70)

        for ticker in ranked.index:
            if slots_available <= 0 or available_cash < 50.0:
                break
            if ticker in pf["positions"] and ticker not in positions_to_remove:
                continue

            rank = int(list(ranked.index).index(ticker) + 1)
            # on n’achète que dans le "top max_pos" (sauf remplacements de rotation)
            is_replacement = any(r["replacement"] == ticker for r in signals["force_rotation"])
            if rank > max_pos and not is_replacement:
                continue

            px_usd = float(prices_usd.get(ticker, np.nan))
            if not np.isfinite(px_usd) or px_usd <= 0:
                continue
            px_eur = usd_to_eur(px_usd, eurusd)

            allocation = get_weighted_allocation(rank, max_pos, available_cash)
            allocation = min(allocation, max(0.0, available_cash - 10.0))
            if allocation < 50.0:
                continue

            shares = allocation / px_eur
            stop_pct = get_stop_loss_pct(defensive)
            stop_price = calculate_stop_price(px_eur, stop_pct)

            signals["buy"].append({
                "ticker": ticker,
                "rank": rank,
                "score": float(ranked[ticker]),
                "price_usd": px_usd,
                "price_eur": px_eur,
                "shares": shares,
                "amount_eur": allocation,
                "stop_loss_eur": stop_price
            })

            available_cash -= allocation
            slots_available -= 1

            print(f"\n🟢 ACHETER #{rank}: {ticker}")
            print(f" Score: {float(ranked[ticker]):.3f}")
            print(f" Montant: {allocation:.2f}€ | Actions: {shares:.4f}")
            print(f" Stop: {stop_price:.2f}€ (-{stop_pct*100:.1f}%)")

    # =========================
    # EXECUTE (paper)
    # =========================
    for sell in signals["sell"]:
        proceeds = max(0.0, float(sell["value_eur"]) - COST_PER_TRADE_EUR)
        pf["cash"] = _safe_float(pf.get("cash", 0.0)) + proceeds

        log_trade(
            hist, "SELL", sell["ticker"],
            sell["price_usd"], sell["price_eur"],
            sell["shares"], sell["value_eur"], eurusd,
            reason=sell["reason"],
            pnl_eur=sell["pnl_eur"], pnl_pct=sell["pnl_pct"]
        )
        pf["positions"].pop(sell["ticker"], None)

    for buy in signals["buy"]:
        cost = float(buy["amount_eur"]) + COST_PER_TRADE_EUR
        if _safe_float(pf.get("cash", 0.0)) < cost:
            continue
        pf["cash"] = _safe_float(pf.get("cash", 0.0)) - cost

        pf["positions"][buy["ticker"]] = {
            "entry_price_eur": float(buy["price_eur"]),
            "entry_price_usd": float(buy["price_usd"]),
            "entry_date": today,
            "shares": float(buy["shares"]),
            "initial_amount_eur": float(buy["amount_eur"]),
            "amount_invested_eur": float(buy["amount_eur"]),
            "score": float(buy["score"]),
            "peak_price_eur": float(buy["price_eur"]),
            "stop_loss_eur": float(buy["stop_loss_eur"]),
            "rank": int(buy["rank"]),
            "pyramided": False,
            "days_zero_score": 0,
            "bad_rank_days": 0
        }

        log_trade(
            hist, "BUY", buy["ticker"],
            buy["price_usd"], buy["price_eur"],
            buy["shares"], buy["amount_eur"], eurusd,
            reason=f"signal_rank{buy['rank']}"
        )

    # =========================
    # Summary
    # =========================
    total_positions_value = 0.0
    for t, pos in pf["positions"].items():
        px_usd = prices_usd.get(t)
        if px_usd is None:
            continue
        px_eur = usd_to_eur(float(px_usd), eurusd)
        total_positions_value += px_eur * _safe_float(pos.get("shares", 0.0))

    total_value = _safe_float(pf.get("cash", 0.0)) + total_positions_value

    try:
        start_date = datetime.strptime(str(pf.get("start_date")), "%Y-%m-%d")
        months_elapsed = (datetime.now().year - start_date.year) * 12 + (datetime.now().month - start_date.month)
        total_invested = _safe_float(pf.get("initial_capital", INITIAL_CAPITAL)) + max(0, months_elapsed) * MONTHLY_DCA
    except Exception:
        total_invested = _safe_float(pf.get("initial_capital", INITIAL_CAPITAL))

    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0

    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ PORTFOLIO")
    print("=" * 70)
    print(f" 💵 Cash: {float(pf['cash']):.2f}€")
    print(f" 📈 Positions: {total_positions_value:.2f}€")
    print(" ────────────────────────────────────")
    print(f" 💰 TOTAL: {total_value:.2f}€")
    print(f" 📊 Investi: {total_invested:.2f}€")
    print(f" {'📈' if total_pnl >= 0 else '📉'} PnL: {total_pnl:+.2f}€ ({total_pnl_pct:+.1f}%)")

    save_portfolio(pf)
    save_trades_history(hist)
    print("\n💾 Sauvegardé: portfolio.json + trades_history.json")

    # =========================
    # Telegram message
    # =========================
    msg = f"📊 <b>APEX v32 BASELINE WINNER</b> - {today}\n"
    msg += f"{regime} | VIX: {vix:.1f}\n"
    msg += f"💱 EUR/USD: {eurusd:.4f}\n"
    msg += "⚙️ Stop: -18% | MFE: +15%/-5%\n\n"

    if signals["sell"] or signals["buy"] or signals["force_rotation"]:
        msg += "🚨 <b>ACTIONS</b>\n\n"
        for r in signals["force_rotation"]:
            msg += "🔄 <b>ROTATION</b>\n"
            msg += f" {r['ticker']} → {r['replacement']}\n"
            msg += f" Raison: {r['reason']}\n\n"

        for s in signals["sell"]:
            msg += f"🔴 <b>VENDRE {s['ticker']}</b>\n"
            msg += f" Montant: ~{s['value_eur']:.2f}€\n"
            msg += f" Raison: {s['reason']}\n"
            msg += f" PnL: {s['pnl_eur']:+.2f}€ ({s['pnl_pct']:+.1f}%)\n\n"

        for b in signals["buy"]:
              msg += f"🟢 <b>ACHETER #{buy['rank']} {buy['ticker']}</b>\n"
              msg += f" 💶 Montant: <b>{buy['amount_eur']:.2f}€</b>\n"
              msg += f" 📊 Actions: <b>{buy['shares']:.4f}</b>\n"
              msg += f" 💵 Prix: {buy['price_eur']:.2f}€\n"
              msg += f" Stop: {buy['stop_loss_eur']:.2f}€ (-18%)\n"
              msg += f" MFE Trigger: {buy['price_eur']*1.15:.2f}€ (+15%)\n\n"
   
    else:
        msg += "✅ <b>Aucun signal - HOLD</b>\n\n"

    msg += "📂 <b>POSITIONS</b>\n"
    for t, pos in pf["positions"].items():
        px_usd = prices_usd.get(t)
        if px_usd is None:
            continue
        px_eur = usd_to_eur(float(px_usd), eurusd)
        entry = _safe_float(pos.get("entry_price_eur", 0.0))
        sh = _safe_float(pos.get("shares", 0.0))
        pnl_pct = (px_eur / entry - 1.0) * 100.0 if entry > 0 else 0.0
        pnl_eur = (px_eur - entry) * sh
        peak = _safe_float(pos.get("peak_price_eur", entry), entry)
        mfe_pct = (peak / entry - 1.0) * 100.0 if entry > 0 else 0.0
        trailing = "🟢ACTIF" if mfe_pct >= 15.0 else "⚪️"
        msg += f"{'📈' if pnl_pct >= 0 else '📉'} {t} #{pos.get('rank','?')}\n"
        msg += f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%) | Trail: {trailing} MFE:+{mfe_pct:.1f}%\n"

    msg += f"\n💰 <b>TOTAL: {total_value:.2f}€</b> ({total_pnl_pct:+.1f}%)\n\n"
    msg += "🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, t in enumerate(ranked.head(5).index, 1):
        px_usd = prices_usd.get(t)
        if px_usd is None:
            continue
        px_eur = usd_to_eur(float(px_usd), eurusd)
        msg += f"{i}. {t} @ {px_eur:.2f}€ ({float(ranked[t]):.3f}) {'📂' if t in pf['positions'] else '👀'}\n"

    send_telegram(msg)
    print("\n✅ Terminé.")


if __name__ == "__main__":
    main()
