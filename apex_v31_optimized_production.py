"""
APEX v32 BASELINE WINNER - PRODUCTION (Option A: Full-invested)
===============================================================
Baseline gagnant (sans filtres freshness/anti-chasse) + améliorations qualité de vie:

✅ Allocation "slot-based" (Option A)
- 3 slots dispo : 50% / 30% / 20%
- 2 slots dispo : 60% / 40%
- 1 slot dispo  : 100% (moins buffer)

✅ Telegram amélioré
- Affiche "Détenu: X€ → Y€" pour chaque position
- Affiche CASH restant

✅ Rotation C (robuste)
- Rotation si score faible (score < SCORE_ROT_THRESHOLD) OU rank trop mauvais (rank > RANK_ROT_THRESHOLD)
- Et doit persister ROTATION_DAYS jours
- "Rotation safe": on évite la rotation si trailing actif (optionnel et activé par défaut)

✅ Schedules
- Le script peut être lancé à 07:00 heure FR (cron GitHub Actions)
- Tu exécutes ensuite à l'ouverture US si tu veux. Pas besoin de run hourly.

Capital: 1500€ initial + 100€/mois DCA
Tracking: portfolio.json + trades_history.json
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
COST_PER_TRADE = 1.0  # frais en EUR

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# PARAMÈTRES STRATÉGIE - BASELINE WINNER
# ============================================================
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

HARD_STOP_PCT = 0.18            # -18% uniforme
MFE_THRESHOLD_PCT = 0.15        # trailing actif dès +15% MFE
TRAILING_PCT = 0.05             # -5% depuis peak

# Rotation C (score faible OU rank mauvais), persistance N jours
ROTATION_DAYS = 10
SCORE_ROT_THRESHOLD = 1.0       # "score faible" si < 1.0
RANK_ROT_THRESHOLD = 15         # "rank mauvais" si > 15

# "Rotation safe": ne pas forcer une rotation si trailing actif
ROTATION_SAFE_NO_IF_TRAILING_ACTIVE = True

# Buffer cash pour éviter d'être à 0€ (frais, arrondis)
CASH_BUFFER_EUR = 10.0

# ============================================================
# UNIVERS - 44 TICKERS
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
# UTILITAIRES
# ============================================================
def _now_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def _now_time() -> str:
    return datetime.now().strftime("%H:%M")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================
# EUR/USD
# ============================================================
def get_eur_usd_rate() -> float:
    try:
        fx = yf.download("EURUSD=X", period="7d", interval="1d", progress=False, auto_adjust=True)
        if not fx.empty:
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


# ============================================================
# STOP & TRAILING
# ============================================================
def get_stop_loss_pct(defensive: bool) -> float:
    # En défensif, stop un peu plus serré (comme ton code)
    return HARD_STOP_PCT * 0.85 if defensive else HARD_STOP_PCT

def calculate_stop_price(entry_price: float, stop_pct: float) -> float:
    return float(entry_price) * (1.0 - float(stop_pct))

def check_hard_stop_exit(current_price: float, entry_price: float, stop_price: float):
    if float(current_price) <= float(stop_price):
        loss_pct = (float(current_price) / float(entry_price) - 1.0) * 100.0
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

def check_mfe_trailing_exit(pos: dict, current_price: float, entry_price: float):
    peak_price = float(pos.get("peak_price_eur", entry_price))
    if current_price > peak_price:
        peak_price = current_price
        pos["peak_price_eur"] = peak_price

    mfe_pct = (peak_price / entry_price - 1.0)
    drawdown_from_peak = (current_price / peak_price - 1.0)
    current_gain = (current_price / entry_price - 1.0)

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


# ============================================================
# INDICATEURS / SCORE MOMENTUM
# ============================================================
def calculate_momentum_score(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series | None = None) -> float:
    """
    Score momentum 0-10.
    Pondération : 45% distance SMA20, 35% retour 10j, 15% pénalité high 60j, 5% volume relatif.
    """
    needed = max(ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK, 20) + 15
    if len(close) < needed:
        return np.nan

    sma20 = close.rolling(SMA_PERIOD).mean()

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    atr_last = float(atr.iloc[-1])
    if np.isnan(atr_last) or atr_last <= 0:
        return np.nan

    # Distance SMA20 normalisée (clamp)
    dist_sma20 = (float(close.iloc[-1]) - float(sma20.iloc[-1])) / atr_last
    norm_dist_sma20 = clamp(dist_sma20, 0.0, 3.0) / 3.0

    # Retour 10j normalisé (clamp)
    retour_10j = float(close.pct_change(10).iloc[-1])
    norm_retour_10j = clamp(retour_10j, 0.0, 0.4) / 0.4

    # Pénalité distance au high 60j (plus proche du high = meilleur)
    high60 = float(high.rolling(HIGH_LOOKBACK).max().iloc[-1])
    dist_high60 = (high60 - float(close.iloc[-1])) / atr_last
    norm_penalite = clamp(dist_high60, 0.0, 5.0) / 5.0
    score_penalite = 1.0 - norm_penalite

    # Volume relatif (optionnel)
    norm_volume = 0.0
    if volume is not None and len(volume.dropna()) >= 20:
        v = float(volume.iloc[-1])
        v_ma = float(volume.rolling(20).mean().iloc[-1])
        if v_ma > 0:
            volume_rel = v / v_ma
            norm_volume = clamp(volume_rel - 1.0, 0.0, 2.0) / 2.0

    score = (0.45 * norm_dist_sma20 + 0.35 * norm_retour_10j + 0.15 * score_penalite + 0.05 * norm_volume) * 10.0
    return float(score) if not np.isnan(score) else np.nan


# ============================================================
# VIX / RÉGIME
# ============================================================
def get_vix() -> float:
    try:
        v = yf.download("^VIX", period="7d", interval="1d", progress=False, auto_adjust=True)
        if not v.empty:
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
# PORTFOLIO / HISTORY
# ============================================================
def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            try:
                pf = json.load(f)
                if "positions" not in pf:
                    pf["positions"] = {}
                if "cash" not in pf:
                    pf["cash"] = float(INITIAL_CAPITAL)
                return pf
            except json.JSONDecodeError:
                pass

    return {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": float(INITIAL_CAPITAL),
        "start_date": _now_date(),
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
        except Exception:
            return default_history
    return default_history

def save_trades_history(history: dict):
    with open(TRADES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def log_trade(history: dict, action: str, ticker: str,
              price_usd: float, price_eur: float, shares: float,
              amount_eur: float, eur_rate: float,
              reason: str = "", pnl_eur: float | None = None, pnl_pct: float | None = None):
    if "trades" not in history:
        history["trades"] = []

    trade = {
        "id": len(history["trades"]) + 1,
        "date": _now_date(),
        "time": _now_time(),
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

    summary = history.setdefault("summary", {})
    summary.setdefault("total_trades", 0)
    summary.setdefault("buys", 0)
    summary.setdefault("sells", 0)
    summary.setdefault("pyramids", 0)
    summary.setdefault("winning_trades", 0)
    summary.setdefault("losing_trades", 0)
    summary.setdefault("total_pnl_eur", 0.0)
    summary.setdefault("total_fees_eur", 0.0)
    summary.setdefault("best_trade_eur", 0.0)
    summary.setdefault("worst_trade_eur", 0.0)
    summary.setdefault("win_rate", 0.0)

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
            total_closed = summary["winning_trades"] + summary["losing_trades"]
            if total_closed > 0:
                summary["win_rate"] = round(summary["winning_trades"] / total_closed * 100.0, 1)
    elif action == "PYRAMID":
        summary["pyramids"] += 1


# ============================================================
# TELEGRAM
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
# MARKET DATA
# ============================================================
def get_market_data(tickers: list[str], days: int = 220):
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


# ============================================================
# ALLOCATION SLOT-BASED (Option A)
# ============================================================
def allocation_by_slots(available_cash: float, slots_available: int, buy_index: int) -> float:
    """
    Option A: Full-invested selon slots disponibles
    - 3 slots: 50/30/20
    - 2 slots: 60/40
    - 1 slot: 100%
    """
    investable_cash = max(0.0, float(available_cash) - CASH_BUFFER_EUR)
    if investable_cash <= 0:
        return 0.0

    if slots_available <= 1:
        return investable_cash

    if slots_available == 2:
        weights = [0.60, 0.40]
    else:
        weights = [0.50, 0.30, 0.20]

    idx = min(max(int(buy_index), 0), len(weights) - 1)
    return investable_cash * weights[idx]


# ============================================================
# ROTATION LOGIC (C): score faible OU rank mauvais, persistance N jours
# ============================================================
def compute_rotation_condition(ticker: str, current_score: float, current_rank: int) -> bool:
    score_bad = current_score < SCORE_ROT_THRESHOLD
    rank_bad = current_rank > RANK_ROT_THRESHOLD
    return bool(score_bad or rank_bad)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🚀 APEX v32 BASELINE WINNER - PRODUCTION (Option A)")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("⚙️ Paramètres: Hard Stop -18%, MFE Trailing +15%/-5%")
    print("⚙️ Allocation: slot-based full-invested (A)")
    print("⚙️ Rotation: C (score faible OU rank mauvais), persist 10j")

    portfolio = load_portfolio()
    history = load_trades_history()

    eur_rate = get_eur_usd_rate()
    current_vix = get_vix()
    regime, max_positions = get_regime(current_vix)
    defensive = current_vix >= VIX_DEFENSIVE

    print(f"\n💱 EUR/USD: {eur_rate:.4f}")
    print(f"📊 VIX: {current_vix:.1f}")
    print(f"📈 Régime: {regime} (max {max_positions} positions)")

    today = _now_date()

    # DCA mensuel
    last_dca = portfolio.get("last_dca_date")
    current_month = datetime.now().strftime("%Y-%m")
    if last_dca is None or not str(last_dca).startswith(current_month):
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + float(MONTHLY_DCA)
        portfolio["last_dca_date"] = today
        print(f"\n💰 DCA mensuel: +{MONTHLY_DCA}€")

    # Download market data
    print("\n📥 Téléchargement des données...")
    data = get_market_data(DATABASE)
    if data is None or getattr(data, "empty", True):
        print("❌ Erreur: pas de données")
        send_telegram("❌ APEX v32: Erreur téléchargement données")
        return

    # Scores & prices
    scores = {}
    current_prices_usd = {}

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

            if len(close) < 10:
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

    signals = {"sell": [], "buy": [], "rotation": []}

    # ============================================================
    # 1) CHECK POSITIONS
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

        # Peak update
        if current_price_eur > float(pos.get("peak_price_eur", entry_price_eur)):
            pos["peak_price_eur"] = current_price_eur

        # Stops
        stop_pct = get_stop_loss_pct(defensive)
        stop_price_eur = calculate_stop_price(entry_price_eur, stop_pct)
        pos["stop_loss_eur"] = stop_price_eur

        # Invested / Value / PnL
        invested = float(pos.get("amount_invested_eur", pos.get("initial_amount_eur", 0.0)))
        value_now = current_price_eur * shares
        pnl_eur = value_now - invested
        pnl_pct = (value_now / invested - 1.0) * 100.0 if invested > 0 else 0.0

        # Score / Rank
        current_score = float(valid_scores.get(ticker, 0.0))
        pos["score"] = current_score
        current_rank = int(list(valid_scores.index).index(ticker) + 1) if ticker in valid_scores.index else 999
        pos["rank"] = current_rank

        # MFE / trailing status
        mfe_pct = (float(pos.get("peak_price_eur", entry_price_eur)) / entry_price_eur - 1.0) * 100.0
        trailing_active = mfe_pct >= (MFE_THRESHOLD_PCT * 100.0)

        print(f"\n🔹 {ticker}")
        print(f" Prix: {current_price_eur:.2f}€ (entrée: {entry_price_eur:.2f}€)")
        print(f" Détenu: {invested:.2f}€ | Valeur: {value_now:.2f}€")
        print(f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f" Peak: {float(pos.get('peak_price_eur', entry_price_eur)):.2f}€ | MFE: {mfe_pct:+.1f}% | Trailing: {'ACTIF' if trailing_active else 'INACTIF'}")
        print(f" Score: {current_score:.3f} | Rank: #{current_rank}")

        should_sell = False
        sell_reason = ""

        # (1) Hard stop
        hit_hs, hs_reason = check_hard_stop_exit(current_price_eur, entry_price_eur, stop_price_eur)
        if hit_hs:
            should_sell = True
            sell_reason = hs_reason
            print(f" ❌ HARD STOP touché! ({stop_price_eur:.2f}€)")

        # (2) MFE trailing
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(pos, current_price_eur, entry_price_eur)
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason
                print(" 📉 MFE TRAILING déclenché!")
                print(f" MFE: +{mfe_details['mfe_pct']:.1f}% | Drawdown: {mfe_details['drawdown_pct']:.1f}%")
            else:
                status = "ACTIF" if mfe_details["trailing_active"] else "INACTIF"
                print(f" 🎯 Trailing: {status} (MFE: +{mfe_details['mfe_pct']:.1f}%)")

        # (3) Rotation C (persistante) — optionnellement bloquée si trailing actif
        if not should_sell:
            rot_condition = compute_rotation_condition(ticker, current_score, current_rank)

            # Rotation safe
            if ROTATION_SAFE_NO_IF_TRAILING_ACTIVE and trailing_active and rot_condition:
                # on n'incrémente pas, on laisse la position respirer
                pos["rotation_days"] = 0
            else:
                if rot_condition:
                    pos["rotation_days"] = int(pos.get("rotation_days", 0)) + 1
                else:
                    pos["rotation_days"] = 0

                if int(pos.get("rotation_days", 0)) >= ROTATION_DAYS:
                    # trouver remplaçant (premier ticker dans top momentum pas déjà en portefeuille)
                    replacement = None
                    for cand in valid_scores.index:
                        if cand not in portfolio["positions"]:
                            replacement = cand
                            break
                    if replacement:
                        should_sell = True
                        sell_reason = f"ROTATION_C_{ROTATION_DAYS}d"
                        signals["rotation"].append({
                            "ticker": ticker,
                            "replacement": replacement,
                            "replacement_score": float(valid_scores[replacement]),
                            "reason": sell_reason
                        })

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
    # 2) BUY OPPORTUNITIES (slot-based full-invested)
    # ============================================================
    available_cash = float(portfolio.get("cash", 0.0))
    future_positions = len(portfolio.get("positions", {})) - len(positions_to_remove)
    slots_available = int(max_positions - future_positions)

    if slots_available > 0 and available_cash > (50.0 + COST_PER_TRADE):
        print(f"\n{'='*70}")
        print(f"🛒 OPPORTUNITÉS D'ACHAT ({slots_available} slots disponibles)")
        print(f"{'='*70}")

        buy_index = 0
        for ticker in valid_scores.index:
            if slots_available <= 0:
                break

            # Skip si déjà en portefeuille (et pas en cours de vente)
            if ticker in portfolio["positions"] and ticker not in positions_to_remove:
                continue

            # Prix
            if ticker not in current_prices_usd.index:
                continue
            current_price_usd = float(current_prices_usd[ticker])
            if np.isnan(current_price_usd) or current_price_usd <= 0:
                continue
            current_price_eur = usd_to_eur(current_price_usd, eur_rate)

            rank = int(list(valid_scores.index).index(ticker) + 1)

            # Allocation (Option A)
            allocation = allocation_by_slots(available_cash, slots_available, buy_index)
            allocation = min(allocation, max(0.0, available_cash - CASH_BUFFER_EUR))

            # Check mini
            if allocation < 50.0:
                continue

            # Shares (approx)
            shares = allocation / current_price_eur

            # Stop
            stop_pct = get_stop_loss_pct(defensive)
            stop_price = calculate_stop_price(current_price_eur, stop_pct)

            signals["buy"].append({
                "ticker": ticker,
                "rank": rank,
                "score": float(valid_scores[ticker]),
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "shares": shares,
                "amount_eur": allocation,
                "stop_loss_eur": stop_price,
                "stop_loss_pct": stop_pct * 100.0
            })

            # simulate cash consumption for next buys
            available_cash -= allocation
            slots_available -= 1
            buy_index += 1

            print(f"\n🟢 ACHETER #{rank}: {ticker}")
            print(f" Score: {valid_scores[ticker]:.3f}")
            print(f" Montant: {allocation:.2f}€")
            print(f" Actions: {shares:.4f}")
            print(f" Stop: {stop_price:.2f}€ (-{stop_pct*100:.1f}%)")

    # ============================================================
    # EXECUTION ORDERS (mise à jour portfolio + history)
    # ============================================================
    # Ventes
    for sell in signals["sell"]:
        ticker = sell["ticker"]
        proceeds = max(0.0, float(sell["value_eur"]) - COST_PER_TRADE)
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

        if ticker in portfolio["positions"]:
            del portfolio["positions"][ticker]

    # Achats
    for buy in signals["buy"]:
        ticker = buy["ticker"]
        cost = float(buy["amount_eur"]) + COST_PER_TRADE
        if float(portfolio.get("cash", 0.0)) < cost:
            continue

        portfolio["cash"] = float(portfolio.get("cash", 0.0)) - cost

        portfolio["positions"][ticker] = {
            "entry_price_eur": float(buy["price_eur"]),
            "entry_price_usd": float(buy["price_usd"]),
            "entry_date": today,
            "shares": float(buy["shares"]),
            "initial_amount_eur": float(buy["amount_eur"]),
            "amount_invested_eur": float(buy["amount_eur"]),  # utile pour Telegram "détenu"
            "score": float(buy["score"]),
            "peak_price_eur": float(buy["price_eur"]),
            "stop_loss_eur": float(buy["stop_loss_eur"]),
            "rank": int(buy["rank"]),
            "pyramided": False,
            "days_zero_score": 0,
            "rotation_days": 0
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
    total_positions_value = 0.0
    for ticker, pos in portfolio.get("positions", {}).items():
        if ticker in current_prices_usd.index:
            current_price_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
            total_positions_value += current_price_eur * float(pos.get("shares", 0.0))

    total_value = float(portfolio.get("cash", 0.0)) + total_positions_value

    # Invested estimation (simple)
    try:
        start_date = datetime.strptime(portfolio.get("start_date", today), "%Y-%m-%d")
        months_elapsed = (datetime.now().year - start_date.year) * 12 + (datetime.now().month - start_date.month)
        total_invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL)) + max(0, months_elapsed) * float(MONTHLY_DCA)
    except Exception:
        total_invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL))

    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0

    save_portfolio(portfolio)
    save_trades_history(history)

    # ============================================================
    # TELEGRAM MESSAGE
    # ============================================================
    msg = f"📊 <b>APEX v32 BASELINE WINNER</b> - {today}\n"
    msg += f"{regime} | VIX: {current_vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | MFE: +15%/-5%\n\n"

    if signals["sell"] or signals["buy"] or signals["rotation"]:
        msg += "🚨 <b>ACTIONS</b>\n\n"

        for rot in signals["rotation"]:
            msg += "🔄 <b>ROTATION</b>\n"
            msg += f" {rot['ticker']} → {rot['replacement']}\n"
            msg += f" Raison: {rot['reason']}\n\n"

        for sell in signals["sell"]:
            msg += f"🔴 <b>VENDRE {sell['ticker']}</b>\n"
            msg += f" Montant: ~{sell['value_eur']:.2f}€\n"
            msg += f" Raison: {sell['reason']}\n"
            msg += f" PnL: {sell['pnl_eur']:+.2f}€ ({sell['pnl_pct']:+.1f}%)\n\n"

        for buy in signals["buy"]:
            msg += f"🟢 <b>ACHETER #{buy['rank']} {buy['ticker']}</b>\n"
            msg += f" 💶 Montant: <b>{buy['amount_eur']:.2f}€</b>\n"
            msg += f" 📊 Actions: <b>{buy['shares']:.4f}</b>\n"
            msg += f" 💵 Prix: {buy['price_eur']:.2f}€\n"
            msg += f" Stop: {buy['stop_loss_eur']:.2f}€ (-{buy['stop_loss_pct']:.1f}%)\n"
            msg += f" MFE Trigger: {(buy['price_eur'] * (1.0 + MFE_THRESHOLD_PCT)):.2f}€ (+15%)\n\n"
    else:
        msg += "✅ <b>Aucun signal - HOLD</b>\n\n"

    msg += f"💵 <b>CASH: {float(portfolio.get('cash', 0.0)):.2f}€</b>\n\n"

    msg += "📂 <b>POSITIONS</b>\n"
    for ticker, pos in portfolio.get("positions", {}).items():
        if ticker not in current_prices_usd.index:
            continue

        current_price_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
        shares = float(pos.get("shares", 0.0))
        invested = float(pos.get("amount_invested_eur", pos.get("initial_amount_eur", 0.0)))
        value_now = current_price_eur * shares
        pnl_eur = value_now - invested
        pnl_pct = (value_now / invested - 1.0) * 100.0 if invested > 0 else 0.0

        entry_price_eur = float(pos.get("entry_price_eur", current_price_eur))
        mfe_pct = (float(pos.get("peak_price_eur", entry_price_eur)) / entry_price_eur - 1.0) * 100.0
        trailing_status = "🟢ACTIF" if mfe_pct >= 15.0 else "⚪️"
        emoji = "📈" if pnl_eur >= 0 else "📉"

        msg += f"{emoji} {ticker} #{pos.get('rank','?')}\n"
        msg += f" Détenu: <b>{invested:.0f}€</b> → <b>{value_now:.0f}€</b>\n"
        msg += f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%) | Trail: {trailing_status} MFE:+{mfe_pct:.1f}%\n"

    msg += f"\n💰 <b>TOTAL: {total_value:.2f}€</b> ({total_pnl_pct:+.1f}%)\n"

    msg += "\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, t in enumerate(valid_scores.head(5).index, 1):
        if t in current_prices_usd.index:
            price_eur = usd_to_eur(float(current_prices_usd[t]), eur_rate)
            in_pf = "📂" if t in portfolio.get("positions", {}) else "👀"
            msg += f"{i}. {t} @ {price_eur:.2f}€ ({valid_scores[t]:.3f}) {in_pf}\n"

    send_telegram(msg)

    print(f"\n{'='*70}")
    print("✅ APEX v32 BASELINE WINNER terminé")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
