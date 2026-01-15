"""
APEX v32 BASELINE WINNER - PRODUCTION (FIXED)
============================================
Baseline gagnant (sans filtres freshness/anti-chasse).
Fixes:
- Pondération dynamique selon slots disponibles (1 slot => 100% cash)
- Telegram: affiche Actions (shares) + Prix + Cash restant estimé
- Rotation safe: pas de force rotation si trailing actif
- Robustesse FX/VIX via yf.download (évite .info instable)

Capital: 1,500€ initial + 100€/mois DCA
Tracking: portfolio.json + trades_history.json
Requirements:
  yfinance>=0.2.0
  pandas>=2.0.0
  numpy>=1.24.0
  requests>=2.28.0
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 1500.0
MONTHLY_DCA = 100.0
COST_PER_TRADE = 1.0  # frais FIXES en EUR (achat ET vente)

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# PARAMÈTRES - BASELINE WINNER
# ============================================================
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25.0
VIX_ULTRA_DEFENSIVE = 35.0

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

HARD_STOP_PCT = 0.18
MFE_THRESHOLD_PCT = 0.15
TRAILING_PCT = 0.05

FORCE_ROTATION_DAYS = 10
MIN_TRADE_EUR = 50.0
CASH_BUFFER_EUR = 10.0  # on garde toujours un petit buffer pour éviter cash négatif

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
# UTILITAIRES SAFE FLOAT
# ============================================================
def _f(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, (pd.Series, pd.DataFrame)):
            # prend la dernière valeur si possible
            if hasattr(x, "iloc") and len(x) > 0:
                return float(x.iloc[-1])
            return float(default)
        return float(x)
    except Exception:
        return float(default)

# ============================================================
# STOP LOSS -18% UNIFORME (avec mode défensif optionnel)
# ============================================================
def get_stop_loss_pct(ticker: str, defensive: bool = False) -> float:
    base = HARD_STOP_PCT
    return base * 0.85 if defensive else base  # ex: 15.3% si défensif

def calculate_stop_price(entry_price: float, stop_pct: float) -> float:
    return float(entry_price) * (1.0 - float(stop_pct))

# ============================================================
# MFE TRAILING STOP
# ============================================================
def check_mfe_trailing_exit(pos: dict, current_price: float, entry_price: float):
    """
    Trailing activé si MFE >= +15%
    Sortie si drawdown depuis peak <= -5%
    """
    entry_price = float(entry_price)
    current_price = float(current_price)

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
            "current_gain_pct": current_gain * 100.0,
        }

    return False, None, {
        "trailing_active": trailing_active,
        "mfe_pct": mfe_pct * 100.0,
        "peak_price": peak_price,
        "drawdown_pct": drawdown_from_peak * 100.0,
        "current_gain_pct": current_gain * 100.0,
    }

def check_hard_stop_exit(current_price: float, entry_price: float, stop_price: float):
    current_price = float(current_price)
    entry_price = float(entry_price)
    stop_price = float(stop_price)

    if current_price <= stop_price:
        loss_pct = (current_price / entry_price - 1.0) * 100.0
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

# ============================================================
# FX / VIX (robustes)
# ============================================================
def get_eur_usd_rate() -> float:
    """
    EURUSD=X : valeur = USD pour 1 EUR.
    Pour convertir USD -> EUR: diviser par ce rate.
    """
    try:
        fx = yf.download("EURUSD=X", period="14d", interval="1d", progress=False, auto_adjust=True)
        if fx is not None and not fx.empty and "Close" in fx.columns:
            rate = _f(fx["Close"].dropna().iloc[-1], default=np.nan)
            if np.isfinite(rate) and rate > 0:
                return float(rate)
    except Exception:
        pass
    return 1.08  # fallback

def usd_to_eur(amount_usd: float, rate: float) -> float:
    rate = float(rate)
    if rate <= 0:
        rate = 1.08
    return float(amount_usd) / rate

def get_vix() -> float:
    try:
        v = yf.download("^VIX", period="14d", interval="1d", progress=False, auto_adjust=True)
        if v is not None and not v.empty and "Close" in v.columns:
            val = _f(v["Close"].dropna().iloc[-1], default=np.nan)
            if np.isfinite(val):
                return float(val)
    except Exception:
        pass
    return 20.0

def get_regime(vix: float):
    vix = float(vix)
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
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": INITIAL_CAPITAL,
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {},
    }

def save_portfolio(portfolio: dict) -> None:
    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
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
            "win_rate": 0.0,
        },
    }
    if not os.path.exists(TRADES_HISTORY_FILE):
        return default_history

    try:
        with open(TRADES_HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return default_history
            hist = json.loads(raw)
            if not isinstance(hist, dict):
                return default_history
            hist.setdefault("trades", [])
            hist.setdefault("summary", default_history["summary"])
            for k, v in default_history["summary"].items():
                hist["summary"].setdefault(k, v)
            return hist
    except Exception:
        return default_history

def save_trades_history(history: dict) -> None:
    with open(TRADES_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def log_trade(history: dict, action: str, ticker: str, price_usd: float, price_eur: float,
              shares: float, amount_eur: float, eur_rate: float,
              reason: str = "", pnl_eur: float | None = None, pnl_pct: float | None = None) -> None:
    history.setdefault("trades", [])
    history.setdefault("summary", {
        "total_trades": 0, "buys": 0, "sells": 0, "pyramids": 0,
        "winning_trades": 0, "losing_trades": 0, "total_pnl_eur": 0.0,
        "total_fees_eur": 0.0, "best_trade_eur": 0.0, "worst_trade_eur": 0.0,
        "win_rate": 0.0
    })

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
        "fee_eur": float(COST_PER_TRADE),
        "eur_usd_rate": round(float(eur_rate), 6),
        "reason": reason,
    }
    if pnl_eur is not None:
        trade["pnl_eur"] = round(float(pnl_eur), 2)
        trade["pnl_pct"] = round(float(pnl_pct), 2)

    history["trades"].append(trade)

    s = history["summary"]
    s["total_trades"] += 1
    s["total_fees_eur"] += float(COST_PER_TRADE)

    if action == "BUY":
        s["buys"] += 1
    elif action == "SELL":
        s["sells"] += 1
        if pnl_eur is not None:
            s["total_pnl_eur"] += float(pnl_eur)
            if pnl_eur > 0:
                s["winning_trades"] += 1
            else:
                s["losing_trades"] += 1
            s["best_trade_eur"] = max(s.get("best_trade_eur", 0.0), float(pnl_eur))
            s["worst_trade_eur"] = min(s.get("worst_trade_eur", 0.0), float(pnl_eur))
            total_closed = s["winning_trades"] + s["losing_trades"]
            if total_closed > 0:
                s["win_rate"] = round(s["winning_trades"] / total_closed * 100.0, 1)

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
# DATA DOWNLOAD
# ============================================================
def get_market_data(tickers: list[str], days: int = 220) -> pd.DataFrame | None:
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
            threads=True,
        )
        return data
    except Exception as e:
        print(f"Erreur download: {e}")
        return None

# ============================================================
# MOMENTUM SCORE
# ============================================================
def calculate_momentum_score(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series | None = None,
                             atr_period: int = 14, sma_period: int = 20, high_lookback: int = 60) -> float:
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
    norm_dist_sma20 = min(max(dist_sma20, 0.0), 3.0) / 3.0

    retour_10j = close.pct_change(10).iloc[-1]
    norm_retour_10j = min(max(retour_10j, 0.0), 0.4) / 0.4

    high60 = high.rolling(high_lookback).max().iloc[-1]
    dist_high60 = (high60 - close.iloc[-1]) / atr_last
    norm_penalite = min(max(dist_high60, 0.0), 5.0) / 5.0
    score_penalite = 1.0 - norm_penalite

    norm_volume = 0.0
    if volume is not None and len(volume.dropna()) >= 20:
        v = float(volume.iloc[-1])
        v_ma = float(volume.rolling(20).mean().iloc[-1])
        if v_ma > 0:
            volume_rel = v / v_ma
            norm_volume = min(max(volume_rel - 1.0, 0.0), 2.0) / 2.0

    score = (
        0.45 * norm_dist_sma20 +
        0.35 * norm_retour_10j +
        0.15 * score_penalite +
        0.05 * norm_volume
    ) * 10.0

    return float(score) if not pd.isna(score) else np.nan

# ============================================================
# PONDÉRATION DYNAMIQUE
# ============================================================
def get_weights_for_n(n: int) -> dict[int, float]:
    """
    Poids par rang (1-index).
    - n=1 => 1.00
    - n=2 => 0.60/0.40
    - n>=3 => 0.50/0.30/0.20 (top3)
    """
    if n <= 1:
        return {1: 1.00}
    if n == 2:
        return {1: 0.60, 2: 0.40}
    return {1: 0.50, 2: 0.30, 3: 0.20}

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🚀 APEX v32 BASELINE WINNER - PRODUCTION (FIXED)")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("⚙️ Hard Stop -18% | MFE Trailing +15%/-5% | Baseline (sans filtres)")

    portfolio = load_portfolio()
    history = load_trades_history()

    # cast safety
    portfolio["cash"] = float(portfolio.get("cash", 0.0))
    portfolio.setdefault("positions", {})

    eur_rate = get_eur_usd_rate()
    current_vix = get_vix()
    regime, max_positions = get_regime(current_vix)
    defensive = current_vix >= VIX_DEFENSIVE

    print(f"\n💱 EUR/USD: {eur_rate:.4f}")
    print(f"📊 VIX: {current_vix:.1f}")
    print(f"📈 Régime: {regime} (max {max_positions} positions)")

    today = datetime.now().strftime("%Y-%m-%d")

    # DCA mensuel (1x par mois)
    last_dca = portfolio.get("last_dca_date")
    current_month = datetime.now().strftime("%Y-%m")
    if last_dca is None or not str(last_dca).startswith(current_month):
        portfolio["cash"] += MONTHLY_DCA
        portfolio["last_dca_date"] = today
        print(f"\n💰 DCA mensuel: +{MONTHLY_DCA:.0f}€")

    # Download market data
    print("\n📥 Téléchargement des données...")
    data = get_market_data(DATABASE)
    if data is None or data.empty:
        print("❌ Erreur: pas de données")
        send_telegram("❌ APEX v32: Erreur téléchargement données")
        return

    # Compute scores and current prices
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

            if tdf is None or tdf.empty:
                continue

            close = tdf["Close"].dropna()
            high = tdf["High"].dropna()
            low = tdf["Low"].dropna()
            volume = tdf["Volume"].dropna() if "Volume" in tdf.columns else None

            if len(close) < (max(ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK) + 2):
                continue

            current_prices[ticker] = float(close.iloc[-1])
            sc = calculate_momentum_score(close, high, low, volume)
            if np.isfinite(sc) and sc > 0:
                scores[ticker] = float(sc)
        except Exception:
            continue

    current_prices = pd.Series(current_prices, dtype=float)
    valid_scores = pd.Series(scores, dtype=float).sort_values(ascending=False)

    print(f"\n📊 {len(valid_scores)} tickers avec score > 0")

    signals = {"sell": [], "buy": [], "force_rotation": []}

    # ============================================================
    # 1) CHECK POSITIONS
    # ============================================================
    print(f"\n{'='*70}")
    print("📂 VÉRIFICATION DES POSITIONS")
    print(f"{'='*70}")

    positions_to_remove = []

    # For cash estimate (telegram): simulate in a separate variable
    cash_estimated = float(portfolio["cash"])

    for ticker, pos in list(portfolio["positions"].items()):
        if ticker not in current_prices.index:
            continue

        current_price_usd = float(current_prices[ticker])
        current_price_eur = usd_to_eur(current_price_usd, eur_rate)

        entry_price_eur = float(pos.get("entry_price_eur", current_price_eur))
        shares = float(pos.get("shares", 0.0))

        # stop
        stop_pct = get_stop_loss_pct(ticker, defensive)
        stop_price_eur = calculate_stop_price(entry_price_eur, stop_pct)
        pos["stop_loss_eur"] = stop_price_eur

        # peak update
        if current_price_eur > float(pos.get("peak_price_eur", entry_price_eur)):
            pos["peak_price_eur"] = current_price_eur

        pnl_eur = (current_price_eur - entry_price_eur) * shares
        pnl_pct = (current_price_eur / entry_price_eur - 1.0) * 100.0

        current_score = float(valid_scores.get(ticker, 0.0))
        pos["score"] = current_score

        if ticker in valid_scores.index:
            pos["rank"] = int(list(valid_scores.index).index(ticker) + 1)
        else:
            pos["rank"] = 999

        print(f"\n🔹 {ticker}")
        print(f" Prix: {current_price_eur:.2f}€ (entrée: {entry_price_eur:.2f}€)")
        print(f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%)")
        print(f" Peak: {float(pos.get('peak_price_eur', entry_price_eur)):.2f}€")
        print(f" Score: {current_score:.3f} | Rank: #{pos['rank']}")

        should_sell = False
        sell_reason = ""

        # CHECK 1: HARD STOP
        hit_hard_stop, hard_stop_reason = check_hard_stop_exit(current_price_eur, entry_price_eur, stop_price_eur)
        if hit_hard_stop:
            should_sell = True
            sell_reason = hard_stop_reason
            print(f" ❌ HARD STOP touché! ({stop_price_eur:.2f}€)")

        # CHECK 2: MFE trailing
        mfe_active = False
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(pos, current_price_eur, entry_price_eur)
            mfe_active = bool(mfe_details.get("trailing_active", False))
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason
                print(" 📉 MFE TRAILING déclenché!")
                print(f" MFE: +{mfe_details['mfe_pct']:.1f}% | Drawdown: {mfe_details['drawdown_pct']:.1f}%")
            else:
                status = "ACTIF" if mfe_active else "INACTIF"
                print(f" 🎯 Trailing: {status} (MFE: +{mfe_details['mfe_pct']:.1f}%)")

        # CHECK 3: Force rotation score<=0 (Rotation safe: pas si trailing actif)
        if not should_sell and current_score <= 0:
            if mfe_active:
                # rotation safe
                pos["days_zero_score"] = 0
                print(" 🛡️ Rotation safe: score<=0 mais trailing actif => pas de rotation.")
            else:
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
                                "days_zero": days_zero,
                            })
                            should_sell = True
                            sell_reason = f"FORCE_ROTATION_{days_zero}j"
                            break
        else:
            pos["days_zero_score"] = 0

        if should_sell:
            value_eur = current_price_eur * shares
            signals["sell"].append({
                "ticker": ticker,
                "shares": shares,
                "price_usd": current_price_usd,
                "price_eur": current_price_eur,
                "value_eur": value_eur,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct,
                "reason": sell_reason,
            })
            positions_to_remove.append(ticker)

            # cash estimate after sell (value - fee)
            cash_estimated += max(0.0, value_eur - COST_PER_TRADE)

    # ============================================================
    # 2) BUY OPPORTUNITIES (pondération dynamique)
    # ============================================================
    # Slots après ventes
    future_positions = len(portfolio["positions"]) - len(positions_to_remove)
    slots_available = max(0, int(max_positions - future_positions))

    # On prépare une shortlist de candidats (top momentum) pour remplir les slots
    buy_candidates = []
    if slots_available > 0 and float(portfolio["cash"]) > (MIN_TRADE_EUR + COST_PER_TRADE + CASH_BUFFER_EUR):
        for ticker in valid_scores.index:
            if len(buy_candidates) >= slots_available:
                break

            if ticker in portfolio["positions"] and ticker not in positions_to_remove:
                continue

            # si c'est un remplacement forcé, on autorise même si rank > max_positions
            is_replacement = any(rot["replacement"] == ticker for rot in signals["force_rotation"])

            rank = int(list(valid_scores.index).index(ticker) + 1)
            if rank > max_positions and not is_replacement:
                continue

            if ticker not in current_prices.index:
                continue

            buy_candidates.append({
                "ticker": ticker,
                "rank": rank,
                "score": float(valid_scores[ticker]),
                "price_usd": float(current_prices[ticker]),
            })

    # Allocation selon nombre réel de buys
    if buy_candidates:
        print(f"\n{'='*70}")
        print(f"🛒 OPPORTUNITÉS D'ACHAT ({len(buy_candidates)} ordre(s) prévu(s) / {slots_available} slot(s))")
        print(f"{'='*70}")

        # cash dispo estimé avant buys = cash actuel + proceeds ventes (estimé)
        # (on reste simple : c’est une estimation, l’exécution réelle est faite plus bas)
        cash_for_buys = cash_estimated
        cash_for_buys = max(0.0, cash_for_buys - CASH_BUFFER_EUR)

        weights = get_weights_for_n(len(buy_candidates))

        for idx, c in enumerate(buy_candidates, start=1):
            rank_local = idx  # on pondère selon l'ordre des buys (1..n), pas le rank global absolu
            w = weights.get(rank_local, 0.0)
            if w <= 0:
                continue

            # budget brut alloué
            allocation = cash_for_buys * w

            # borne: pas plus que cash restant - buffer
            allocation = min(allocation, max(0.0, cash_estimated - CASH_BUFFER_EUR))
            if allocation < (MIN_TRADE_EUR + COST_PER_TRADE):
                continue

            ticker = c["ticker"]
            price_usd = float(c["price_usd"])
            price_eur = usd_to_eur(price_usd, eur_rate)

            shares = allocation / price_eur  # approximatif (frais fixes à part)
            stop_pct = get_stop_loss_pct(ticker, defensive)
            stop_price = calculate_stop_price(price_eur, stop_pct)

            signals["buy"].append({
                "ticker": ticker,
                "rank": int(c["rank"]),
                "score": float(c["score"]),
                "price_usd": price_usd,
                "price_eur": price_eur,
                "shares": shares,
                "amount_eur": allocation,
                "stop_loss_eur": stop_price,
                "stop_loss_pct": stop_pct * 100.0,
            })

            # estimation cash après achat (allocation + fee)
            cash_estimated -= (allocation + COST_PER_TRADE)

            print(f"\n🟢 ACHETER: {ticker} (rank momentum #{c['rank']})")
            print(f" Score: {c['score']:.3f}")
            print(f" Montant: {allocation:.2f}€ | Prix ~{price_eur:.2f}€ | Actions ~{shares:.4f}")
            print(f" Stop: {stop_price:.2f}€ (-{stop_pct*100:.0f}%)")

    # ============================================================
    # EXECUTION ORDERS (mise à jour portfolio.json + trades_history.json)
    # ============================================================
    # VENTES
    for sell in signals["sell"]:
        ticker = sell["ticker"]
        proceeds = max(0.0, float(sell["value_eur"]) - COST_PER_TRADE)
        portfolio["cash"] = float(portfolio["cash"]) + proceeds

        log_trade(
            history=history,
            action="SELL",
            ticker=ticker,
            price_usd=sell["price_usd"],
            price_eur=sell["price_eur"],
            shares=sell["shares"],
            amount_eur=sell["value_eur"],
            eur_rate=eur_rate,
            reason=sell["reason"],
            pnl_eur=sell["pnl_eur"],
            pnl_pct=sell["pnl_pct"],
        )

        if ticker in portfolio["positions"]:
            del portfolio["positions"][ticker]

    # ACHATS
    for buy in signals["buy"]:
        ticker = buy["ticker"]
        cost_total = float(buy["amount_eur"]) + COST_PER_TRADE

        if float(portfolio["cash"]) < cost_total:
            continue  # pas assez de cash

        portfolio["cash"] = float(portfolio["cash"]) - cost_total

        portfolio["positions"][ticker] = {
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
        }

        log_trade(
            history=history,
            action="BUY",
            ticker=ticker,
            price_usd=buy["price_usd"],
            price_eur=buy["price_eur"],
            shares=buy["shares"],
            amount_eur=buy["amount_eur"],
            eur_rate=eur_rate,
            reason=f"signal_rank{buy['rank']}",
        )

    # ============================================================
    # PORTFOLIO SUMMARY
    # ============================================================
    total_positions_value = 0.0
    for t, p in portfolio["positions"].items():
        if t in current_prices.index:
            cur_eur = usd_to_eur(float(current_prices[t]), eur_rate)
            total_positions_value += cur_eur * float(p.get("shares", 0.0))

    total_value = float(portfolio["cash"]) + total_positions_value

    # total invested approximation by months elapsed
    start_date = datetime.strptime(portfolio["start_date"], "%Y-%m-%d")
    months_elapsed = (datetime.now().year - start_date.year) * 12 + (datetime.now().month - start_date.month)
    total_invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL)) + max(0, months_elapsed) * float(MONTHLY_DCA)

    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1.0) * 100.0 if total_invested > 0 else 0.0

    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ PORTFOLIO")
    print(f"{'='*70}")
    print(f"💵 Cash disponible: {float(portfolio['cash']):.2f}€")
    print(f"📈 Valeur positions: {total_positions_value:.2f}€")
    print("────────────────────────────────────")
    print(f"💰 VALEUR TOTALE: {total_value:.2f}€")
    print(f"📊 Total investi: {total_invested:.2f}€")
    print(f"{'📈' if total_pnl >= 0 else '📉'} PnL: {total_pnl:+.2f}€ ({total_pnl_pct:+.1f}%)")

    save_portfolio(portfolio)
    save_trades_history(history)

    # ============================================================
    # TELEGRAM MESSAGE
    # ============================================================
    msg = f"📊 <b>APEX v32 BASELINE WINNER</b> - {today}\n"
    msg += f"{regime} | VIX: {current_vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | MFE: +15%/-5%\n\n"

    if signals["sell"] or signals["buy"] or signals["force_rotation"]:
        msg += "🚨 <b>ACTIONS</b>\n\n"

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
            msg += f" 💵 Prix: {buy['price_eur']:.2f}€\n"
            msg += f" Stop: {buy['stop_loss_eur']:.2f}€ (-18%)\n"
            msg += f" MFE trigger: {buy['price_eur']*1.15:.2f}€ (+15%)\n\n"

        # cash restant estimé après les ordres proposés (approx, inclut fee fixe)
        msg += f"💶 <b>Cash restant estimé après ordres:</b> {cash_estimated:.2f}€\n\n"
    else:
        msg += "✅ <b>Aucun signal - HOLD</b>\n\n"
        msg += f"💶 Cash restant estimé: {float(portfolio['cash']):.2f}€\n\n"

    # Positions
    msg += "📂 <b>POSITIONS</b>\n"
    for t, p in portfolio["positions"].items():
        if t in current_prices.index:
            cur_eur = usd_to_eur(float(current_prices[t]), eur_rate)
            entry_eur = float(p.get("entry_price_eur", cur_eur))
            shares = float(p.get("shares", 0.0))
            pnl_pct = (cur_eur / entry_eur - 1.0) * 100.0
            pnl_eur = (cur_eur - entry_eur) * shares
            mfe_pct = (float(p.get("peak_price_eur", entry_eur)) / entry_eur - 1.0) * 100.0
            trailing_status = "🟢ACTIF" if mfe_pct >= 15.0 else "⚪️"
            emoji = "📈" if pnl_pct >= 0 else "📉"
            msg += f"{emoji} {t} #{p.get('rank', 'N/A')}\n"
            msg += f" PnL: {pnl_eur:+.2f}€ ({pnl_pct:+.1f}%) | Trail: {trailing_status} MFE:+{mfe_pct:.1f}%\n"

    msg += f"\n💰 <b>TOTAL:</b> {total_value:.2f}€ ({total_pnl_pct:+.1f}%)\n"

    # Top 5
    msg += "\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, t in enumerate(valid_scores.head(5).index, start=1):
        if t in current_prices.index:
            price = usd_to_eur(float(current_prices[t]), eur_rate)
            in_pf = "📂" if t in portfolio["positions"] else "👀"
            msg += f"{i}. {t} @ {price:.2f}€ ({float(valid_scores[t]):.3f}) {in_pf}\n"

    send_telegram(msg)

    print(f"\n{'='*70}")
    print("✅ APEX v32 BASELINE WINNER terminé")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
