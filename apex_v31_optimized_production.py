"""
APEX v32 BASELINE WINNER — DAILY REPORT (07:00 Paris)
====================================================
Objectif (simple) :
- Tu reçois UN message à 07:00 (heure FR) basé sur les DERNIÈRES DONNÉES DAILY (close US le plus récent).
- Tu décides ensuite de trader (ou pas) à l’ouverture US.

Stratégie (baseline gagnant) :
- Hard Stop: -18% uniforme (défensif: -15.3%)
- MFE Trailing: activé dès +15% (MFE), sortie si -5% depuis le plus haut
- Pas de filtres freshness/anti-chasse
- Force rotation si score <= 0 pendant X jours
- Rotation safe: PAS de rotation forcée si trailing déjà actif (évite de sortir des winners)

Capital :
- 1,500€ initial + 100€/mois DCA
Fichiers :
- portfolio.json
- trades_history.json
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ============================================================
# CONFIGURATION
# ============================================================
TZ_PARIS = ZoneInfo("Europe/Paris")

INITIAL_CAPITAL = 1500
MONTHLY_DCA = 100
COST_PER_TRADE = 1.0  # frais fixe en EUR (simulation)

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# PARAMÈTRES STRAT
# ============================================================
MAX_POSITIONS_NORMAL = 3
MAX_POSITIONS_DEFENSIVE = 2
MAX_POSITIONS_ULTRA_DEFENSIVE = 1

VIX_DEFENSIVE = 25
VIX_ULTRA_DEFENSIVE = 35

ATR_PERIOD = 14
SMA_PERIOD = 20
HIGH_LOOKBACK = 60

HARD_STOP_PCT = 0.18     # -18% uniforme
MFE_THRESHOLD_PCT = 0.15 # trailing activé si MFE >= +15%
TRAILING_PCT = 0.05      # sortie si -5% depuis peak (quand trailing actif)

FORCE_ROTATION_DAYS = 10

# ============================================================
# UNIVERS — 44 TICKERS
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

# (Info seulement)
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
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configuré (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

def get_regime(vix: float):
    if vix >= VIX_ULTRA_DEFENSIVE:
        return "🔴 ULTRA-DÉFENSIF", MAX_POSITIONS_ULTRA_DEFENSIVE
    if vix >= VIX_DEFENSIVE:
        return "🟡 DÉFENSIF", MAX_POSITIONS_DEFENSIVE
    return "🟢 NORMAL", MAX_POSITIONS_NORMAL

def get_stop_loss_pct(defensive: bool = False) -> float:
    # Défensif = stop un peu plus serré
    return HARD_STOP_PCT * 0.85 if defensive else HARD_STOP_PCT

def calculate_stop_price(entry_price: float, stop_pct: float) -> float:
    return entry_price * (1 - stop_pct)

# ============================================================
# DATA DOWNLOAD (DAILY CLOSE)
# ============================================================
def download_daily(tickers, days=240):
    """
    Télécharge en DAILY (interval=1d).
    À 07:00 Paris, la dernière bougie = dernier close US disponible (souvent J-1).
    """
    end = datetime.now(TZ_PARIS).replace(tzinfo=None)
    start = end - timedelta(days=days)
    return yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        progress=False,
        auto_adjust=True,
        threads=True,
    )

def get_eur_usd_rate_daily() -> float:
    try:
        fx = download_daily(["EURUSD=X"], days=30)
        if fx is None or fx.empty:
            return 1.08
        if isinstance(fx.columns, pd.MultiIndex):
            close = fx["EURUSD=X"]["Close"].dropna()
        else:
            close = fx["Close"].dropna()
        if len(close) == 0:
            return 1.08
        rate = float(close.iloc[-1])
        return rate if rate > 0 else 1.08
    except Exception:
        return 1.08

def usd_to_eur(amount_usd: float, rate: float) -> float:
    return amount_usd / rate

def get_vix_daily() -> float:
    try:
        v = download_daily(["^VIX"], days=60)
        if v is None or v.empty:
            return 20.0
        if isinstance(v.columns, pd.MultiIndex):
            close = v["^VIX"]["Close"].dropna()
        else:
            close = v["Close"].dropna()
        if len(close) == 0:
            return 20.0
        return float(close.iloc[-1])
    except Exception:
        return 20.0

# ============================================================
# SCORE MOMENTUM (0-10)
# ============================================================
def calculate_momentum_score(close, high, low, volume=None,
                             atr_period=14, sma_period=20, high_lookback=60):
    """
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

# ============================================================
# PORTFOLIO / HISTORY
# ============================================================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "currency": "EUR",
        "initial_capital": INITIAL_CAPITAL,
        "monthly_dca": MONTHLY_DCA,
        "cash": float(INITIAL_CAPITAL),
        "start_date": datetime.now(TZ_PARIS).strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {}
    }

def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now(TZ_PARIS).strftime("%Y-%m-%d %H:%M")
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
    if not os.path.exists(TRADES_HISTORY_FILE):
        return default_history
    try:
        with open(TRADES_HISTORY_FILE, "r") as f:
            content = f.read().strip()
        if not content:
            return default_history
        history = json.loads(content)
        if not isinstance(history, dict):
            return default_history
        history.setdefault("trades", [])
        history.setdefault("summary", default_history["summary"])
        for k, v in default_history["summary"].items():
            history["summary"].setdefault(k, v)
        return history
    except Exception as e:
        print(f"⚠️ Erreur chargement trades_history ({e}): reset.")
        return default_history

def save_trades_history(history):
    with open(TRADES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def log_trade(history, action, ticker, price_usd, price_eur, shares, amount_eur, eur_rate,
              reason="", pnl_eur=None, pnl_pct=None):
    history.setdefault("trades", [])
    history.setdefault("summary", {
        "total_trades": 0, "buys": 0, "sells": 0, "pyramids": 0,
        "winning_trades": 0, "losing_trades": 0, "total_pnl_eur": 0.0,
        "total_fees_eur": 0.0, "best_trade_eur": 0.0, "worst_trade_eur": 0.0,
        "win_rate": 0.0
    })

    trade = {
        "id": len(history["trades"]) + 1,
        "date": datetime.now(TZ_PARIS).strftime("%Y-%m-%d"),
        "time": datetime.now(TZ_PARIS).strftime("%H:%M"),
        "action": action,
        "ticker": ticker,
        "shares": round(float(shares), 6),
        "price_usd": round(float(price_usd), 4),
        "price_eur": round(float(price_eur), 4),
        "amount_eur": round(float(amount_eur), 2),
        "fee_eur": COST_PER_TRADE,
        "eur_usd_rate": round(float(eur_rate), 6),
        "reason": reason,
    }
    if pnl_eur is not None:
        trade["pnl_eur"] = round(float(pnl_eur), 2)
        trade["pnl_pct"] = round(float(pnl_pct), 2)

    history["trades"].append(trade)

    s = history["summary"]
    s["total_trades"] += 1
    s["total_fees_eur"] += COST_PER_TRADE

    if action == "BUY":
        s["buys"] += 1
    elif action == "SELL":
        s["sells"] += 1
        if pnl_eur is not None:
            pnl_eur = float(pnl_eur)
            s["total_pnl_eur"] += pnl_eur
            if pnl_eur > 0:
                s["winning_trades"] += 1
            else:
                s["losing_trades"] += 1
            s["best_trade_eur"] = max(s.get("best_trade_eur", 0.0), pnl_eur)
            s["worst_trade_eur"] = min(s.get("worst_trade_eur", 0.0), pnl_eur)
            total_closed = s["winning_trades"] + s["losing_trades"]
            if total_closed > 0:
                s["win_rate"] = round(s["winning_trades"] / total_closed * 100, 1)

# ============================================================
# EXIT LOGIC
# ============================================================
def trailing_active_from_pos(pos: dict) -> bool:
    entry = float(pos.get("entry_price_eur", 0.0))
    peak = float(pos.get("peak_price_eur", entry))
    if entry <= 0:
        return False
    mfe = (peak / entry) - 1
    return mfe >= MFE_THRESHOLD_PCT

def check_mfe_trailing_exit(pos: dict, current_price_eur: float, entry_price_eur: float):
    peak_price = float(pos.get("peak_price_eur", entry_price_eur))
    if current_price_eur > peak_price:
        peak_price = current_price_eur
        pos["peak_price_eur"] = peak_price

    mfe_pct = (peak_price / entry_price_eur - 1)
    drawdown_from_peak = (current_price_eur / peak_price - 1)

    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT

    if trailing_active and drawdown_from_peak <= -TRAILING_PCT:
        return True, "MFE_TRAILING", {
            "mfe_pct": mfe_pct * 100,
            "drawdown_pct": drawdown_from_peak * 100,
            "peak_price": peak_price
        }

    return False, None, {
        "trailing_active": trailing_active,
        "mfe_pct": mfe_pct * 100,
        "drawdown_pct": drawdown_from_peak * 100,
        "peak_price": peak_price
    }

def check_hard_stop_exit(current_price_eur: float, entry_price_eur: float, stop_price_eur: float):
    if current_price_eur <= stop_price_eur:
        loss_pct = (current_price_eur / entry_price_eur - 1) * 100
        return True, f"HARD_STOP_{abs(int(loss_pct))}%"
    return False, None

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🚀 APEX v32 BASELINE WINNER — DAILY REPORT")
    print("=" * 70)
    now_paris = datetime.now(TZ_PARIS)
    print(f"📅 Run: {now_paris.strftime('%Y-%m-%d %H:%M')} (Paris)")

    # Load portfolio & history
    portfolio = load_portfolio()
    history = load_trades_history()

    # FX + VIX (daily close)
    eur_rate = get_eur_usd_rate_daily()
    current_vix = get_vix_daily()
    regime, max_positions = get_regime(current_vix)
    defensive = current_vix >= VIX_DEFENSIVE

    print(f"💱 EUR/USD (daily): {eur_rate:.4f}")
    print(f"📊 VIX (daily): {current_vix:.1f}")
    print(f"📈 Régime: {regime} (max {max_positions} pos)")

    today = now_paris.strftime("%Y-%m-%d")

    # DCA mensuel
    last_dca = portfolio.get("last_dca_date")
    current_month = now_paris.strftime("%Y-%m")
    if last_dca is None or not str(last_dca).startswith(current_month):
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + MONTHLY_DCA
        portfolio["last_dca_date"] = today
        print(f"💰 DCA mensuel: +{MONTHLY_DCA}€")

    # Download DAILY data for universe (close)
    print("\n📥 Téléchargement données DAILY (dernier close US disponible)...")
    data = download_daily(DATABASE, days=240)
    if data is None or data.empty:
        print("❌ Erreur: pas de données")
        send_telegram("❌ APEX v32: Erreur téléchargement données")
        return

    # Determine "as of" date (last available close from a liquid ticker like SPY)
    asof_date = None
    try:
        if isinstance(data.columns, pd.MultiIndex) and "SPY" in data.columns.get_level_values(0):
            spy_close = data["SPY"]["Close"].dropna()
            if len(spy_close) > 0:
                asof_date = spy_close.index[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    if asof_date is None:
        # fallback: try first ticker
        for t in DATABASE:
            try:
                if isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0):
                    s = data[t]["Close"].dropna()
                    if len(s) > 0:
                        asof_date = s.index[-1].strftime("%Y-%m-%d")
                        break
            except Exception:
                continue
    asof_date = asof_date or today

    # Compute scores & current prices (last close)
    scores = {}
    current_prices_usd = {}

    for ticker in DATABASE:
        try:
            if not isinstance(data.columns, pd.MultiIndex):
                tdf = data.dropna()
            else:
                if ticker not in data.columns.get_level_values(0):
                    continue
                tdf = data[ticker].dropna()

            close = tdf["Close"].dropna()
            high = tdf["High"].dropna()
            low = tdf["Low"].dropna()
            volume = tdf["Volume"].dropna() if "Volume" in tdf.columns else None

            if len(close) < 80:
                continue

            px = float(close.iloc[-1])
            if px <= 0:
                continue
            current_prices_usd[ticker] = px

            score = calculate_momentum_score(close, high, low, volume)
            if not np.isnan(score) and score > 0:
                scores[ticker] = float(score)

        except Exception:
            continue

    current_prices_usd = pd.Series(current_prices_usd, dtype=float)
    valid_scores = pd.Series(scores, dtype=float).sort_values(ascending=False)

    print(f"\n📊 {len(valid_scores)} tickers avec score > 0")

    # Signals
    signals = {"sell": [], "buy": [], "force_rotation": []}
    positions_to_remove = []

    # ============================================================
    # 1) CHECK POSITIONS
    # ============================================================
    for ticker, pos in list(portfolio.get("positions", {}).items()):
        if ticker not in current_prices_usd.index:
            continue

        current_usd = float(current_prices_usd[ticker])
        current_eur = usd_to_eur(current_usd, eur_rate)

        entry_eur = float(pos.get("entry_price_eur", 0.0))
        shares = float(pos.get("shares", 0.0))
        if entry_eur <= 0 or shares <= 0:
            continue

        # update peak (EUR)
        if current_eur > float(pos.get("peak_price_eur", entry_eur)):
            pos["peak_price_eur"] = current_eur

        stop_pct = get_stop_loss_pct(defensive)
        stop_eur = calculate_stop_price(entry_eur, stop_pct)
        pos["stop_loss_eur"] = stop_eur

        pnl_eur = (current_eur - entry_eur) * shares
        pnl_pct = (current_eur / entry_eur - 1) * 100

        current_score = float(valid_scores.get(ticker, 0.0))
        pos["score"] = current_score
        pos["rank"] = int(list(valid_scores.index).index(ticker) + 1) if ticker in valid_scores.index else 999

        should_sell = False
        sell_reason = ""

        # Hard stop
        hit_hs, hs_reason = check_hard_stop_exit(current_eur, entry_eur, stop_eur)
        if hit_hs:
            should_sell = True
            sell_reason = hs_reason

        # MFE trailing
        if not should_sell:
            hit_mfe, mfe_reason, mfe_details = check_mfe_trailing_exit(pos, current_eur, entry_eur)
            if hit_mfe:
                should_sell = True
                sell_reason = mfe_reason

        # Force rotation (score <= 0) — ROTATION SAFE
        # => on ne force PAS la rotation si trailing déjà actif
        if not should_sell and current_score <= 0:
            if trailing_active_from_pos(pos):
                # trailing actif => on protège le winner, on ne fait pas de rotation forcée
                pos["days_zero_score"] = 0
            else:
                days_zero = int(pos.get("days_zero_score", 0)) + 1
                pos["days_zero_score"] = days_zero
                if days_zero >= FORCE_ROTATION_DAYS:
                    # propose a replacement (best score not in portfolio)
                    for candidate in valid_scores.index:
                        if candidate not in portfolio["positions"]:
                            signals["force_rotation"].append({
                                "ticker": ticker,
                                "replacement": candidate,
                                "replacement_score": float(valid_scores[candidate]),
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
                "price_usd": current_usd,
                "price_eur": current_eur,
                "value_eur": current_eur * shares,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct,
                "reason": sell_reason
            })
            positions_to_remove.append(ticker)

    # ============================================================
    # 2) BUY IDEAS (PLAN) — on calcule des suggestions, mais tu exécutes manuel à l’open
    # ============================================================
    available_cash = float(portfolio.get("cash", 0.0))
    future_positions = len(portfolio.get("positions", {})) - len(positions_to_remove)
    slots_available = max_positions - future_positions

    buy_ideas = []
    tmp_cash = available_cash

    if slots_available > 0 and tmp_cash >= 50 and len(valid_scores) > 0:
        for ticker in valid_scores.index:
            if slots_available <= 0 or tmp_cash < 50:
                break
            if ticker in portfolio["positions"] and ticker not in positions_to_remove:
                continue

            rank = list(valid_scores.index).index(ticker) + 1
            if rank > max_positions:
                continue
            if ticker not in current_prices_usd.index:
                continue

            px_usd = float(current_prices_usd[ticker])
            px_eur = usd_to_eur(px_usd, eur_rate)

            # allocation pondérée 50/30/20
            if max_positions == 1:
                allocation = tmp_cash
            elif max_positions == 2:
                allocation = tmp_cash * (0.60 if rank == 1 else 0.40)
            else:
                allocation = tmp_cash * (0.50 if rank == 1 else 0.30 if rank == 2 else 0.20)

            allocation = min(allocation, max(0.0, tmp_cash - 10.0))
            if allocation < 50:
                continue

            shares = allocation / px_eur
            stop_pct = get_stop_loss_pct(defensive)
            stop_eur = calculate_stop_price(px_eur, stop_pct)

            buy_ideas.append({
                "ticker": ticker,
                "rank": rank,
                "score": float(valid_scores[ticker]),
                "price_eur": px_eur,
                "amount_eur": allocation,
                "shares": shares,
                "stop_eur": stop_eur,
                "stop_pct": stop_pct * 100
            })

            tmp_cash -= allocation
            slots_available -= 1

    # ============================================================
    # SUMMARY PORTFOLIO (mark-to-close)
    # ============================================================
    total_positions_value = 0.0
    for ticker, pos in portfolio.get("positions", {}).items():
        if ticker in current_prices_usd.index:
            px_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
            total_positions_value += px_eur * float(pos.get("shares", 0.0))

    total_value = float(portfolio.get("cash", 0.0)) + total_positions_value

    # Estimation investi (initial + DCA mensuel)
    try:
        start_date = datetime.strptime(portfolio["start_date"], "%Y-%m-%d")
    except Exception:
        start_date = datetime.now(TZ_PARIS).replace(tzinfo=None)

    now_naive = datetime.now(TZ_PARIS).replace(tzinfo=None)
    months_elapsed = (now_naive.year - start_date.year) * 12 + (now_naive.month - start_date.month)
    total_invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL)) + max(0, months_elapsed) * float(MONTHLY_DCA)

    total_pnl = total_value - total_invested
    total_pnl_pct = (total_value / total_invested - 1) * 100 if total_invested > 0 else 0.0

    # Save (positions may have updated peak/score/days_zero/stop)
    save_portfolio(portfolio)
    save_trades_history(history)

    # ============================================================
    # TELEGRAM MESSAGE (REPORT)
    # ============================================================
    msg = f"🕖 <b>APEX v32 BASELINE WINNER</b> — 07:00 Paris\n"
    msg += f"📅 Données (close): <b>{asof_date}</b>\n"
    msg += f"{regime} | VIX: {current_vix:.1f}\n"
    msg += f"💱 EUR/USD: {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | Trail: +15%/-5%\n\n"

    # Sells (plan)
    if signals["sell"]:
        msg += "🔴 <b>SELL (plan)</b>\n"
        for s in signals["sell"]:
            msg += f"• {s['ticker']} — {s['reason']} | PnL {s['pnl_pct']:+.1f}%\n"
        msg += "\n"
    else:
        msg += "✅ <b>Aucun SELL détecté</b>\n\n"

    # Buy ideas (plan)
    if buy_ideas:
        msg += "🟢 <b>BUY (idées / plan)</b>\n"
        for b in buy_ideas:
            msg += f"• #{b['rank']} {b['ticker']} (score {b['score']:.2f}) ~{b['amount_eur']:.0f}€ | stop ~{b['stop_pct']:.0f}%\n"
        msg += "\n"
    else:
        msg += "🟡 <b>Pas d’achat proposé</b> (slots/cash insuffisant ou scores)\n\n"

    # Positions
    msg += "📂 <b>POSITIONS (mark-to-close)</b>\n"
    if portfolio.get("positions"):
        for ticker, pos in portfolio["positions"].items():
            if ticker not in current_prices_usd.index:
                continue
            px_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
            entry = float(pos.get("entry_price_eur", 0.0))
            sh = float(pos.get("shares", 0.0))
            if entry <= 0 or sh <= 0:
                continue

            pnl_pct = (px_eur / entry - 1) * 100
            pnl_eur = (px_eur - entry) * sh
            peak = float(pos.get("peak_price_eur", entry))
            mfe_pct = (peak / entry - 1) * 100
            trail_status = "🟢ACTIF" if mfe_pct >= 15 else "⚪️"
            emoji = "📈" if pnl_pct >= 0 else "📉"

            msg += f"{emoji} {ticker} #{pos.get('rank','?')} | {pnl_pct:+.1f}% | Trail {trail_status} (MFE {mfe_pct:.1f}%)\n"
    else:
        msg += "— aucune position —\n"

    msg += f"\n💰 <b>Total:</b> {total_value:.2f}€ ({total_pnl_pct:+.1f}%)\n"

    # Top 5 momentum
    msg += "\n🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, ticker in enumerate(valid_scores.head(5).index, 1):
        if ticker in current_prices_usd.index:
            px_eur = usd_to_eur(float(current_prices_usd[ticker]), eur_rate)
            in_pf = "📂" if ticker in portfolio.get("positions", {}) else "👀"
            msg += f"{i}. {ticker} ~{px_eur:.2f}€ ({valid_scores[ticker]:.2f}) {in_pf}\n"

    send_telegram(msg)

    print("\n✅ Report envoyé (si Telegram configuré).")
    print(f"   Close utilisé: {asof_date}")

if __name__ == "__main__":
    main()
