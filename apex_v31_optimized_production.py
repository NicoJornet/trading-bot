"""
APEX v32 BASELINE WINNER - PRODUCTION (FIXED SCHEDULING)
========================================================
Objectif: cohérence "daily" + éviter les faux signaux intraday.

- 07:00 Europe/Paris -> Morning report (INFO only, no trades)
- 22:05 Europe/Paris -> Execute after US close (TRADES + report)

Core logic baseline winner:
- Hard stop -18% (uniform, -15.3% in defensive)
- MFE trailing: activate at +15%, sell if -5% from peak
- No freshness/anti-chase filters
- Force rotation if score <=0 for X days (SAFE: never rotate if trailing active)
"""

import os, json, requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse

# ============================================================
# CONFIG
# ============================================================
TZ = ZoneInfo("Europe/Paris")

INITIAL_CAPITAL = 1500
MONTHLY_DCA = 100
COST_PER_TRADE_EUR = 1.0

PORTFOLIO_FILE = "portfolio.json"
TRADES_HISTORY_FILE = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

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

DATABASE = [
    "NVDA","MSFT","GOOGL","AMZN","AAPL","META","TSLA",
    "AMD","MU","ASML","TSM","LRCX","AMAT",
    "PLTR","APP","CRWD","NET","DDOG","ZS",
    "RKLB","SHOP","ABNB","VRT","SMCI","UBER",
    "MSTR","MARA","RIOT","CEG",
    "LLY","NVO","UNH","JNJ","ABBV",
    "WMT","COST","PG","KO",
    "XOM","CVX",
    "QQQ","SPY","GLD","SLV",
]

# ============================================================
# UTILS
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

def get_eur_usd_rate() -> float:
    try:
        fx = yf.download("EURUSD=X", period="10d", interval="1d",
                         progress=False, auto_adjust=True, threads=True)
        if not fx.empty:
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 0:
                return rate
    except Exception:
        pass
    return 1.08

def usd_to_eur(amount_usd: float, rate: float) -> float:
    return float(amount_usd) / float(rate)

def get_vix_close() -> float:
    # close daily (stable)
    try:
        v = yf.download("^VIX", period="10d", interval="1d",
                        progress=False, auto_adjust=True, threads=True)
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

def get_market_data(tickers, days=220):
    end = datetime.now(TZ).date()
    start = end - timedelta(days=days)
    try:
        data = yf.download(
            tickers,
            start=str(start),
            end=str(end + timedelta(days=1)),
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
# INDICATORS
# ============================================================
def calculate_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_momentum_score(close, high, low, volume=None,
                             atr_period=14, sma_period=20, high_lookback=60):
    needed = max(atr_period, sma_period, high_lookback, 20) + 15
    if len(close) < needed:
        return np.nan

    sma20 = close.rolling(sma_period).mean()
    atr = calculate_atr(high, low, close, atr_period)
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

    score = (0.45 * norm_dist_sma20
             + 0.35 * norm_retour_10j
             + 0.15 * score_penalite
             + 0.05 * norm_volume) * 10
    return float(score) if not pd.isna(score) else np.nan

# ============================================================
# STOPS
# ============================================================
def get_stop_loss_pct(defensive=False):
    return HARD_STOP_PCT * 0.85 if defensive else HARD_STOP_PCT

def calculate_stop_price(entry_price, stop_pct):
    return entry_price * (1 - stop_pct)

def check_hard_stop_exit(current_price, stop_price):
    return current_price <= stop_price

def check_mfe_trailing_exit(pos, current_price, entry_price):
    peak = float(pos.get("peak_price_eur", entry_price))
    if current_price > peak:
        peak = current_price
        pos["peak_price_eur"] = peak

    mfe_pct = (peak / entry_price - 1)
    dd_from_peak = (current_price / peak - 1)
    trailing_active = mfe_pct >= MFE_THRESHOLD_PCT

    if trailing_active and dd_from_peak <= -TRAILING_PCT:
        return True, "MFE_TRAILING", mfe_pct * 100, dd_from_peak * 100
    return False, None, mfe_pct * 100, dd_from_peak * 100

# ============================================================
# STORAGE
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
        "cash": INITIAL_CAPITAL,
        "start_date": datetime.now(TZ).strftime("%Y-%m-%d"),
        "last_dca_date": None,
        "positions": {}
    }

def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

def load_trades_history():
    default = {"trades": [], "summary": {"total_trades":0,"buys":0,"sells":0,
                                        "winning_trades":0,"losing_trades":0,
                                        "total_pnl_eur":0.0,"total_fees_eur":0.0,
                                        "best_trade_eur":0.0,"worst_trade_eur":0.0,
                                        "win_rate":0.0}}
    if os.path.exists(TRADES_HISTORY_FILE):
        try:
            with open(TRADES_HISTORY_FILE, "r") as f:
                h = json.load(f)
            if not isinstance(h, dict): return default
            h.setdefault("trades", [])
            h.setdefault("summary", default["summary"])
            for k,v in default["summary"].items():
                h["summary"].setdefault(k, v)
            return h
        except Exception:
            return default
    return default

def save_trades_history(history):
    with open(TRADES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def log_trade(history, action, ticker, price_eur, shares, amount_eur, reason="", pnl_eur=None, pnl_pct=None):
    history.setdefault("trades", [])
    history.setdefault("summary", {})
    s = history["summary"]
    for k in ["total_trades","buys","sells","winning_trades","losing_trades","total_pnl_eur","total_fees_eur","best_trade_eur","worst_trade_eur","win_rate"]:
        s.setdefault(k, 0 if "eur" not in k and "rate" not in k else 0.0)

    trade = {
        "id": len(history["trades"]) + 1,
        "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "ticker": ticker,
        "price_eur": round(float(price_eur), 4),
        "shares": round(float(shares), 6),
        "amount_eur": round(float(amount_eur), 2),
        "fee_eur": COST_PER_TRADE_EUR,
        "reason": reason
    }
    if pnl_eur is not None:
        trade["pnl_eur"] = round(float(pnl_eur), 2)
        trade["pnl_pct"] = round(float(pnl_pct), 2)

    history["trades"].append(trade)
    s["total_trades"] += 1
    s["total_fees_eur"] += COST_PER_TRADE_EUR

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
            s["best_trade_eur"] = max(float(s.get("best_trade_eur", 0.0)), float(pnl_eur))
            s["worst_trade_eur"] = min(float(s.get("worst_trade_eur", 0.0)), float(pnl_eur))
            closed = s["winning_trades"] + s["losing_trades"]
            if closed > 0:
                s["win_rate"] = round(100.0 * s["winning_trades"] / closed, 1)

# ============================================================
# MAIN LOGIC
# ============================================================
def weighted_alloc(rank, num_pos, total_cash):
    if num_pos == 1:
        return total_cash
    if num_pos == 2:
        w = {1:0.60, 2:0.40}
    elif num_pos == 3:
        w = {1:0.50, 2:0.30, 3:0.20}
    else:
        w = {i: 1.0/num_pos for i in range(1, num_pos+1)}
    return total_cash * w.get(rank, 1.0/num_pos)

def run(mode: str):
    """
    mode:
      - 'morning' : report only (no execution)
      - 'execute' : execute orders + report
    """
    portfolio = load_portfolio()
    history = load_trades_history()

    eur_rate = get_eur_usd_rate()
    vix = get_vix_close()
    regime, max_positions = get_regime(vix)
    defensive = vix >= VIX_DEFENSIVE

    today = datetime.now(TZ).strftime("%Y-%m-%d")

    # DCA monthly
    last_dca = portfolio.get("last_dca_date")
    current_month = datetime.now(TZ).strftime("%Y-%m")
    if last_dca is None or not str(last_dca).startswith(current_month):
        portfolio["cash"] = float(portfolio["cash"]) + MONTHLY_DCA
        portfolio["last_dca_date"] = today

    # Data
    data = get_market_data(DATABASE)
    if data is None or data.empty:
        send_telegram("❌ APEX: Erreur téléchargement données")
        return

    # Scores & prices (daily close)
    scores = {}
    prices = {}

    for t in DATABASE:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.get_level_values(0):
                    continue
                tdf = data[t].dropna()
            else:
                tdf = data.dropna()

            close = tdf["Close"].dropna()
            high = tdf["High"].dropna()
            low  = tdf["Low"].dropna()
            vol  = tdf["Volume"].dropna() if "Volume" in tdf.columns else None
            if len(close) < 80:
                continue

            px_usd = float(close.iloc[-1])
            prices[t] = px_usd

            sc = calculate_momentum_score(close, high, low, vol,
                                         ATR_PERIOD, SMA_PERIOD, HIGH_LOOKBACK)
            if not np.isnan(sc) and sc > 0:
                scores[t] = float(sc)
        except Exception:
            continue

    if not scores:
        send_telegram("⚠️ APEX: Aucun score valide.")
        return

    ranked = pd.Series(scores).sort_values(ascending=False)
    prices = pd.Series(prices, dtype=float)

    # Signals
    sells = []
    buys = []
    to_remove = []

    # 1) Check positions
    for t, pos in list(portfolio["positions"].items()):
        if t not in prices.index:
            continue

        px_eur = usd_to_eur(float(prices[t]), eur_rate)
        entry = float(pos["entry_price_eur"])
        shares = float(pos["shares"])

        # Stop
        stop_pct = get_stop_loss_pct(defensive)
        stop_price = calculate_stop_price(entry, stop_pct)
        pos["stop_loss_eur"] = stop_price

        pnl_eur = (px_eur - entry) * shares
        pnl_pct = (px_eur / entry - 1) * 100

        # Update rank/score
        sc = float(ranked.get(t, 0.0))
        pos["score"] = sc
        pos["rank"] = int(list(ranked.index).index(t) + 1) if t in ranked.index else 999

        # Trailing check
        hit_hard = check_hard_stop_exit(px_eur, stop_price)
        hit_mfe, mfe_reason, mfe_pct, dd_pct = check_mfe_trailing_exit(pos, px_eur, entry)
        trailing_active = mfe_pct >= (MFE_THRESHOLD_PCT * 100)

        # SAFE rotation: never rotate if trailing active
        should_sell = False
        reason = ""

        if hit_hard:
            should_sell = True
            reason = "HARD_STOP"
        elif hit_mfe:
            should_sell = True
            reason = "MFE_TRAILING"
        else:
            # Rotation only if NOT trailing active
            if (not trailing_active) and sc <= 0:
                dz = int(pos.get("days_zero_score", 0)) + 1
                pos["days_zero_score"] = dz
                if dz >= FORCE_ROTATION_DAYS:
                    should_sell = True
                    reason = f"FORCE_ROTATION_{dz}d"
            else:
                pos["days_zero_score"] = 0

        if should_sell:
            sells.append({
                "ticker": t,
                "shares": shares,
                "price_eur": px_eur,
                "value_eur": px_eur * shares,
                "pnl_eur": pnl_eur,
                "pnl_pct": pnl_pct,
                "reason": reason
            })
            to_remove.append(t)

    # 2) Buy candidates
    cash = float(portfolio["cash"])
    future_positions = len(portfolio["positions"]) - len(to_remove)
    slots = max_positions - future_positions

    if slots > 0 and cash > 60:
        for t in ranked.index:
            if slots <= 0 or cash < 60:
                break
            if t in portfolio["positions"] and t not in to_remove:
                continue
            if t not in prices.index:
                continue

            rank = int(list(ranked.index).index(t) + 1)
            if rank > max_positions:
                continue

            px_eur = usd_to_eur(float(prices[t]), eur_rate)
            alloc = weighted_alloc(rank, max_positions, cash)
            alloc = min(alloc, max(0.0, cash - 10.0))
            if alloc < 60:
                continue

            shares = alloc / px_eur
            stop_pct = get_stop_loss_pct(defensive)
            stop_price = calculate_stop_price(px_eur, stop_pct)

            buys.append({
                "ticker": t,
                "rank": rank,
                "score": float(ranked[t]),
                "price_eur": px_eur,
                "shares": shares,
                "amount_eur": alloc,
                "stop_loss_eur": stop_price
            })
            cash -= alloc
            slots -= 1

    # EXECUTE (only in execute mode)
    if mode == "execute":
        # sells first
        for s in sells:
            proceeds = max(0.0, s["value_eur"] - COST_PER_TRADE_EUR)
            portfolio["cash"] = float(portfolio["cash"]) + proceeds

            log_trade(history, "SELL", s["ticker"], s["price_eur"], s["shares"], s["value_eur"],
                      reason=s["reason"], pnl_eur=s["pnl_eur"], pnl_pct=s["pnl_pct"])

            if s["ticker"] in portfolio["positions"]:
                del portfolio["positions"][s["ticker"]]

        # buys
        for b in buys:
            cost = b["amount_eur"] + COST_PER_TRADE_EUR
            if float(portfolio["cash"]) < cost:
                continue
            portfolio["cash"] = float(portfolio["cash"]) - cost

            portfolio["positions"][b["ticker"]] = {
                "entry_price_eur": b["price_eur"],
                "entry_date": today,
                "shares": b["shares"],
                "initial_amount_eur": b["amount_eur"],
                "amount_invested_eur": b["amount_eur"],
                "score": b["score"],
                "peak_price_eur": b["price_eur"],
                "stop_loss_eur": b["stop_loss_eur"],
                "rank": b["rank"],
                "days_zero_score": 0
            }

            log_trade(history, "BUY", b["ticker"], b["price_eur"], b["shares"], b["amount_eur"],
                      reason=f"signal_rank{b['rank']}")

        save_portfolio(portfolio)
        save_trades_history(history)

    # Build message
    total_pos_value = 0.0
    for t, pos in portfolio["positions"].items():
        if t in prices.index:
            px_eur = usd_to_eur(float(prices[t]), eur_rate)
            total_pos_value += px_eur * float(pos["shares"])
    total_value = float(portfolio["cash"]) + total_pos_value

    title = "📌 <b>APEX v32 Morning</b>" if mode == "morning" else "📊 <b>APEX v32 Execute</b>"
    msg = f"{title} - {today}\n"
    msg += f"{regime} | VIX(close): {vix:.1f}\n"
    msg += f"💱 EUR/USD(close): {eur_rate:.4f}\n"
    msg += "⚙️ Stop: -18% | Trail: +15%/-5%\n\n"

    if mode == "execute":
        if sells or buys:
            msg += "🚨 <b>ACTIONS</b>\n\n"
            for s in sells:
                msg += f"🔴 <b>SELL {s['ticker']}</b> ({s['reason']})\n"
                msg += f" PnL: {s['pnl_eur']:+.0f}€ ({s['pnl_pct']:+.1f}%)\n\n"
            for b in buys:
                msg += f"🟢 <b>BUY #{b['rank']} {b['ticker']}</b>\n"
                msg += f" Montant: {b['amount_eur']:.0f}€ | Stop: {b['stop_loss_eur']:.2f}€\n\n"
        else:
            msg += "✅ <b>Aucun ordre</b>\n\n"
    else:
        msg += "🧠 <b>INFO only</b> (pas d'exécution)\n\n"

    msg += f"💰 TOTAL: {total_value:.2f}€ | Cash: {float(portfolio['cash']):.2f}€\n\n"
    msg += "🏆 <b>TOP 5 MOMENTUM</b>\n"
    for i, t in enumerate(ranked.head(5).index, 1):
        px = usd_to_eur(float(prices[t]), eur_rate) if t in prices.index else np.nan
        mark = "📂" if t in portfolio["positions"] else "👀"
        msg += f"{i}. {t} @ {px:.2f}€ ({ranked[t]:.3f}) {mark}\n"

    send_telegram(msg)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["morning","execute"], default="execute",
                   help="morning: report only, execute: trades+report")
    args = p.parse_args()
    run(args.mode)

if __name__ == "__main__":
    main()
