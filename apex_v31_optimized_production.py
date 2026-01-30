from __future__ import annotations

"""
APEX PROD (V33-compatible) — Champion params — YFINANCE ONLY (MIN DIFF)
======================================================================

But:
- Repartir de la logique "prod V33" (Telegram morning message, suivi portfolio.json, historique trades).
- NE PAS lire de parquet/CSV : données uniquement via yfinance.
- Changer le MINIMUM pour appliquer les paramètres Champion fournis.

Paramètres Champion appliqués
----------------------------
Portfolio/Rotation
- MAX_POSITIONS = 3
- EDGE_MULT = 1.00 (trace seulement, pas indispensable sans swap edge)
- CONFIRM_DAYS = 3
- COOLDOWN_DAYS = 1
- FULLY_INVESTED = True (plus d'allocation 50/30/20)

Signals
- SMA200_WIN = 200
- HIGH60_WIN = 60
- Momentum score cross-sectional: R63, R126, R252 pondérés
  WEIGHTS = {r126:0.5, r252:0.3, r63:0.2}

Gates
- BREADTH_THR = 0.55  (si OFF => aucun BUY)
- CORR_WIN = 63
- CORR_THR = 0.65

Exits
- HARD_STOP = -18% (depuis entry)
- MFE_TRIGGER = +15% puis trailing -5% depuis peak
- Trend break: Close < SMA200 => SELL
- Désactivation explicite des exits V33: dead trade / duration / force rotation

Exécution (prod)
----------------
- Les signaux/conditions sont évalués sur le dernier Close disponible de yfinance.
- Le message Telegram sert à déclencher une action manuelle (ou broker) : ce n'est pas un backtest T+1 open.

Variables d'environnement (optionnel)
-------------------------------------
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import yfinance as yf

try:
    import requests
except Exception:
    requests = None


# =============================================================================
# CONFIG (Champion)
# =============================================================================

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR = 100.0
FEE_EUR = 1.0

MAX_POSITIONS = 3
FULLY_INVESTED = True
EDGE_MULT = 1.00
CONFIRM_DAYS = 3
COOLDOWN_DAYS = 1

SMA200_WIN = 200
HIGH60_WIN = 60
R63 = 63
R126 = 126
R252 = 252
WEIGHTS = {"r126": 0.5, "r252": 0.3, "r63": 0.2}

BREADTH_THR = 0.55
CORR_WIN = 63
CORR_THR = 0.65

HARD_STOP_PCT = 0.18
MFE_TRIGGER_PCT = 0.15
TRAIL_FROM_PEAK_PCT = 0.05

# Download lookback (calendar days) so we have enough bars for SMA200 + R252
LOOKBACK_CAL_DAYS = 800

# Universe (U44) — garde le même que ton V33 prod (modifiable si besoin)
UNIVERSE = [
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

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# =============================================================================
# Helpers IO
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            p = json.load(f)
        # defaults / backward compat
        p.setdefault("currency", "EUR")
        p.setdefault("cash", INITIAL_CAPITAL_EUR)
        p.setdefault("initial_capital", INITIAL_CAPITAL_EUR)
        p.setdefault("monthly_dca", MONTHLY_DCA_EUR)
        p.setdefault("positions", {})
        p.setdefault("start_date", datetime.now().strftime("%Y-%m-%d"))
        p.setdefault("last_dca_month", None)
        # champion state
        p.setdefault("confirm_state", {})    # ticker -> consecutive confirms
        p.setdefault("cooldown_until", {})   # ticker -> YYYY-MM-DD
        return p

    return {
        "currency": "EUR",
        "cash": float(INITIAL_CAPITAL_EUR),
        "initial_capital": float(INITIAL_CAPITAL_EUR),
        "monthly_dca": float(MONTHLY_DCA_EUR),
        "positions": {},
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_month": None,
        "confirm_state": {},
        "cooldown_until": {},
        "created_at": _now_str(),
    }


def save_portfolio(p: dict) -> None:
    p["last_updated"] = _now_str()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)


def load_trades() -> dict:
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    return {"trades": [], "summary": {}}


def save_trades(t: dict) -> None:
    with open(TRADES_FILE, "w") as f:
        json.dump(t, f, indent=2)


def append_trade(trades: dict, row: dict) -> None:
    row = dict(row)
    row["id"] = len(trades.get("trades", [])) + 1
    row["ts"] = _now_str()
    trades.setdefault("trades", []).append(row)


# =============================================================================
# Telegram
# =============================================================================

def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or requests is None:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception:
        pass


# =============================================================================
# Data (yfinance only)
# =============================================================================

def download_ohlcv(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=int(lookback_days))
    df = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError("yfinance a renvoyé un DataFrame vide.")

    if not isinstance(df.columns, pd.MultiIndex):
        # yfinance pour un seul ticker peut renvoyer colonnes simples
        # => on normalise en MultiIndex (ticker, field)
        if len(tickers) == 1:
            t = tickers[0]
            df.columns = pd.MultiIndex.from_product([[t], [c.lower() for c in df.columns]])
        else:
            raise RuntimeError("Format yfinance inattendu (pas de MultiIndex).")

    # df peut être (field, ticker) ou (ticker, field) selon group_by
    fields = {"open", "high", "low", "close", "volume"}
    lev0 = set(str(x).lower() for x in df.columns.get_level_values(0))
    if fields.issubset(lev0):
        # (field, ticker) => swap
        df = df.swaplevel(0, 1, axis=1)

    df.columns = pd.MultiIndex.from_tuples(
        [(str(t), str(f).lower()) for (t, f) in df.columns],
        names=["ticker", "field"]
    )
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # keep only needed fields
    keep = [c for c in df.columns if c[1] in fields]
    return df[keep]


def get_eur_usd_rate() -> float:
    """EURUSD=X: USD per 1 EUR"""
    try:
        t = yf.Ticker("EURUSD=X")
        info = t.fast_info if hasattr(t, "fast_info") else {}
        px = None
        if info:
            px = info.get("last_price") or info.get("previous_close")
        if px is None:
            px = t.info.get("regularMarketPrice") or t.info.get("previousClose")
        px = float(px) if px else 1.0
        return px if px > 0 else 1.0
    except Exception:
        return 1.0


def usd_to_eur(price_usd: float, eurusd: float) -> float:
    return float(price_usd) / float(eurusd) if eurusd and eurusd > 0 else float(price_usd)


# =============================================================================
# Indicators / signals
# =============================================================================

def safe_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def sma_last(close: pd.Series, win: int) -> float:
    c = close.dropna()
    if len(c) < win:
        return np.nan
    return float(c.rolling(win).mean().iloc[-1])


def rolling_high_last(high: pd.Series, win: int) -> float:
    h = high.dropna()
    if len(h) < win:
        return np.nan
    return float(h.rolling(win).max().iloc[-1])


def ret_window(close: pd.Series, win: int) -> float:
    c = close.dropna()
    if len(c) < win + 1:
        return np.nan
    prev = float(c.iloc[-win-1])
    last = float(c.iloc[-1])
    if prev <= 0:
        return np.nan
    return last / prev - 1.0


def momentum_score(close: pd.Series) -> float:
    r63 = ret_window(close, R63)
    r126 = ret_window(close, R126)
    r252 = ret_window(close, R252)
    if any(pd.isna(x) for x in [r63, r126, r252]):
        return np.nan
    return WEIGHTS["r63"] * r63 + WEIGHTS["r126"] * r126 + WEIGHTS["r252"] * r252


def corr_max_with_held(ohlcv: pd.DataFrame, d: pd.Timestamp, candidate: str, held: List[str]) -> float:
    if not held:
        return 0.0

    def _rets(ticker: str) -> pd.Series:
        c = ohlcv.get((ticker, "close"), pd.Series(dtype=float)).loc[:d].dropna()
        if len(c) < CORR_WIN + 2:
            return pd.Series(dtype=float)
        return c.pct_change().dropna().tail(CORR_WIN)

    rc = _rets(candidate)
    if rc.empty:
        return 0.0

    vals = []
    for t in held:
        rt = _rets(t)
        if rt.empty:
            continue
        joined = pd.concat([rc, rt], axis=1, join="inner").dropna()
        if joined.shape[0] < 10:
            continue
        vals.append(float(joined.corr().iloc[0, 1]))
    if not vals:
        return 0.0
    return float(np.nanmax(np.abs(vals)))


def is_cooldown_active(portfolio: dict, ticker: str, today_str: str) -> bool:
    until = portfolio.get("cooldown_until", {}).get(ticker)
    if not until:
        return False
    try:
        return pd.to_datetime(today_str) <= pd.to_datetime(until)
    except Exception:
        return False


def allocate_equal(cash: float, slots_left: int) -> float:
    return float(cash) / float(max(1, slots_left))


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("🚀 APEX PROD — Champion params — yfinance only")
    print("=" * 90)
    print(f"🕒 {_now_str()}")

    portfolio = load_portfolio()
    trades = load_trades()

    # DCA monthly (compatible V33)
    today = datetime.now()
    month_key = f"{today.year}-{today.month:02d}"
    if portfolio.get("last_dca_month") != month_key:
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + MONTHLY_DCA_EUR
        portfolio["last_dca_month"] = month_key
        print(f"💰 DCA: +{MONTHLY_DCA_EUR:.2f}€ (month={month_key})")

    # Load OHLCV via yfinance only
    ohlcv = download_ohlcv(UNIVERSE, LOOKBACK_CAL_DAYS)
    d = ohlcv.index.max()
    d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
    print(f"📅 Dernière date OHLCV: {d_str}")

    eurusd = get_eur_usd_rate()
    print(f"💱 EURUSD=X {eurusd:.4f}")

    # Precompute per ticker
    score_map: Dict[str, float] = {}
    sma200_map: Dict[str, float] = {}
    high60_map: Dict[str, float] = {}
    close_usd: Dict[str, float] = {}
    high_usd: Dict[str, float] = {}
    low_usd: Dict[str, float] = {}
    eligible = []

    for t in UNIVERSE:
        c = ohlcv.get((t, "close"), pd.Series(dtype=float)).loc[:d]
        h = ohlcv.get((t, "high"), pd.Series(dtype=float)).loc[:d]
        l = ohlcv.get((t, "low"), pd.Series(dtype=float)).loc[:d]
        if c.dropna().shape[0] < max(SMA200_WIN, R252) + 5:
            continue
        last_c = safe_last(c)
        last_h = safe_last(h)
        last_l = safe_last(l)
        sm = sma_last(c, SMA200_WIN)
        hh = rolling_high_last(h, HIGH60_WIN)
        sc = momentum_score(c)
        if any(pd.isna(x) for x in [last_c, sm, hh, sc]):
            continue

        close_usd[t] = float(last_c)
        high_usd[t] = float(last_h)
        low_usd[t] = float(last_l)
        sma200_map[t] = float(sm)
        high60_map[t] = float(hh)
        score_map[t] = float(sc)
        eligible.append(t)

    # Breadth gate
    if eligible:
        above = sum(1 for t in eligible if close_usd[t] > sma200_map[t])
        breadth = above / len(eligible)
    else:
        breadth = 0.0
    breadth_ok = breadth >= BREADTH_THR

    # Rank top
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}

    # Prices in EUR
    close_eur = {t: usd_to_eur(px, eurusd) for t, px in close_usd.items()}
    high_eur = {t: usd_to_eur(px, eurusd) for t, px in high_usd.items()}
    low_eur = {t: usd_to_eur(px, eurusd) for t, px in low_usd.items()}
    sma200_eur = {t: usd_to_eur(px, eurusd) for t, px in sma200_map.items()}
    high60_eur = {t: usd_to_eur(px, eurusd) for t, px in high60_map.items()}

    # -------------------------
    # SELL logic
    # -------------------------
    sells = []
    positions = portfolio.get("positions", {})

    for t, pos in list(positions.items()):
        px = close_eur.get(t, np.nan)
        if np.isnan(px):
            continue

        entry = float(pos.get("entry_price_eur", pos.get("entry_price", px)))
        shares = float(pos.get("shares", 0.0))
        if entry <= 0 or shares <= 0:
            continue

        peak = float(pos.get("peak_price_eur", entry))
        trough = float(pos.get("trough_price_eur", entry))
        peak = max(peak, float(high_eur.get(t, px)))
        trough = min(trough, float(low_eur.get(t, px)))

        mfe_pct = (peak / entry - 1.0) * 100.0
        mae_pct = (trough / entry - 1.0) * 100.0
        trailing_active = (mfe_pct / 100.0) >= MFE_TRIGGER_PCT

        pos["peak_price_eur"] = peak
        pos["trough_price_eur"] = trough
        pos["mfe_pct"] = mfe_pct
        pos["mae_pct"] = mae_pct
        pos["trailing_active"] = trailing_active
        pos["rank"] = int(rank_map.get(t, 999))
        pos["score"] = float(score_map.get(t, 0.0))

        pnl_pct = (px / entry - 1.0) * 100.0
        pnl_eur = (px - entry) * shares - FEE_EUR

        reason = None

        # Trend break
        sm = sma200_eur.get(t, np.nan)
        if reason is None and not np.isnan(sm) and px < sm:
            reason = "TREND_BREAK_SMA200"

        # Hard stop
        if reason is None and px <= entry * (1.0 - HARD_STOP_PCT):
            reason = "HARD_STOP"

        # Trailing
        if reason is None and trailing_active:
            dd_from_peak = (px / peak - 1.0)
            if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
                reason = "MFE_TRAILING"

        if reason:
            sells.append({
                "ticker": t,
                "price_eur": float(px),
                "shares": float(shares),
                "value_eur": float(px * shares),
                "pnl_eur": float(pnl_eur),
                "pnl_pct": float(pnl_pct),
                "mfe_pct": float(mfe_pct),
                "mae_pct": float(mae_pct),
                "reason": reason,
            })

        positions[t] = pos

    # Execute sells
    for s in sells:
        t = s["ticker"]
        proceeds = float(s["value_eur"]) - FEE_EUR
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + proceeds

        append_trade(trades, {
            "action": "SELL",
            "ticker": t,
            "date": d_str,
            "price_eur": s["price_eur"],
            "shares": s["shares"],
            "amount_eur": s["value_eur"],
            "fee_eur": float(FEE_EUR),
            "reason": s["reason"],
            "pnl_eur": s["pnl_eur"],
            "pnl_pct": s["pnl_pct"],
            "mfe_pct": s["mfe_pct"],
            "mae_pct": s["mae_pct"],
        })

        # cooldown
        cd_until = (pd.to_datetime(d_str) + pd.Timedelta(days=COOLDOWN_DAYS)).strftime("%Y-%m-%d")
        portfolio.setdefault("cooldown_until", {})[t] = cd_until
        portfolio.setdefault("confirm_state", {}).pop(t, None)

        # remove pos
        portfolio.get("positions", {}).pop(t, None)

    # -------------------------
    # BUY logic (Champion)
    # -------------------------
    buys = []
    held = set(portfolio.get("positions", {}).keys())
    slots_left = MAX_POSITIONS - len(held)

    # Update confirm state from current eligible top ranks (1..MAX_POSITIONS)
    confirm_state = portfolio.get("confirm_state", {})
    top_set = set()

    for t, sc in ranked:
        r = int(rank_map.get(t, 999))
        if r > MAX_POSITIONS:
            continue
        # entry filters
        if t not in close_eur:
            continue
        if close_eur[t] <= sma200_eur.get(t, np.nan):
            continue
        if close_eur[t] < high60_eur.get(t, np.nan):
            continue
        top_set.add(t)

    # reset confirms if not in set
    for t in list(confirm_state.keys()):
        if t not in top_set:
            confirm_state[t] = 0
    for t in top_set:
        confirm_state[t] = int(confirm_state.get(t, 0)) + 1
    portfolio["confirm_state"] = confirm_state

    cash = float(portfolio.get("cash", 0.0))

    if slots_left > 0 and cash > 50 and breadth_ok:
        for t, sc in ranked:
            if slots_left <= 0:
                break
            r = int(rank_map.get(t, 999))
            if r > MAX_POSITIONS:
                continue
            if t in held:
                continue
            if is_cooldown_active(portfolio, t, d_str):
                continue

            # entry filters
            if close_eur.get(t, np.nan) <= sma200_eur.get(t, np.nan):
                continue
            if close_eur.get(t, np.nan) < high60_eur.get(t, np.nan):
                continue

            # confirm days
            if int(confirm_state.get(t, 0)) < CONFIRM_DAYS:
                continue

            # corr gate vs held
            cmax = corr_max_with_held(ohlcv, d, t, list(held))
            if cmax >= CORR_THR:
                continue

            px = float(close_eur.get(t, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue

            alloc = allocate_equal(cash, slots_left)
            # keep tiny buffer
            alloc = min(alloc, max(0.0, cash - 10.0))
            if alloc < 50:
                continue

            shares = alloc / px
            cost = alloc + FEE_EUR
            if cost > cash:
                continue

            # execute buy
            cash -= cost
            portfolio["cash"] = cash
            portfolio.setdefault("positions", {})[t] = {
                "entry_date": d_str,
                "entry_price_eur": px,
                "shares": float(shares),
                "initial_amount_eur": float(alloc),
                "amount_invested_eur": float(alloc),
                "peak_price_eur": px,
                "trough_price_eur": px,
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "trailing_active": False,
                "rank": r,
                "score": float(sc),
            }

            buys.append({
                "ticker": t,
                "rank": r,
                "score": float(sc),
                "price_eur": px,
                "amount_eur": float(alloc),
                "shares": float(shares),
                "confirm": int(confirm_state.get(t, 0)),
                "corr_max": float(cmax),
            })

            append_trade(trades, {
                "action": "BUY",
                "ticker": t,
                "date": d_str,
                "price_eur": px,
                "shares": float(shares),
                "amount_eur": float(alloc),
                "fee_eur": float(FEE_EUR),
                "reason": f"BUY_RANK{r}_CONF{CONFIRM_DAYS}",
                "rank": r,
                "score": float(sc),
                "confirm": int(confirm_state.get(t, 0)),
                "corr_max": float(cmax),
            })

            held.add(t)
            slots_left -= 1

    # -------------------------
    # Summary / Telegram
    # -------------------------
    pos_value = 0.0
    lines_pos = []
    for t, pos in portfolio.get("positions", {}).items():
        px = float(close_eur.get(t, np.nan))
        if not np.isfinite(px):
            continue
        entry = float(pos.get("entry_price_eur", px))
        sh = float(pos.get("shares", 0.0))
        val = px * sh
        pos_value += val
        pnl_pct = (px / entry - 1.0) * 100.0 if entry > 0 else 0.0
        mfe = float(pos.get("mfe_pct", 0.0))
        trail = "ON" if bool(pos.get("trailing_active", False)) else "OFF"
        rk = int(pos.get("rank", 999))
        lines_pos.append(f"- {t} (#{rk}) PnL {pnl_pct:+.1f}% | MFE {mfe:+.1f}% | Trail {trail}")

    cash = float(portfolio.get("cash", 0.0))
    total = cash + pos_value

    # invested estimate (compatible)
    start_date = pd.to_datetime(portfolio.get("start_date", d_str))
    months = (today.year - start_date.year) * 12 + (today.month - start_date.month)
    invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL_EUR)) + max(0, months) * float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
    pnl_total = total - invested
    pnl_total_pct = (total / invested - 1.0) * 100.0 if invested > 0 else 0.0

    msg = []
    msg.append(f"APEX PROD — {d_str}")
    msg.append(f"EURUSD=X {eurusd:.4f}")
    msg.append(f"Cash {cash:.2f}€ | Pos {pos_value:.2f}€ | Total {total:.2f}€")
    msg.append(f"Invested~ {invested:.2f}€ | PnL {pnl_total:+.2f}€ ({pnl_total_pct:+.1f}%)")
    msg.append(f"Breadth {breadth:.1%} (thr {int(BREADTH_THR*100)}%) => {'ON' if breadth_ok else 'OFF'}")
    msg.append("")
    msg.append("ACTIONS:")
    if sells:
        for s in sells:
            msg.append(f"SELL {s['ticker']} — {s['reason']} | PnL {s['pnl_pct']:+.1f}% | MFE {s['mfe_pct']:+.1f}%")
    if buys:
        for b in buys:
            msg.append(f"BUY  {b['ticker']} (#{b['rank']}) amt {b['amount_eur']:.0f}€ | score {b['score']:+.3f} | conf {b['confirm']} | corr {b['corr_max']:.2f}")
    if not sells and not buys:
        msg.append("HOLD — no action")
    msg.append("")
    msg.append("POSITIONS:")
    msg.extend(lines_pos if lines_pos else ["- (none)"])
    msg.append("")
    msg.append("TOP 5 MOMENTUM:")
    if ranked:
        for i, (t, sc) in enumerate(ranked[:5], 1):
            px = close_eur.get(t, np.nan)
            rk = rank_map.get(t, 999)
            conf = int(confirm_state.get(t, 0))
            flt = []
            if close_eur.get(t, np.nan) > sma200_eur.get(t, np.nan):
                flt.append("SMA200")
            if close_eur.get(t, np.nan) >= high60_eur.get(t, np.nan):
                flt.append("HIGH60")
            pxs = f"{float(px):.2f}€" if np.isfinite(px) else "-"
            msg.append(f"{i}. {t} rank {rk} score {sc:+.3f} px {pxs} conf {conf} [{','.join(flt) if flt else '-'}]")
    else:
        msg.append("(ranking indisponible)")

    message = "\n".join(msg)

    save_portfolio(portfolio)
    save_trades(trades)

    print(message)
    send_telegram(message)

    print("=" * 90)
    print("✅ Run terminé | portfolio.json + trades_history.json mis à jour")
    print("=" * 90)


if __name__ == "__main__":
    main()
