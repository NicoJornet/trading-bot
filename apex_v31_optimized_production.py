#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX v31/v33 — PROD (GitHub) — FIX + TOP 5 MOMENTUM DISPLAY
===========================================================

- Fix NameError LOOKBACK_CAL_DAYS
- Ajout affichage TOP 5 Momentum (console + Telegram)
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import requests
except Exception:
    requests = None


# =============================================================================
# CONFIG
# =============================================================================

PARQUET_PATH = os.environ.get("APEX_OHLCV_PARQUET", "ohlcv_44tickers_2015_2025.parquet")

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR = 100.0
FEE_EUR = 1.0

MAX_POSITIONS = 3
ALLOC_WEIGHTS = {1: 0.33, 2: 0.33, 3: 0.34}

# Stops
HARD_STOP_PCT = 0.18
MFE_TRIGGER_PCT = 0.15
TRAIL_FROM_PEAK_PCT = 0.05

# Momentum score
SMA_PERIOD = 20
ATR_PERIOD = 14
HIGH_LOOKBACK = 60

# Rotation
FORCE_ROTATION_DAYS = 15

# Entry filter RF0
MAX_RED_FLAGS = 0
RF_RSI_OVERBOUGHT = 75
RF_DIST_HIGH_52W_MIN = -30.0
RF_ATR_PCT_MAX = 7.0
RF_DIST_SMA20_MAX = 20.0

# Quality exits
Q1_BARS = 10
Q1_MFE_PCT = 5.0
Q2_BARS = 15
Q2_MFE_PCT = 8.0

# COMBO guards
RANK_GATE = 15
VOL_RELAX_ATR_PCT = 2.5
SLOW_ASSETS_MULT = 2

# Display
TOP_MOMENTUM_N = 5  # ✅ Affichage Top N

# ✅ FIX: Download window for indicators when parquet missing
def _get_int_env(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name, "")).strip())
        return v if v > 0 else default
    except Exception:
        return default

LOOKBACK_CAL_DAYS = _get_int_env("APEX_LOOKBACK_CAL_DAYS", 420)

# Universe (44 tickers)
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

SLOW_ASSETS = {
    "GLD", "SLV", "SPY", "QQQ",
    "JNJ", "PG", "KO", "WMT", "COST",
    "XOM", "CVX", "UNH", "ABBV",
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# =============================================================================
# IO: portfolio + trades
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            p = json.load(f)
        p.setdefault("currency", "EUR")
        p.setdefault("cash", INITIAL_CAPITAL_EUR)
        p.setdefault("initial_capital", INITIAL_CAPITAL_EUR)
        p.setdefault("monthly_dca", MONTHLY_DCA_EUR)
        p.setdefault("positions", {})
        p.setdefault("start_date", datetime.now().strftime("%Y-%m-%d"))
        p.setdefault("last_dca_month", None)
        return p

    return {
        "currency": "EUR",
        "cash": float(INITIAL_CAPITAL_EUR),
        "initial_capital": float(INITIAL_CAPITAL_EUR),
        "monthly_dca": float(MONTHLY_DCA_EUR),
        "positions": {},
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_month": None,
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
# Data loading (parquet > yfinance)
# =============================================================================

def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).lower(): c for c in df.columns}
    out = df.copy()
    ren = {}
    for target in ["open", "high", "low", "close", "volume"]:
        if target in cols:
            ren[cols[target]] = target
    return out.rename(columns=ren)


def load_ohlcv_parquet(path: str, tickers: List[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if {"date", "ticker"}.issubset(df.columns):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["ticker"].isin(tickers)]
        df = _standardize_ohlcv_columns(df)
        needed = ["open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet long: colonnes manquantes {missing}")
        pivot = df.pivot_table(index="date", columns="ticker", values=needed)
        pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)  # (ticker, field)
        pivot.index = pd.to_datetime(pivot.index)
        return pivot.sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        fields = {"open", "high", "low", "close", "volume"}
        lev0 = set(map(lambda x: str(x).lower(), df.columns.get_level_values(0)))
        if fields.issubset(lev0):
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
        df.columns = pd.MultiIndex.from_tuples(
            [(t, str(f).lower()) for (t, f) in df.columns],
            names=["ticker", "field"]
        )
        keep = [c for c in df.columns if c[0] in tickers and c[1] in fields]
        out = df[keep].copy()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    raise ValueError("Format parquet non reconnu.")


def load_ohlcv_yfinance(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance indisponible.")
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("yfinance n'a pas renvoyé un MultiIndex attendu.")
    lev0 = set(map(lambda x: str(x).lower(), data.columns.get_level_values(0)))
    if {"open", "high", "low", "close", "volume"}.issubset(lev0):
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
    data.columns = pd.MultiIndex.from_tuples(
        [(t, str(f).lower()) for (t, f) in data.columns],
        names=["ticker", "field"]
    )
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


def load_data() -> pd.DataFrame:
    if os.path.exists(PARQUET_PATH):
        ohlcv = load_ohlcv_parquet(PARQUET_PATH, UNIVERSE)
        return ohlcv.sort_index()

    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_CAL_DAYS)
    return load_ohlcv_yfinance(UNIVERSE, start=start, end=end)


# =============================================================================
# Indicators
# =============================================================================

def rsi(close: pd.Series, period: int = 14) -> float:
    c = close.dropna()
    if c.shape[0] < period + 5:
        return np.nan
    d = c.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1])


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    h = high.dropna()
    l = low.dropna()
    c = close.dropna()
    if min(h.shape[0], l.shape[0], c.shape[0]) < period + 5:
        return np.nan
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last = float(c.iloc[-1])
    if pd.isna(atr) or last <= 0:
        return np.nan
    return float(atr) / last * 100.0


def dist_sma20(close: pd.Series, period: int = 20) -> float:
    c = close.dropna()
    if c.shape[0] < period + 5:
        return np.nan
    sma = c.rolling(period).mean().iloc[-1]
    if pd.isna(sma) or sma <= 0:
        return np.nan
    return (float(c.iloc[-1]) / float(sma) - 1.0) * 100.0


def dist_high_52w(close: pd.Series, lookback: int = 252) -> float:
    c = close.dropna()
    if c.shape[0] < 30:
        return np.nan
    h52 = c.rolling(lookback).max().iloc[-1] if c.shape[0] >= lookback else c.max()
    if pd.isna(h52) or h52 <= 0:
        return np.nan
    return (float(c.iloc[-1]) / float(h52) - 1.0) * 100.0


def momentum_score(close: pd.Series, high: pd.Series) -> float:
    c = close.dropna()
    h = high.dropna()
    if c.shape[0] < max(SMA_PERIOD, ATR_PERIOD, HIGH_LOOKBACK) + 5:
        return np.nan

    sma = c.rolling(SMA_PERIOD).mean()
    prev = c.shift(1)
    tr = (h - prev).abs()
    atr = tr.rolling(ATR_PERIOD).mean().replace(0, np.nan)

    high60 = h.rolling(HIGH_LOOKBACK).max().replace(0, np.nan)
    base = (c - sma) / atr
    penalty = (high60 / c.replace(0, np.nan))
    s = (base / penalty).iloc[-1]
    return float(s) if not pd.isna(s) else np.nan


def entry_red_flags(close: pd.Series, high: pd.Series, low: pd.Series) -> Tuple[int, List[str], dict]:
    ind = {
        "rsi14": rsi(close, 14),
        "atr_pct": atr_percent(high, low, close, 14),
        "dist_sma20_pct": dist_sma20(close, 20),
        "dist_high_52w_pct": dist_high_52w(close, 252),
    }
    flags = []
    if not pd.isna(ind["rsi14"]) and ind["rsi14"] > RF_RSI_OVERBOUGHT:
        flags.append(f"RSI>{RF_RSI_OVERBOUGHT}")
    if not pd.isna(ind["dist_high_52w_pct"]) and ind["dist_high_52w_pct"] < RF_DIST_HIGH_52W_MIN:
        flags.append(f"52W<{RF_DIST_HIGH_52W_MIN}%")
    if not pd.isna(ind["atr_pct"]) and ind["atr_pct"] > RF_ATR_PCT_MAX:
        flags.append(f"ATR%>{RF_ATR_PCT_MAX}")
    if not pd.isna(ind["dist_sma20_pct"]) and ind["dist_sma20_pct"] > RF_DIST_SMA20_MAX:
        flags.append(f"EXT>{RF_DIST_SMA20_MAX}%SMA20")
    return len(flags), flags, ind


# =============================================================================
# FX (USD -> EUR)
# =============================================================================

def get_eur_usd_rate() -> float:
    if yf is None:
        return 1.0
    try:
        t = yf.Ticker("EURUSD=X")
        px = t.info.get("regularMarketPrice") or t.info.get("previousClose")
        if px and float(px) > 0:
            return float(px)
    except Exception:
        pass
    return 1.0


def usd_to_eur(price_usd: float, eurusd: float) -> float:
    if eurusd <= 0:
        return float(price_usd)
    return float(price_usd) / float(eurusd)


# =============================================================================
# Trading logic helpers
# =============================================================================

def allocate_cash(cash: float, slot_rank: int) -> float:
    w = ALLOC_WEIGHTS.get(slot_rank, 1.0 / MAX_POSITIONS)
    return cash * w


def bars_held(index: pd.Index, entry_date_str: str, current_date: pd.Timestamp) -> int:
    try:
        entry_dt = pd.to_datetime(entry_date_str)
    except Exception:
        return 0
    mask = (index >= entry_dt) & (index <= current_date)
    return int(mask.sum())


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 90)
    print("🚀 APEX — PROD (fixed LOOKBACK_CAL_DAYS + TOP Momentum)")
    print("=" * 90)
    print(f"🕒 {_now_str()}")
    print(f"🔎 LOOKBACK_CAL_DAYS={LOOKBACK_CAL_DAYS} (fallback yfinance)")

    portfolio = load_portfolio()
    trades = load_trades()

    # Monthly DCA
    today = datetime.now()
    month_key = f"{today.year}-{today.month:02d}"
    if portfolio.get("last_dca_month") != month_key:
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + MONTHLY_DCA_EUR
        portfolio["last_dca_month"] = month_key
        print(f"💰 DCA: +{MONTHLY_DCA_EUR:.2f}€ (month={month_key})")

    # Load data
    ohlcv = load_data()
    if ohlcv.empty:
        raise RuntimeError("OHLCV vide.")

    # Use latest available date as "signal day"
    d = ohlcv.index.max()
    d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
    print(f"📅 Dernière date OHLCV: {d_str}")

    eurusd = get_eur_usd_rate()
    print(f"💱 EURUSD=X: {eurusd:.4f} (prixUSD -> prixEUR = USD / eurusd)")

    # Precompute score/rank + indicators for all tickers
    score_map: Dict[str, float] = {}
    ind_map: Dict[str, dict] = {}

    for t in UNIVERSE:
        if (t, "close") not in ohlcv.columns or (t, "high") not in ohlcv.columns or (t, "low") not in ohlcv.columns:
            continue

        c = ohlcv[(t, "close")].loc[:d]
        h = ohlcv[(t, "high")].loc[:d]
        l = ohlcv[(t, "low")].loc[:d]

        if c.dropna().shape[0] < 60:
            continue

        sc = momentum_score(c, h)
        if pd.isna(sc) or sc <= 0:
            continue

        rf_n, rf_flags, ind = entry_red_flags(c, h, l)
        score_map[t] = float(sc)
        ind_map[t] = {"rf_n": rf_n, "rf_flags": rf_flags, **ind}

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    rank_map = {t: (i + 1) for i, (t, _) in enumerate(ranked)}

    # Latest prices (USD), then convert to EUR
    last_close_usd = {}
    last_high_usd = {}
    last_low_usd = {}
    for t in UNIVERSE:
        if (t, "close") not in ohlcv.columns:
            last_close_usd[t] = np.nan
            last_high_usd[t] = np.nan
            last_low_usd[t] = np.nan
            continue
        c = ohlcv[(t, "close")].loc[d]
        hi = ohlcv[(t, "high")].loc[d] if (t, "high") in ohlcv.columns else np.nan
        lo = ohlcv[(t, "low")].loc[d] if (t, "low") in ohlcv.columns else np.nan
        last_close_usd[t] = float(c) if pd.notna(c) else np.nan
        last_high_usd[t] = float(hi) if pd.notna(hi) else np.nan
        last_low_usd[t] = float(lo) if pd.notna(lo) else np.nan

    # ✅ TOP MOMENTUM (display only)
    topN = ranked[:TOP_MOMENTUM_N]
    top_lines_console = []
    for i, (t, sc) in enumerate(topN, 1):
        rf = ind_map.get(t, {})
        flags = ",".join(rf.get("rf_flags", [])) if rf.get("rf_flags") else "-"
        top_lines_console.append(
            f"{i:>2}. {t:<5} | score {sc:>7.3f} | RSI {rf.get('rsi14', np.nan):>5.1f} | "
            f"ATR% {rf.get('atr_pct', np.nan):>4.1f} | RF {rf.get('rf_n', 0)} [{flags}]"
        )

    print("\n📈 TOP MOMENTUM:")
    if top_lines_console:
        for line in top_lines_console:
            print("   " + line)
    else:
        print("   (aucun score valide)")

    # =====================================================================
    # 1) Evaluate positions -> SELL signals
    # =====================================================================
    sells = []
    positions = portfolio.get("positions", {})

    for t, pos in list(positions.items()):
        px_usd = last_close_usd.get(t, np.nan)
        hi_usd = last_high_usd.get(t, np.nan)
        lo_usd = last_low_usd.get(t, np.nan)
        if np.isnan(px_usd):
            continue

        px_eur = usd_to_eur(px_usd, eurusd)
        hi_eur = usd_to_eur(hi_usd, eurusd) if not np.isnan(hi_usd) else px_eur
        lo_eur = usd_to_eur(lo_usd, eurusd) if not np.isnan(lo_usd) else px_eur

        entry_price = float(pos.get("entry_price_eur", pos.get("entry_price", px_eur)))
        shares = float(pos.get("shares", 0.0))
        if shares <= 0:
            continue

        peak = float(pos.get("peak_price_eur", entry_price))
        trough = float(pos.get("trough_price_eur", entry_price))
        peak = max(peak, hi_eur)
        trough = min(trough, lo_eur)

        mfe_pct = (peak / entry_price - 1.0) * 100.0
        mae_pct = (trough / entry_price - 1.0) * 100.0
        trailing_active = (mfe_pct / 100.0) >= MFE_TRIGGER_PCT

        pos["peak_price_eur"] = peak
        pos["trough_price_eur"] = trough
        pos["mfe_pct"] = mfe_pct
        pos["mae_pct"] = mae_pct
        pos["trailing_active"] = trailing_active

        cur_score = float(score_map.get(t, 0.0))
        cur_rank = int(rank_map.get(t, 999))
        pos["score"] = cur_score
        pos["rank"] = cur_rank

        if cur_score <= 0:
            pos["days_score_le0"] = int(pos.get("days_score_le0", 0)) + 1
        else:
            pos["days_score_le0"] = 0

        entry_date = pos.get("entry_date", pos.get("date", d_str))
        bh = bars_held(ohlcv.index, entry_date, pd.to_datetime(d))
        pos["bars_held"] = int(bh)

        pnl_eur = (px_eur - entry_price) * shares
        pnl_pct = (px_eur / entry_price - 1.0) * 100.0

        reason = None

        stop_price = entry_price * (1.0 - HARD_STOP_PCT)
        if px_eur <= stop_price:
            reason = "HARD_STOP"

        if reason is None and trailing_active:
            dd_from_peak = (px_eur / peak - 1.0)
            if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
                reason = "MFE_TRAILING"

        if reason is None and trailing_active:
            pass
        else:
            if reason is None and int(pos.get("days_score_le0", 0)) >= FORCE_ROTATION_DAYS:
                reason = f"FORCE_ROTATION_{FORCE_ROTATION_DAYS}d"

            if reason is None and cur_rank > RANK_GATE:
                atrp = ind_map.get(t, {}).get("atr_pct", np.nan)
                mult = 1
                if t in SLOW_ASSETS:
                    mult = max(mult, SLOW_ASSETS_MULT)
                if not pd.isna(atrp) and atrp < VOL_RELAX_ATR_PCT:
                    mult = max(mult, 2)

                q1_b = Q1_BARS * mult
                q2_b = Q2_BARS * mult

                if bh >= q1_b and mfe_pct < Q1_MFE_PCT:
                    reason = f"QUALITY_MFE<{Q1_MFE_PCT}%_{q1_b}b"
                elif bh >= q2_b and mfe_pct < Q2_MFE_PCT:
                    reason = f"QUALITY_MFE<{Q2_MFE_PCT}%_{q2_b}b"

        if reason is not None:
            value_eur = px_eur * shares
            sells.append({
                "ticker": t,
                "price_eur": px_eur,
                "shares": shares,
                "value_eur": value_eur,
                "pnl_eur": pnl_eur - FEE_EUR,
                "pnl_pct": pnl_pct,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "bars_held": bh,
                "reason": reason,
                "rank": cur_rank,
                "score": cur_score,
                "entry_date": entry_date,
                "entry_price_eur": entry_price,
            })

        positions[t] = pos

    for s in sells:
        t = s["ticker"]
        proceeds = float(s["value_eur"]) - FEE_EUR
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + proceeds

        append_trade(trades, {
            "action": "SELL",
            "ticker": t,
            "date": d_str,
            "price_eur": float(s["price_eur"]),
            "shares": float(s["shares"]),
            "amount_eur": float(s["value_eur"]),
            "fee_eur": float(FEE_EUR),
            "reason": s["reason"],
            "pnl_eur": float(s["pnl_eur"]),
            "pnl_pct": float(s["pnl_pct"]),
            "mfe_pct": float(s["mfe_pct"]),
            "mae_pct": float(s["mae_pct"]),
            "bars_held": int(s["bars_held"]),
            "rank": int(s["rank"]),
            "score": float(s["score"]),
        })

        if t in portfolio.get("positions", {}):
            del portfolio["positions"][t]

    # =====================================================================
    # 2) BUY signals (RF0) for top ranks 1..MAX_POSITIONS
    # =====================================================================
    buys = []
    held = set(portfolio.get("positions", {}).keys())
    slots = MAX_POSITIONS - len(held)

    cash = float(portfolio.get("cash", 0.0))
    if slots > 0 and cash > 50:
        for t, sc in ranked:
            if slots <= 0:
                break
            r = int(rank_map.get(t, 999))
            if r > MAX_POSITIONS:
                continue
            if t in held:
                continue

            rf = ind_map.get(t, {})
            if int(rf.get("rf_n", 99)) > MAX_RED_FLAGS:
                continue

            px_usd = last_close_usd.get(t, np.nan)
            if np.isnan(px_usd) or px_usd <= 0:
                continue
            px_eur = usd_to_eur(px_usd, eurusd)

            alloc = allocate_cash(cash, r)
            alloc = min(alloc, max(0.0, cash - 10.0))
            if alloc < 50:
                continue

            shares = alloc / px_eur
            cost = alloc + FEE_EUR
            if cost > cash:
                continue

            buys.append({
                "ticker": t,
                "rank": r,
                "score": float(sc),
                "price_eur": px_eur,
                "shares": shares,
                "amount_eur": alloc,
                "rsi14": rf.get("rsi14", np.nan),
                "atr_pct": rf.get("atr_pct", np.nan),
            })

            cash -= cost
            portfolio["cash"] = cash
            portfolio["positions"][t] = {
                "entry_date": d_str,
                "entry_price_eur": float(px_eur),
                "shares": float(shares),
                "initial_amount_eur": float(alloc),
                "amount_invested_eur": float(alloc),
                "peak_price_eur": float(px_eur),
                "trough_price_eur": float(px_eur),
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "trailing_active": False,
                "days_score_le0": 0,
                "rank": int(r),
                "score": float(sc),
            }

            append_trade(trades, {
                "action": "BUY",
                "ticker": t,
                "date": d_str,
                "price_eur": float(px_eur),
                "shares": float(shares),
                "amount_eur": float(alloc),
                "fee_eur": float(FEE_EUR),
                "reason": f"BUY_RANK{r}_RF0",
                "rank": int(r),
                "score": float(sc),
            })

            held.add(t)
            slots -= 1

    # =====================================================================
    # 3) Portfolio summary + Telegram
    # =====================================================================
    pos_value = 0.0
    lines_pos = []
    for t, pos in portfolio.get("positions", {}).items():
        px_usd = last_close_usd.get(t, np.nan)
        if np.isnan(px_usd):
            continue
        px_eur = usd_to_eur(px_usd, eurusd)
        entry = float(pos.get("entry_price_eur", px_eur))
        sh = float(pos.get("shares", 0.0))
        val = px_eur * sh
        pos_value += val
        pnl_pct = (px_eur / entry - 1.0) * 100.0 if entry > 0 else 0.0
        mfe = float(pos.get("mfe_pct", 0.0))
        trail = "ON" if bool(pos.get("trailing_active", False)) else "OFF"
        rk = int(pos.get("rank", 999))
        lines_pos.append(f"- {t} (#{rk}) PnL {pnl_pct:+.1f}% | MFE {mfe:+.1f}% | Trail {trail}")

    cash = float(portfolio.get("cash", 0.0))
    total = cash + pos_value

    start_date = pd.to_datetime(portfolio.get("start_date", d_str))
    months = (today.year - start_date.year) * 12 + (today.month - start_date.month)
    invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL_EUR)) + max(0, months) * float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
    pnl_total = total - invested
    pnl_total_pct = (total / invested - 1.0) * 100.0 if invested > 0 else 0.0

    # ✅ Add Top Momentum to message
    top_lines_msg = []
    for i, (t, sc) in enumerate(topN, 1):
        rf = ind_map.get(t, {})
        flags = ",".join(rf.get("rf_flags", [])) if rf.get("rf_flags") else "-"
        top_lines_msg.append(
            f"{i}. {t} score {sc:.3f} | RSI {rf.get('rsi14', np.nan):.0f} | ATR% {rf.get('atr_pct', np.nan):.1f} | RF {rf.get('rf_n', 0)} [{flags}]"
        )

    msg = []
    msg.append(f"APEX PROD — {d_str}")
    msg.append(f"EURUSD=X {eurusd:.4f}")
    msg.append(f"Cash {cash:.2f}€ | Pos {pos_value:.2f}€ | Total {total:.2f}€")
    msg.append(f"Invested~ {invested:.2f}€ | PnL {pnl_total:+.2f}€ ({pnl_total_pct:+.1f}%)")
    msg.append("")
    msg.append(f"TOP {TOP_MOMENTUM_N} MOMENTUM:")
    msg.extend(top_lines_msg if top_lines_msg else ["(aucun score valide)"])
    msg.append("")
    msg.append("ACTIONS:")
    if sells:
        for s in sells:
            msg.append(f"SELL {s['ticker']} — {s['reason']} | PnL {s['pnl_pct']:+.1f}% | MFE {s['mfe_pct']:+.1f}% | Hold {s['bars_held']}b")
    if buys:
        for b in buys:
            msg.append(f"BUY  {b['ticker']} (#{b['rank']}) amt {b['amount_eur']:.0f}€ | score {b['score']:.3f} | RSI {b['rsi14']:.0f} | ATR% {b['atr_pct']:.1f}")
    if not sells and not buys:
        msg.append("HOLD — no action")
    msg.append("")
    msg.append("POSITIONS:")
    msg.extend(lines_pos if lines_pos else ["- (none)"])

    message = "\n".join(msg)

    save_portfolio(portfolio)
    save_trades(trades)

    print("\n" + "=" * 90)
    print(message)
    print("=" * 90)

    send_telegram(message)

    print("=" * 90)
    print("✅ Run terminé | portfolio.json + trades_history.json mis à jour")
    print("=" * 90)


if __name__ == "__main__":
    main()
