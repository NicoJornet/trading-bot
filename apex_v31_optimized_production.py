#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX — CHAMPION (PROD) — basé sur la base APEX v33
===================================================

Objectif: utiliser EXACTEMENT la même plomberie que ton script v33
(portfolio.json + trades_history.json + Telegram), mais avec la logique
du Champion validé sur U54 (2015–2025).

⚙️ Logique Champion (résumé)
- Exécution: T+1 open en backtest, et en PROD on estime au close (car l'open n'est pas connu à 8h)
- Régime: Breadth >= 0.55 (part des tickers au-dessus SMA200)
- Entrées: Breakout (Close > High60 précédent) ET Close > SMA200
- Score: 0.5*R126 + 0.3*R252 + 0.2*R63
- SwapEdge: swap si best_score >= worst_score * 1.15, confirmé 3 jours, cooldown 2 jours
- Corr gate: évite d'empiler des actifs très corrélés (corr 63j < 0.60)
- Sorties: HARD_STOP (-18%), TRAILING après MFE (+18%) avec trail -5%, TREND_BREAK (Close<SMA200)
- Sizing: Option A “suspect” 40% ; si gap down (open vs close) < -1.5% => 20% (GapGuard)
  (en PROD, gap connu seulement à l'open → on met une note pour l'appliquer manuellement si besoin)

✅ Outputs
- Met à jour: portfolio.json / trades_history.json
- Envoie un message Telegram avec les actions à faire (BUY/SELL) pour le matin.

"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
# CONFIG (compatible v33)
# =============================================================================
PARQUET_PATH = os.environ.get("APEX_OHLCV_PARQUET", "ohlcv_54tickers_2015_2025.parquet")

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE    = "trades_history.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR     = 100.0

# =============================================================================
# CHAMPION PARAMS (prod defaults)
# =============================================================================
MAX_POSITIONS = 3

FEE_BPS      = 20.0     # 0.20%
SLIPPAGE_BPS = 5.0      # 0.05%

BREADTH_THR = 0.55
SMA200_WIN  = 200
HIGH60_WIN  = 60

# Momentum score windows
R63  = 63
R126 = 126
R252 = 252

# SwapEdge
EDGE_MULT     = 1.15
CONFIRM_DAYS  = 3
COOLDOWN_DAYS = 2
CORR_WIN      = 63
CORR_THR      = 0.60   # set None to disable

# Exits
HARD_STOP_PCT     = 0.18
MFE_TRIGGER_PCT   = 0.18
TRAIL_FROM_PEAK_PCT = 0.05

# Suspect sizing (Option A + GapGuard)
SUSPECT_DIST_SMA200 = 0.70  # +70% above SMA200
SUSPECT_ATRP14      = 0.05  # ATR%14 > 5%
SUSPECT_SIZE        = 0.40  # 40% of slot
GAP_GUARD_THR       = 0.015 # 1.5% gap down
GAP_GUARD_SIZE      = 0.20  # 20% of slot

# Tickers considered "already EUR" (avoid EURUSD conversion)
EUR_SUFFIXES = (".PA", ".AS", ".DE", ".MI", ".MC", ".BR", ".SW", ".LS")

# =============================================================================
# Utilities
# =============================================================================
def is_eur_ticker(t: str) -> bool:
    return any(t.endswith(s) for s in EUR_SUFFIXES)

def px_to_eur(ticker: str, px: float, eurusd: float) -> float:
    """Best-effort conversion: treat most US tickers as USD, EU tickers as EUR."""
    if not np.isfinite(px):
        return float("nan")
    if is_eur_ticker(ticker):
        return float(px)
    if eurusd and eurusd > 0:
        return float(px) / float(eurusd)
    return float(px)

def compute_score(close: pd.DataFrame) -> pd.DataFrame:
    r63  = close.pct_change(R63,  fill_method=None)
    r126 = close.pct_change(R126, fill_method=None)
    r252 = close.pct_change(R252, fill_method=None)
    return 0.5*r126 + 0.3*r252 + 0.2*r63

def compute_atr_pct14(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=0).groupby(level=0).max()
    atr14 = tr.rolling(14).mean()
    return atr14 / close

def corr_ok(returns: pd.DataFrame, i: int, candidate: str, held: List[str]) -> bool:
    if CORR_THR is None or len(held) == 0 or i < CORR_WIN:
        return True
    window = returns.iloc[i-CORR_WIN+1:i+1]
    c = window[candidate]
    for h in held:
        corr = c.corr(window[h])
        if pd.notna(corr) and corr >= CORR_THR:
            return False
    return True

def is_suspect(close_row: pd.Series, sma_row: pd.Series, atrp_row: pd.Series, t: str) -> bool:
    if pd.isna(sma_row[t]) or pd.isna(atrp_row[t]):
        return False
    dist = float(close_row[t]/sma_row[t] - 1.0)
    atrp = float(atrp_row[t])
    return (dist > SUSPECT_DIST_SMA200) and (atrp > SUSPECT_ATRP14)

def gap_open_next(open_next: float, close_today: float) -> float:
    if not np.isfinite(open_next) or not np.isfinite(close_today) or close_today == 0:
        return float("nan")
    return float(open_next/close_today - 1.0)



def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            p = json.load(f)
        # Backward compatible defaults
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
    cols = {c.lower(): c for c in df.columns}
    out = df.copy()
    ren = {}
    for target in ["open", "high", "low", "close", "volume"]:
        if target in cols:
            ren[cols[target]] = target
    return out.rename(columns=ren)



def load_ohlcv_parquet(path: str, tickers: List[str]) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date, columns MultiIndex (ticker, field) where field in open/high/low/close/volume.
    Supports:
      - long: date,ticker,open,high,low,close,volume
      - multiindex: (ticker, field) or (field, ticker)
    """
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
            # (field, ticker)
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
        # now assume (ticker, field)
        df.columns = pd.MultiIndex.from_tuples(
            [(t, str(f).lower()) for (t, f) in df.columns],
            names=["ticker", "field"]
        )
        keep = [c for c in df.columns if c[0] in tickers and c[1] in fields]
        out = df[keep].copy()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    raise ValueError("Format parquet non reconnu.")



def get_eur_usd_rate() -> float:
    """
    Returns EURUSD (USD per 1 EUR). If unavailable, returns 1.0.
    """
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



def load_data() -> pd.DataFrame:
    # Prefer parquet cache
    if os.path.exists(PARQUET_PATH):
        ohlcv = load_ohlcv_parquet(PARQUET_PATH, UNIVERSE)
        return ohlcv.sort_index()

    # Fallback: yfinance (last 420 calendar days)
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_CAL_DAYS)
    return load_ohlcv_yfinance(UNIVERSE, start=start, end=end)


# =============================================================================
# Indicators
# =============================================================================


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")



# =============================================================================
# MAIN (prod run)
# =============================================================================
def main() -> None:
    print("=" * 90)
    print("🚀 APEX CHAMPION — PROD (base v33)")
    print("=" * 90)

    portfolio = load_portfolio()
    trades = load_trades()

    # Monthly DCA (same behavior as v33)
    today = datetime.now()
    month_key = today.strftime("%Y-%m")
    if portfolio.get("last_dca_month") != month_key:
        portfolio["cash"] = float(portfolio.get("cash", INITIAL_CAPITAL_EUR)) + float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
        portfolio["last_dca_month"] = month_key
        print(f"💰 DCA: +{MONTHLY_DCA_EUR:.2f}€ (month={month_key})")

    # Load data (parquet preferred)
    ohlcv = load_data()
    if ohlcv.empty:
        raise RuntimeError("OHLCV vide.")

    # Need at least 2 bars for "T+1" logic (signal day + execution day)
    if len(ohlcv.index) < 2:
        raise RuntimeError("OHLCV doit contenir au moins 2 jours.")

    # Use last available date as execution date if available
    exec_date = ohlcv.index.max()
    sig_date = ohlcv.index[-2]  # signal at close of previous bar
    print(f"📅 Signal (close): {pd.to_datetime(sig_date).strftime('%Y-%m-%d')} | Execute: {pd.to_datetime(exec_date).strftime('%Y-%m-%d')} (open est. si dispo)")

    eurusd = get_eur_usd_rate()
    print(f"💱 EURUSD=X: {eurusd:.4f} (USD -> EUR = USD / eurusd ; EUR tickers non convertis)")

    tickers = sorted({t for t, _ in ohlcv.columns})
    # Build panels
    def field(f): 
        x = ohlcv.loc[:, (tickers, f)].copy()
        x.columns = tickers
        return x.ffill()

    open_  = field("open")  if (tickers[0], "open")  in ohlcv.columns else field("Open")
    high_  = field("high")  if (tickers[0], "high")  in ohlcv.columns else field("High")
    low_   = field("low")   if (tickers[0], "low")   in ohlcv.columns else field("Low")
    close_ = field("close") if (tickers[0], "close") in ohlcv.columns else field("Close")

    # Indicators
    sma200 = close_.rolling(SMA200_WIN).mean()
    high60 = close_.rolling(HIGH60_WIN).max()
    score  = compute_score(close_)
    atrp14 = compute_atr_pct14(high_, low_, close_)
    rets   = close_.pct_change(fill_method=None)

    # Breadth gate (exclude market ticker if present)
    u_b = [t for t in tickers if t != "SPY"]
    breadth = (close_[u_b] > sma200[u_b]).mean(axis=1) if len(u_b)>0 else pd.Series(index=close_.index, data=np.nan)

    # Index positions
    i_sig = close_.index.get_loc(sig_date)
    i_exe = close_.index.get_loc(exec_date)

    close_row = close_.iloc[i_sig]
    high_row  = high_.iloc[i_sig]
    low_row   = low_.iloc[i_sig]
    sma_row   = sma200.iloc[i_sig]
    score_row = score.iloc[i_sig]
    atrp_row  = atrp14.iloc[i_sig]
    breadth_val = float(breadth.iloc[i_sig]) if pd.notna(breadth.iloc[i_sig]) else float("nan")

    regime_on = (breadth_val >= BREADTH_THR) if np.isfinite(breadth_val) else False

    # Execution prices:
    # - if we have open for exec_date, use it as "execution price proxy"
    # - else fallback to close(sig_date)
    px_exec_open = open_.iloc[i_exe] if i_exe < len(open_) else close_.iloc[i_sig]
    px_sig_close = close_.iloc[i_sig]

    fee = FEE_BPS/10000.0
    slip= SLIPPAGE_BPS/10000.0

    # Portfolio state (v33 structure)
    cash = float(portfolio.get("cash", INITIAL_CAPITAL_EUR))
    positions = portfolio.get("positions", {})

    # Cooldown map stored in portfolio (add if missing)
    cooldown = portfolio.get("cooldown", {})
    if not isinstance(cooldown, dict):
        cooldown = {}
    # Decrement cooldown
    new_cooldown = {}
    for t, v in cooldown.items():
        try:
            v = int(v)
        except Exception:
            v = 0
        if v > 1:
            new_cooldown[t] = v - 1

    # Pending swap stored in portfolio (single pair)
    pending = portfolio.get("pending_swap", {}) or {}
    pending_worst = pending.get("worst")
    pending_best  = pending.get("best")
    pending_count = int(pending.get("count", 0) or 0)

    sells = []
    buys  = []

    # =====================================================================
    # 1) Update position stats + exits (decide on close, execute at exec price proxy)
    # =====================================================================
    for t, pos in list(positions.items()):
        if t not in tickers:
            continue

        px_close_eur = px_to_eur(t, float(close_row[t]), eurusd)
        px_exec_eur  = px_to_eur(t, float(px_exec_open[t]), eurusd)

        entry = float(pos.get("entry_price_eur", px_close_eur))
        shares = float(pos.get("shares", 0.0))

        # update peak/trough using signal day range
        peak = float(pos.get("peak_price_eur", entry))
        trough = float(pos.get("trough_price_eur", entry))
        peak = max(peak, px_to_eur(t, float(high_row[t]), eurusd))
        trough = min(trough, px_to_eur(t, float(low_row[t]), eurusd))

        pos["peak_price_eur"] = float(peak)
        pos["trough_price_eur"] = float(trough)

        # MFE/MAE
        mfe = (peak/entry - 1.0) if entry>0 else 0.0
        mae = (trough/entry - 1.0) if entry>0 else 0.0
        pos["mfe_pct"] = float(max(float(pos.get("mfe_pct", 0.0)), mfe*100.0))
        pos["mae_pct"] = float(min(float(pos.get("mae_pct", 0.0)), mae*100.0))

        # activate trailing if MFE trigger hit
        trailing_active = bool(pos.get("trailing_active", False))
        if (not trailing_active) and mfe >= MFE_TRIGGER_PCT:
            trailing_active = True
            pos["trailing_active"] = True

        reason = None

        # HARD STOP (close-based)
        if px_close_eur <= entry * (1.0 - HARD_STOP_PCT):
            reason = "HARD_STOP"

        # TRAILING
        if reason is None and trailing_active:
            dd_from_peak = (px_close_eur / peak - 1.0) if peak>0 else 0.0
            if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
                reason = "TRAILING"

        # TREND_BREAK
        if reason is None and pd.notna(sma_row[t]):
            if float(close_row[t]) < float(sma_row[t]):
                reason = "TREND_BREAK"

        if reason is not None:
            # execute sell at proxy open next (net costs)
            px_out = px_exec_eur * (1.0 - fee - slip)
            proceeds = px_out * shares
            cash += proceeds

            pnl_pct = (px_out/entry - 1.0) * 100.0 if entry>0 else 0.0

            sells.append((t, shares, reason, pnl_pct))
            append_trade(trades, {
                "side": "SELL",
                "ticker": t,
                "date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                "reason": reason,
                "shares": shares,
                "price_eur": px_out,
                "entry_price_eur": entry,
                "pnl_pct": pnl_pct,
                "mfe_pct": pos.get("mfe_pct", 0.0),
                "mae_pct": pos.get("mae_pct", 0.0),
            })
            # remove position
            del positions[t]
            # cooldown
            new_cooldown[t] = COOLDOWN_DAYS

    # =====================================================================
    # 2) Entries / SwapEdge (only if regime ON)
    # =====================================================================
    if regime_on:
        # Eligible candidates: breakout + above SMA200
        eligible = (close_row > high60.shift(1).iloc[i_sig]).fillna(False)
        eligible &= (close_row > sma_row).fillna(False)

        # apply cooldown filter
        for t in new_cooldown.keys():
            if t in eligible.index:
                eligible[t] = False

        candidates = score_row.where(eligible).dropna().sort_values(ascending=False)

        def tier_and_sf(t: str) -> Tuple[str, float, float]:
            sus = is_suspect(close_row, sma_row, atrp_row, t)
            # In PROD, open next may not be known; if we have it in parquet, we can compute.
            gap = gap_open_next(float(px_exec_open[t]), float(px_sig_close[t])) if i_exe != i_sig else float("nan")
            tier="normal"; sf=1.0
            if sus:
                tier="suspect"; sf=SUSPECT_SIZE
                if np.isfinite(gap) and gap < -GAP_GUARD_THR:
                    tier="suspect_gap"; sf=GAP_GUARD_SIZE
            return tier, sf, gap

        # Fill empty slots first
        while len(positions) < MAX_POSITIONS and len(candidates) > 0:
            slots_left = MAX_POSITIONS - len(positions)
            alloc_cash = cash/slots_left if slots_left>0 else 0.0

            pick=None
            for t in candidates.index:
                if t in positions:
                    continue
                if corr_ok(rets, i_sig, t, list(positions.keys())):
                    pick=t; break
            if pick is None:
                break

            tier, sf, gap = tier_and_sf(pick)
            alloc = alloc_cash*sf
            px_in_eur = px_to_eur(pick, float(px_exec_open[pick]), eurusd) * (1.0 + fee + slip)

            shares = math.floor((alloc/px_in_eur)*1000)/1000
            if shares <= 0 or shares*px_in_eur > cash + 1e-9:
                candidates = candidates.drop(index=pick, errors="ignore")
                continue

            note=""
            if tier == "suspect":
                note="suspect 40% (apply 20% if open gap < -1.5%)"
            elif tier == "suspect_gap":
                note="gap-guard 20%"

            cash -= shares*px_in_eur
            buys.append((pick, shares, "ENTRY", tier, note))

            positions[pick] = {
                "entry_date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                "entry_price_eur": float(px_in_eur),
                "shares": float(shares),
                "initial_amount_eur": float(alloc),
                "amount_invested_eur": float(alloc),
                "peak_price_eur": float(px_in_eur),
                "trough_price_eur": float(px_in_eur),
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "trailing_active": False,
                "rank": int(0),
                "score": float(score_row.get(pick, np.nan)),
            }

            append_trade(trades, {
                "side": "BUY",
                "ticker": pick,
                "date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                "reason": "ENTRY",
                "tier": tier,
                "note": note,
                "shares": shares,
                "price_eur": px_in_eur,
            })

            candidates = candidates.drop(index=pick, errors="ignore")

        # SwapEdge only if full
        if len(positions) == MAX_POSITIONS and len(candidates) > 0:
            held = list(positions.keys())
            held_scores = score_row[held].dropna()
            if not held_scores.empty:
                worst = held_scores.sort_values().index[0]
                worst_score = float(held_scores[worst])

                best = None
                for t in candidates.index:
                    if t in positions:
                        continue
                    if corr_ok(rets, i_sig, t, [x for x in held if x != worst]):
                        best=t; break

                if best is not None and np.isfinite(worst_score):
                    best_score = float(score_row[best])
                    if np.isfinite(best_score) and best_score >= worst_score * EDGE_MULT:
                        # update pending pair
                        if pending_worst == worst and pending_best == best:
                            pending_count += 1
                        else:
                            pending_worst, pending_best, pending_count = worst, best, 1

                        if pending_count >= CONFIRM_DAYS:
                            # SELL worst
                            shares_out = float(positions[worst]["shares"])
                            px_out = px_to_eur(worst, float(px_exec_open[worst]), eurusd) * (1.0 - fee - slip)
                            entry = float(positions[worst].get("entry_price_eur", px_out))
                            pnl_pct = (px_out/entry - 1.0)*100.0 if entry>0 else 0.0

                            cash += px_out * shares_out
                            del positions[worst]
                            new_cooldown[worst] = COOLDOWN_DAYS

                            sells.append((worst, shares_out, "SWAP_OUT", pnl_pct))
                            append_trade(trades, {
                                "side": "SELL",
                                "ticker": worst,
                                "date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                                "reason": "SWAP_OUT",
                                "shares": shares_out,
                                "price_eur": px_out,
                                "entry_price_eur": entry,
                                "pnl_pct": pnl_pct,
                            })

                            # BUY best
                            tier, sf, gap = tier_and_sf(best)
                            alloc = cash * sf
                            px_in = px_to_eur(best, float(px_exec_open[best]), eurusd) * (1.0 + fee + slip)
                            shares_in = math.floor((alloc/px_in)*1000)/1000
                            if shares_in > 0 and shares_in*px_in <= cash + 1e-9:
                                note=""
                                if tier == "suspect":
                                    note="suspect 40% (apply 20% if open gap < -1.5%)"
                                elif tier == "suspect_gap":
                                    note="gap-guard 20%"

                                cash -= shares_in*px_in
                                positions[best] = {
                                    "entry_date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                                    "entry_price_eur": float(px_in),
                                    "shares": float(shares_in),
                                    "initial_amount_eur": float(alloc),
                                    "amount_invested_eur": float(alloc),
                                    "peak_price_eur": float(px_in),
                                    "trough_price_eur": float(px_in),
                                    "mfe_pct": 0.0,
                                    "mae_pct": 0.0,
                                    "trailing_active": False,
                                    "rank": int(0),
                                    "score": float(best_score),
                                }

                                buys.append((best, shares_in, "SWAP_IN", tier, note))
                                append_trade(trades, {
                                    "side": "BUY",
                                    "ticker": best,
                                    "date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                                    "reason": "SWAP_IN",
                                    "tier": tier,
                                    "note": note,
                                    "shares": shares_in,
                                    "price_eur": px_in,
                                })

                            pending_worst, pending_best, pending_count = None, None, 0
                    else:
                        pending_worst, pending_best, pending_count = None, None, 0

    # =====================================================================
    # 3) Summary + Telegram (v33 style)
    # =====================================================================
    # Mark-to-market using signal close (or execution proxy)
    pos_value = 0.0
    lines_pos = []
    for t, pos in positions.items():
        px = px_to_eur(t, float(px_sig_close[t]), eurusd)
        v = float(pos.get("shares", 0.0)) * px
        pos_value += v
        lines_pos.append(f"{t}: {v:.0f}€ ({pos.get('shares',0):.3f} sh)")

    total = cash + pos_value
    start_date = datetime.strptime(portfolio.get("start_date", today.strftime("%Y-%m-%d")), "%Y-%m-%d")
    months = (today.year - start_date.year) * 12 + (today.month - start_date.month)
    invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL_EUR)) + max(0, months) * float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
    pnl_total = total - invested
    pnl_total_pct = (total / invested - 1.0) * 100.0 if invested > 0 else 0.0

    # Build telegram message
    msg = []
    msg.append(f"APEX CHAMPION — {pd.to_datetime(sig_date).strftime('%Y-%m-%d')} -> {pd.to_datetime(exec_date).strftime('%Y-%m-%d')}")
    msg.append(f"EURUSD=X {eurusd:.4f} | Breadth {breadth_val:.2f} (gate {BREADTH_THR})")
    msg.append(f"Cash {cash:.2f}€ | Pos {pos_value:.2f}€ | Total {total:.2f}€")
    msg.append(f"Invested~ {invested:.2f}€ | PnL {pnl_total:+.2f}€ ({pnl_total_pct:+.1f}%)")
    msg.append("")
    msg.append("ACTIONS (execute next open):")
    if sells:
        for t, sh, reason, pnl in sells:
            msg.append(f"SELL {t} {sh:.3f} ({reason}) PnL {pnl:+.1f}%")
    if buys:
        for t, sh, reason, tier, note in buys:
            extra = f" [{tier}]" if tier != "normal" else ""
            nn = f" — {note}" if note else ""
            msg.append(f"BUY  {t} {sh:.3f} ({reason}){extra}{nn}")
    if not sells and not buys:
        msg.append("No orders.")
    msg.append("")
    msg.append("POSITIONS:")
    if lines_pos:
        msg.extend(lines_pos[:20])
        if len(lines_pos) > 20:
            msg.append(f"... (+{len(lines_pos)-20} autres)")
    else:
        msg.append("None")

    message = "\n".join(msg)

    # persist
    portfolio["cash"] = float(cash)
    portfolio["positions"] = positions
    portfolio["cooldown"] = new_cooldown
    portfolio["pending_swap"] = ({"worst": pending_worst, "best": pending_best, "count": pending_count} if pending_worst and pending_best else {})

    save_portfolio(portfolio)
    save_trades(trades)

    print(message)
    send_telegram(message)

    print("=" * 90)
    print("✅ Run terminé | portfolio.json + trades_history.json mis à jour")
    print("=" * 90)


if __name__ == "__main__":
    main()
