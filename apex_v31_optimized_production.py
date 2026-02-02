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
- Sizing: Option A "suspect" 40% ; si gap down (open vs close) < -1.5% => 20% (GapGuard)
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

# Lookback period for data loading (in calendar days)
LOOKBACK_CAL_DAYS = 500  # ~2 years to ensure enough data for 252-day momentum

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
    row["timestamp"] = _now_str()
    trades["trades"].append(row)


def _now_str() -> str:
    return datetime.now().isoformat()


def send_telegram(message: str) -> None:
    """Envoie un message via Telegram (v33 style)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Bot token or chat ID not set. Skipping notification.")
        return
    if not requests:
        print("[Telegram] requests module not found. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        resp.raise_for_status()
        print("[Telegram] Notification envoyée avec succès")
    except Exception as e:
        print(f"[Telegram] Erreur lors de l'envoi: {e}")


# =============================================================================
# Data loading (v33-compatible approach)
# =============================================================================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns: (close, high, low, open, eurusd_close)
    where each is a DataFrame or Series indexed by date, columns=tickers.
    """
    print("[Data] Loading OHLCV data...")

    if os.path.exists(PARQUET_PATH):
        # Parquet mode (backtest)
        print(f"[Data] Found local parquet: {PARQUET_PATH}")
        ohlcv = pd.read_parquet(PARQUET_PATH)
        
        # Assume columns: date, ticker, open, high, low, close
        # Pivot to wide format
        close_df = ohlcv.pivot(index="date", columns="ticker", values="close").sort_index()
        high_df  = ohlcv.pivot(index="date", columns="ticker", values="high").sort_index()
        low_df   = ohlcv.pivot(index="date", columns="ticker", values="low").sort_index()
        open_df  = ohlcv.pivot(index="date", columns="ticker", values="open").sort_index()

        # EURUSD
        eur_df = ohlcv[ohlcv["ticker"] == "EURUSD=X"].set_index("date")[["close"]].sort_index()
        eurusd_close = eur_df["close"] if not eur_df.empty else pd.Series(dtype=float)

    else:
        # Live yfinance mode
        if not yf:
            raise RuntimeError("yfinance not installed and no local parquet found.")
        print("[Data] No parquet found, fetching from yfinance...")

        # Build ticker list from code or default
        tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO",
            "JPM", "V", "UNH", "XOM", "WMT", "MA", "JNJ", "PG", "ORCL", "COST",
            "HD", "NFLX", "BAC", "CRM", "ABBV", "CVX", "KO", "MRK", "AMD", "ADBE",
            "PEP", "TMO", "ACN", "MCD", "CSCO", "LIN", "ABT", "WFC", "TXN", "QCOM",
            "INTU", "DHR", "PM", "AMGN", "VZ", "CAT", "IBM", "GE", "NOW", "COP",
            "BKNG", "ISRG", "UNP", "RTX"
        ]
        tickers.append("EURUSD=X")

        end = datetime.now()
        start = end - timedelta(days=LOOKBACK_CAL_DAYS)

        data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=False, progress=False)
        
        # Build dataframes
        close_df = pd.DataFrame()
        high_df = pd.DataFrame()
        low_df = pd.DataFrame()
        open_df = pd.DataFrame()

        for t in tickers:
            if t == "EURUSD=X":
                continue
            if t in data.columns.levels[0]:
                close_df[t] = data[t]["Close"]
                high_df[t]  = data[t]["High"]
                low_df[t]   = data[t]["Low"]
                open_df[t]  = data[t]["Open"]

        # EURUSD
        if "EURUSD=X" in data.columns.levels[0]:
            eurusd_close = data["EURUSD=X"]["Close"]
        else:
            eurusd_close = pd.Series(dtype=float)

        close_df = close_df.sort_index()
        high_df  = high_df.sort_index()
        low_df   = low_df.sort_index()
        open_df  = open_df.sort_index()

    print(f"[Data] Loaded {len(close_df)} days for {len(close_df.columns)} tickers")
    return close_df, high_df, low_df, open_df, eurusd_close


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 90)
    print("🚀 APEX CHAMPION — PROD (base v33)")
    print("=" * 90)

    # 1) Load data
    close, high, low, open_df, eurusd = load_data()
    tickers = [t for t in close.columns if t != "EURUSD=X"]

    # 2) Compute indicators
    print("[Indicators] Computing SMA200, High60, score, ATR%14, returns...")
    sma200 = close.rolling(SMA200_WIN).mean()
    high60 = high.rolling(HIGH60_WIN).max()
    score  = compute_score(close)
    atrp14 = compute_atr_pct14(high, low, close)
    rets   = close.pct_change(fill_method=None)

    # 3) Breadth
    above_sma = (close > sma200).sum(axis=1)
    breadth = above_sma / len(tickers)

    # 4) Load portfolio
    portfolio = load_portfolio()
    trades = load_trades()

    cash = float(portfolio.get("cash", INITIAL_CAPITAL_EUR))
    positions = portfolio.get("positions", {})
    cooldown = portfolio.get("cooldown", {})
    pending_swap = portfolio.get("pending_swap", {})

    pending_worst = pending_swap.get("worst")
    pending_best = pending_swap.get("best")
    pending_count = pending_swap.get("count", 0)

    # 5) Determine trading dates
    today = datetime.now()
    dates = close.index

    # Signal date = latest close we have
    sig_date = dates[-1]
    sig_idx = len(dates) - 1

    # Execution date = T+1 (in backtest we used open_next, in prod we approximate at close)
    if sig_idx + 1 < len(dates):
        exec_date = dates[sig_idx + 1]
        exec_idx = sig_idx + 1
    else:
        # If no future bar, we can't do anything
        print("⚠️  No future bar to execute => EXIT")
        return

    print(f"[Dates] Signal={sig_date.strftime('%Y-%m-%d')} | Exec={exec_date.strftime('%Y-%m-%d')}")

    # Prices for signal
    px_sig_close = close.iloc[sig_idx]
    px_sig_high60 = high60.iloc[sig_idx]
    px_sig_sma200 = sma200.iloc[sig_idx]

    # Prices for execution (in prod, we use exec_date close as proxy)
    px_exec_open = close.iloc[exec_idx]  # Approximation
    px_exec_close = close.iloc[exec_idx]

    # EURUSD
    eurusd_val = float(eurusd.iloc[exec_idx]) if exec_idx < len(eurusd) else 1.05

    # Breadth gate
    breadth_val = float(breadth.iloc[sig_idx])
    if breadth_val < BREADTH_THR:
        print(f"[Breadth] {breadth_val:.2f} < {BREADTH_THR} => EXIT RISK-OFF")
        # Exit all positions
        sells = []
        for t in list(positions.keys()):
            shares = float(positions[t]["shares"])
            px_out = px_to_eur(t, float(px_exec_open[t]), eurusd_val) * (1.0 - FEE_BPS/10000.0 - SLIPPAGE_BPS/10000.0)
            entry = float(positions[t].get("entry_price_eur", px_out))
            pnl_pct = (px_out/entry - 1.0)*100.0 if entry>0 else 0.0

            cash += px_out * shares
            del positions[t]

            sells.append((t, shares, "RISK_OFF", pnl_pct))
            append_trade(trades, {
                "side": "SELL",
                "ticker": t,
                "date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                "reason": "RISK_OFF",
                "shares": shares,
                "price_eur": px_out,
                "entry_price_eur": entry,
                "pnl_pct": pnl_pct,
            })

        # Send message
        msg = []
        msg.append(f"APEX CHAMPION — {pd.to_datetime(sig_date).strftime('%Y-%m-%d')} -> {pd.to_datetime(exec_date).strftime('%Y-%m-%d')}")
        msg.append(f"EURUSD=X {eurusd_val:.4f} | Breadth {breadth_val:.2f} < {BREADTH_THR} => RISK-OFF")
        msg.append(f"Cash {cash:.2f}€")
        msg.append("")
        msg.append("ACTIONS (execute next open):")
        for t, sh, reason, pnl in sells:
            msg.append(f"SELL {t} {sh:.3f} ({reason}) PnL {pnl:+.1f}%")

        message = "\n".join(msg)
        portfolio["cash"] = float(cash)
        portfolio["positions"] = positions
        portfolio["cooldown"] = {}
        portfolio["pending_swap"] = {}

        save_portfolio(portfolio)
        save_trades(trades)

        print(message)
        send_telegram(message)
        print("=" * 90)
        print("✅ Run terminé (RISK-OFF)")
        print("=" * 90)
        return

    # =====================================================================
    # RISK-ON: Process exits & entries
    # =====================================================================
    fee = FEE_BPS / 10000.0
    slip = SLIPPAGE_BPS / 10000.0

    sells = []
    buys = []

    score_row = score.iloc[sig_idx]

    # Decrement cooldown
    new_cooldown = {}
    for t, cnt in cooldown.items():
        if cnt > 1:
            new_cooldown[t] = cnt - 1

    # =====================================================================
    # 1) Exits
    # =====================================================================
    i_sig = sig_idx
    for t in list(positions.keys()):
        pos = positions[t]
        entry_px = float(pos.get("entry_price_eur", 0.0))
        shares = float(pos.get("shares", 0.0))
        peak_px = float(pos.get("peak_price_eur", entry_px))
        trough_px = float(pos.get("trough_price_eur", entry_px))
        mfe_pct = float(pos.get("mfe_pct", 0.0))
        mae_pct = float(pos.get("mae_pct", 0.0))
        trailing = bool(pos.get("trailing_active", False))

        # Current price (signal close)
        curr_px = px_to_eur(t, float(px_sig_close[t]), eurusd_val)
        if not np.isfinite(curr_px) or curr_px <= 0:
            continue

        # Update peak/trough/MFE/MAE
        if curr_px > peak_px:
            peak_px = curr_px
            mfe_now = (curr_px / entry_px - 1.0) if entry_px > 0 else 0.0
            if mfe_now > mfe_pct:
                mfe_pct = mfe_now
        if curr_px < trough_px:
            trough_px = curr_px
            mae_now = (curr_px / entry_px - 1.0) if entry_px > 0 else 0.0
            if mae_now < mae_pct:
                mae_pct = mae_now

        pos["peak_price_eur"] = float(peak_px)
        pos["trough_price_eur"] = float(trough_px)
        pos["mfe_pct"] = float(mfe_pct)
        pos["mae_pct"] = float(mae_pct)

        # Check exits
        reason = None

        # Hard stop
        if entry_px > 0:
            loss = (curr_px / entry_px - 1.0)
            if loss <= -HARD_STOP_PCT:
                reason = "HARD_STOP"

        # Trailing
        if not trailing and mfe_pct >= MFE_TRIGGER_PCT:
            trailing = True
            pos["trailing_active"] = True

        if trailing:
            drawdown_from_peak = (curr_px / peak_px - 1.0) if peak_px > 0 else 0.0
            if drawdown_from_peak <= -TRAIL_FROM_PEAK_PCT:
                reason = "TRAILING"

        # Trend break
        sma_val = px_to_eur(t, float(px_sig_sma200[t]), eurusd_val)
        if np.isfinite(sma_val) and curr_px < sma_val:
            reason = "TREND_BREAK"

        if reason:
            px_out = px_to_eur(t, float(px_exec_open[t]), eurusd_val) * (1.0 - fee - slip)
            pnl_pct = (px_out/entry_px - 1.0)*100.0 if entry_px>0 else 0.0

            cash += px_out * shares
            del positions[t]
            new_cooldown[t] = COOLDOWN_DAYS

            sells.append((t, shares, reason, pnl_pct))
            append_trade(trades, {
                "side": "SELL",
                "ticker": t,
                "date": pd.to_datetime(exec_date).strftime("%Y-%m-%d"),
                "reason": reason,
                "shares": shares,
                "price_eur": px_out,
                "entry_price_eur": entry_px,
                "pnl_pct": pnl_pct,
            })

    # =====================================================================
    # 2) Entries & SwapEdge
    # =====================================================================
    # Build candidates: Breakout & Close > SMA200 & not in cooldown
    candidates = []
    for t in tickers:
        if t in new_cooldown:
            continue
        c_close = float(px_sig_close[t])
        c_high60 = float(px_sig_high60[t])
        c_sma200 = float(px_sig_sma200[t])

        if not np.isfinite(c_close) or not np.isfinite(c_high60) or not np.isfinite(c_sma200):
            continue

        # Breakout
        prev_high60_idx = i_sig - 1
        if prev_high60_idx < 0:
            continue
        prev_high60_val = float(high60.iloc[prev_high60_idx, close.columns.get_loc(t)])
        if not np.isfinite(prev_high60_val):
            continue

        if c_close > prev_high60_val and c_close > c_sma200:
            candidates.append(t)

    candidates = pd.Series(candidates)
    candidates = candidates[candidates.isin(score_row.index)]
    candidates = pd.Series(candidates.values, index=candidates.values)
    candidates = candidates[score_row[candidates.index].notna()]
    candidates = candidates.reindex(score_row[candidates.index].sort_values(ascending=False).index)

    # Tier & sizing function
    def tier_and_sf(ticker: str) -> Tuple[str, float, float]:
        """Returns (tier, size_fraction, gap)"""
        susp = is_suspect(px_sig_close, px_sig_sma200, atrp14.iloc[sig_idx], ticker)
        # In prod, we don't know tomorrow's open yet
        # gap_val = gap_open_next(float(px_exec_open[ticker]), float(px_sig_close[ticker]))
        gap_val = float("nan")  # unknown

        if susp:
            # We mark as suspect, user will adjust if gap < -1.5%
            return ("suspect", SUSPECT_SIZE, gap_val)
        else:
            return ("normal", 1.0/MAX_POSITIONS, gap_val)

    # Fill empty slots
    if len(positions) < MAX_POSITIONS:
        slots_needed = MAX_POSITIONS - len(positions)
        alloc_cash = cash / slots_needed if slots_needed > 0 else 0.0

        for pick in candidates.index:
            if len(positions) >= MAX_POSITIONS:
                break
            if pick in positions:
                continue

            # Corr gate
            if not corr_ok(rets, i_sig, pick, list(positions.keys())):
                candidates = candidates.drop(index=pick, errors="ignore")
                continue

            tier, sf, gap = tier_and_sf(pick)
            alloc = alloc_cash*sf
            px_in_eur = px_to_eur(pick, float(px_exec_open[pick]), eurusd_val) * (1.0 + fee + slip)

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
                            px_out = px_to_eur(worst, float(px_exec_open[worst]), eurusd_val) * (1.0 - fee - slip)
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
                            px_in = px_to_eur(best, float(px_exec_open[best]), eurusd_val) * (1.0 + fee + slip)
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
        px = px_to_eur(t, float(px_sig_close[t]), eurusd_val)
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
    msg.append(f"EURUSD=X {eurusd_val:.4f} | Breadth {breadth_val:.2f} (gate {BREADTH_THR})")
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
