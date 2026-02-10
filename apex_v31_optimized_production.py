#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX PROD — PACK_FINAL_V6C (A_rank5) — Telegram + Portfolio (v31 scaffold)
=========================================================================

✅ But
- Conserver la plomberie "prod" de v31 : portfolio.json, trades_history.json, message Telegram.
- Remplacer uniquement la logique stratégie par la version figée PACK_FINAL_V6C / A_rank5.
- Source de prix : yfinance uniquement (aucun fichier OHLCV local).
- Exécution : signal Close(J) -> exécution théorique Open(J+1) (message d'alerte).

📌 Stratégie figée
- Ranking: score = 0.2*R63 + 0.5*R126 + 0.3*R252 (Close)
- Filtre: Close > SMA200 (par actif)
- Sélection: Top-3 avec "A_rank5" (no-swap zone)
    - on conserve une position si elle est encore dans le TOP 5 du ranking
- Corr-aware:
    - calcule max corr(63j) sur le pool ranked (rank_pool=15)
    - si max_corr > 0.92 -> sélection greedy anti-corr (corr<0.80) sur 10 candidats
- Sizing: inverse-vol (vol20) normalisé
- Delta-rebalance: trade uniquement si |diff valeur| >= 5% equity
- Frais: 1€ par ordre
- DCA mensuel: +100€ au début de chaque mois (clé last_dca_month comme v31)

📦 Fichiers
- portfolio.json (état)
- trades_history.json (historique)
- Message Telegram (si TELEGRAM_BOT_TOKEN/CHAT_ID set)

⚠️ IMPORTANT (FX)
- Aucune conversion FX n'est utilisée ici (EURUSD=X désactivé). Le portefeuille est géré en "unités"
  cohérentes avec le dataset (souvent USD pour tickers US). Si ton dataset mélange devises, il faut
  séparer en blocs (US/EUR) — non fait dans ce script prod.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None


# =============================================================================
# CONFIG (v31-like)
# =============================================================================

DATA_SOURCE = os.environ.get("APEX_DATA_SOURCE", "yfinance")  # yfinance only
YF_START = os.environ.get("APEX_YF_START", "2014-01-01")
YF_END = os.environ.get("APEX_YF_END", "")  # empty => today
YF_PERIOD = os.environ.get("APEX_YF_PERIOD", "")  # optional, e.g. "5y"
YF_INTERVAL = os.environ.get("APEX_YF_INTERVAL", "1d")


PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

# Capital & costs
INITIAL_CAPITAL = float(os.environ.get("APEX_INITIAL_CAPITAL", "2000"))
MONTHLY_DCA = float(os.environ.get("APEX_MONTHLY_DCA", "100"))
FEE_PER_ORDER = float(os.environ.get("APEX_FEE_PER_ORDER", "1"))

# Strategy params (PACK_FINAL_V6C)
MAX_POSITIONS = 3
KEEP_RANK = 5
RANK_POOL = 15
REBALANCE_TD = int(os.environ.get("APEX_REBALANCE_TD", "10"))  # informational for prod msg
DELTA_REBAL = float(os.environ.get("APEX_DELTA_REBAL", "0.05"))

SMA200_WIN = 200
VOL_WIN = 20
R63, R126, R252 = 63, 126, 252
W63, W126, W252 = 0.2, 0.5, 0.3

CORR_WIN = 63
CORR_GATE_THR = 0.92
CORR_PICK_THR = 0.80
CORR_SCAN = 10

TOP_MOMENTUM_N = 5

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# =============================================================================
# Universe (copied from v31 file)
# =============================================================================

UNIVERSE = [
    # Magnificent Seven & Big Tech (7)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",

    # Semi-conducteurs & IA (9)
    "AMD", "MU", "ASML", "TSM", "LRCX", "AMAT", "AVGO", "ANET", "SMCI",

    # Cybersécurité & Cloud (5)
    "PLTR", "CRWD", "NET", "DDOG", "ZS",

    # Growth Tech (6)
    "SHOP", "UBER", "ABNB", "RKLB", "APP", "VRT",

    # Crypto (3)
    "MSTR", "MARA", "RIOT",

    # Santé & Biotech (16)
    "LLY", "NVO", "UNH", "JNJ", "ABBV", "VRTX", "ISRG", "TMO", "DHR", "ABT",
    "PFE", "MRK", "BMY", "AMGN", "GILD", "REGN",

    # Conso / Retail (10)
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "NFLX",

    # Energie / Commodities (8)
    "XOM", "CVX", "SLB", "COP", "OXY", "CNQ", "EOG", "PSX",

    # Finance (10)
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK-B", "BLK",

    # Industrie / Défense (10)
    "CAT", "DE", "BA", "GE", "HON", "LMT", "RTX", "NOC", "GD", "ETN",

    # Utilities / Infrastructure (5)
    "NEE", "DUK", "SO", "AEP", "D",

    # Matériaux (5)
    "LIN", "APD", "SHW", "FCX", "NEM",

    # REITs (5)
    "AMT", "PLD", "CCI", "EQIX", "O",

    # ETFs / Indices (13)
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI",
    "TLT", "IEF", "HYG",

    # Métaux précieux (2)
    "GLD", "SLV",

    # Europe (some)
    "MC.PA", "OR.PA", "RMS.PA", "ASML.AS", "SAP.DE", "DAX", "CAC40",
]


# =============================================================================
# IO: portfolio + trades (v31 style)
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            p = json.load(f)
        p.setdefault("cash", INITIAL_CAPITAL)
        p.setdefault("initial_capital", INITIAL_CAPITAL)
        p.setdefault("monthly_dca", MONTHLY_DCA)
        p.setdefault("positions", {})
        p.setdefault("start_date", datetime.now().strftime("%Y-%m-%d"))
        p.setdefault("last_dca_month", None)
        return p
    return {
        "cash": float(INITIAL_CAPITAL),
        "initial_capital": float(INITIAL_CAPITAL),
        "monthly_dca": float(MONTHLY_DCA),
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
# Data loading (CSV -> MultiIndex like v31)
# =============================================================================

# =============================================================================
# Data loading (YFINANCE ONLY) — MultiIndex-compatible long format
# =============================================================================

def load_data_yfinance(tickers: List[str]) -> pd.DataFrame:
    """Download OHLCV from yfinance and return long format columns:
    date,ticker,open,high,low,close,volume

    Notes:
    - interval fixed to daily by default (YF_INTERVAL=1d)
    - if YF_PERIOD is set (e.g. "5y"), it is used; else start/end are used.
    - no FX conversion (data in local currencies).
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is required (pip install yfinance>=0.2.33)") from e

    if not tickers:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])

    kwargs = dict(
        tickers=" ".join(tickers),
        interval=YF_INTERVAL,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if YF_PERIOD:
        kwargs["period"] = YF_PERIOD
    else:
        kwargs["start"] = YF_START
        if YF_END:
            kwargs["end"] = YF_END

    data = yf.download(**kwargs)

    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])

    # yfinance returns:
    # - MultiIndex columns for multiple tickers: (field, ticker) or (ticker, field) depending on version
    # Normalize to long format
    if isinstance(data.columns, pd.MultiIndex):
        cols = data.columns
        # detect orientation
        lvl0 = set(map(str, cols.get_level_values(0)))
        lvl1 = set(map(str, cols.get_level_values(1)))
        fields = {"Open","High","Low","Close","Volume"}
        if fields.issubset(lvl0):
            # (field, ticker)
            pieces = []
            for field in ["Open","High","Low","Close","Volume"]:
                wide = data[field].copy()
                wide.columns = wide.columns.astype(str)
                pieces.append(wide.stack().rename(field.lower()))
            out = pd.concat(pieces, axis=1).reset_index()
            out.columns = ["date","ticker","open","high","low","close","volume"]
        elif fields.issubset(lvl1):
            # (ticker, field)
            pieces=[]
            for t in sorted(lvl0):
                sub = data[t].copy()
                sub.columns = sub.columns.astype(str)
                sub = sub.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                sub["ticker"]=t
                pieces.append(sub.reset_index())
            out = pd.concat(pieces, axis=0, ignore_index=True)
            out = out[["Date","ticker","open","high","low","close","volume"]].rename(columns={"Date":"date"})
        else:
            raise RuntimeError("Unexpected yfinance column MultiIndex format.")
    else:
        # single ticker: columns are fields
        df1 = data.copy()
        df1 = df1.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df1["ticker"] = tickers[0]
        out = df1.reset_index()[["Date","ticker","open","high","low","close","volume"]].rename(columns={"Date":"date"})

    
    out = out.dropna(subset=["date","ticker","close"]).copy()
    out["ticker"] = out["ticker"].astype(str)
    out = out.sort_values(["date","ticker"])

    # Pivot to v31-like wide MultiIndex: columns = (ticker, field), index = date
    frames = []
    for field in ["open","high","low","close","volume"]:
        wide = out.pivot(index="date", columns="ticker", values=field).sort_index()
        wide.columns = pd.MultiIndex.from_product([wide.columns.astype(str), [field]])
        frames.append(wide)
    ohlcv = pd.concat(frames, axis=1).sort_index(axis=1)

    # Ensure DatetimeIndex (no timezone) and float columns
    ohlcv.index = pd.to_datetime(ohlcv.index)
    return ohlcv



def _corr_matrix(win: pd.DataFrame) -> np.ndarray:
    mat = win.to_numpy(dtype=float)
    mat = mat - np.nanmean(mat, axis=0, keepdims=True)
    mat = mat / (np.nanstd(mat, axis=0, keepdims=True) + 1e-12)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (mat.T @ mat) / max(mat.shape[0]-1, 1)
    return np.clip(corr, -1, 1)

def _max_offdiag(corr: np.ndarray) -> float:
    if corr is None or corr.shape[0] < 2:
        return float("nan")
    m = corr.copy()
    np.fill_diagonal(m, np.nan)
    return float(np.nanmax(m))

def greedy_pick_low_corr(candidates: List[str], corr_assets: List[str], corr: np.ndarray,
                         topk: int, thr: float, max_scan: int) -> List[str]:
    if corr is None or len(corr_assets) < 2:
        return candidates[:topk]
    idx = {t:i for i,t in enumerate(corr_assets)}
    chosen=[]
    scanned=0
    for t in candidates:
        scanned += 1
        if scanned > max_scan:
            break
        if t not in idx:
            continue
        ti = idx[t]
        ok = True
        for c in chosen:
            if c in idx and corr[ti, idx[c]] >= thr:
                ok = False
                break
        if ok:
            chosen.append(t)
        if len(chosen) >= topk:
            break
    if len(chosen) < topk:
        for t in candidates:
            if t not in chosen:
                chosen.append(t)
            if len(chosen) >= topk:
                break
    return chosen

def apply_a_rank5(current: List[str], ranked: List[str], topk: int, keep_rank: int) -> List[str]:
    if not ranked:
        return []
    rank_pos = {t:i+1 for i,t in enumerate(ranked)}  # 1-based
    kept = [t for t in current if rank_pos.get(t, 10**9) <= keep_rank]
    kept = kept[:topk]
    out = list(kept)
    for t in ranked:
        if t in out:
            continue
        out.append(t)
        if len(out) >= topk:
            break
    return out[:topk]

def invvol_weights(vol: pd.Series, tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    v = vol.reindex(tickers).replace(0, np.nan)
    inv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty:
        return {}
    inv = inv / inv.sum()
    return {t: float(inv.loc[t]) for t in inv.index}


# =============================================================================
# MAIN (prod daily)
# =============================================================================

def main() -> None:
    print("="*90)
    print("APEX PROD — PACK_FINAL_V6C (A_rank5) — YFINANCE ONLY")
    print("="*90)
    print(f"🕒 {_now_str()}")
    portfolio = load_portfolio()
    trades = load_trades()

    # Monthly DCA (v31 mechanism)
    today = datetime.now()
    month_key = f"{today.year}-{today.month:02d}"
    if portfolio.get("last_dca_month") != month_key:
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + MONTHLY_DCA
        portfolio["last_dca_month"] = month_key
        print(f"💰 DCA: +{MONTHLY_DCA:.2f} (month={month_key})")

    ohlcv = load_data_yfinance(UNIVERSE)
    if ohlcv.empty:
        raise RuntimeError("OHLCV vide.")

    d = ohlcv.index.max()
    d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
    print(f"📅 Dernière date OHLCV: {d_str}")

    # Precompute cross-sectional features at date d
    closes = pd.Series({t: ohlcv[(t,"close")].loc[d] for t in UNIVERSE if (t,"close") in ohlcv.columns}, dtype=float)
    opens_next = pd.Series({t: ohlcv[(t,"open")].iloc[-1] for t in UNIVERSE if (t,"open") in ohlcv.columns}, dtype=float)

    # Build Close matrix for indicators
    C = pd.DataFrame({t: ohlcv[(t,"close")] for t in UNIVERSE if (t,"close") in ohlcv.columns})
    r1 = C.pct_change()

    R63s  = C / C.shift(R63)  - 1.0
    R126s = C / C.shift(R126) - 1.0
    R252s = C / C.shift(R252) - 1.0
    score = W63*R63s + W126*R126s + W252*R252s
    sma200 = C.rolling(SMA200_WIN, min_periods=SMA200_WIN).mean()
    vol20 = r1.rolling(VOL_WIN, min_periods=VOL_WIN).std()

    elig = (C.loc[d] > sma200.loc[d]) & score.loc[d].notna() & vol20.loc[d].notna()
    s = score.loc[d].where(elig, np.nan).dropna()
    ranked_all = list(s.sort_values(ascending=False).index)
    ranked_pool = ranked_all[:RANK_POOL]

    # Corr gate and pick (for informational + selection)
    max_corr = float("nan")
    corr_gate_on = False
    desired_ranked = ranked_pool[:MAX_POSITIONS]
    if len(ranked_pool) >= 2:
        win = r1[ranked_pool].iloc[-CORR_WIN:]
        if win.shape[0] >= int(0.8*CORR_WIN):
            corr = _corr_matrix(win)
            max_corr = _max_offdiag(corr)
            corr_gate_on = bool(np.isfinite(max_corr) and max_corr > CORR_GATE_THR)
            if corr_gate_on:
                desired_ranked = greedy_pick_low_corr(ranked_pool, ranked_pool, corr, MAX_POSITIONS, CORR_PICK_THR, CORR_SCAN)
            else:
                desired_ranked = ranked_pool[:MAX_POSITIONS]

    current = list(portfolio.get("positions", {}).keys())
    desired = apply_a_rank5(current=current, ranked=desired_ranked, topk=MAX_POSITIONS, keep_rank=KEEP_RANK)

    # Top momentum display
    topN = ranked_all[:TOP_MOMENTUM_N]
    top_lines = []
    for i, t in enumerate(topN, 1):
        top_lines.append(f"{i}. {t} score {float(score.loc[d, t]):.3f}")

    # Portfolio valuation at close d
    cash = float(portfolio.get("cash", 0.0))
    pos_val = 0.0
    for t, pos in portfolio.get("positions", {}).items():
        px = float(closes.get(t, np.nan))
        if np.isnan(px) or px <= 0:
            continue
        pos_val += float(pos.get("shares", 0.0)) * px
    equity = cash + pos_val

    # Target weights
    w = invvol_weights(vol20.loc[d], desired)
    targets_val = {t: w[t]*equity for t in w}

    # Build orders (rebalance at next open, but we simulate with close for sizing)
    orders = []

    # SELL tickers not in targets
    for t in list(portfolio.get("positions", {}).keys()):
        if t not in targets_val:
            px = float(closes.get(t, np.nan))
            if np.isnan(px) or px <= 0:
                continue
            sh = float(portfolio["positions"][t].get("shares", 0.0))
            val = sh * px
            cash += val - FEE_PER_ORDER
            orders.append(("SELL", t, sh, px, val, "exit"))
            append_trade(trades, {
                "action":"SELL","ticker":t,"date":d_str,"price":px,"shares":sh,"amount":val,"fee":FEE_PER_ORDER,"reason":"exit"
            })
            del portfolio["positions"][t]

    # REBAL / BUY targets (delta threshold)
    equity = cash + sum(float(pos.get("shares",0.0))*float(closes.get(t,np.nan))
                        for t,pos in portfolio.get("positions",{}).items()
                        if (t in closes.index and float(closes.get(t,np.nan))>0))
    for t, tgt in targets_val.items():
        px = float(closes.get(t, np.nan))
        if np.isnan(px) or px <= 0:
            continue
        cur_sh = float(portfolio.get("positions", {}).get(t, {}).get("shares", 0.0))
        cur_val = cur_sh * px
        diff = tgt - cur_val
        if abs(diff) < DELTA_REBAL * equity:
            continue

        if diff < 0 and cur_sh > 0:
            # sell down
            sh = min((-diff)/px, cur_sh)
            val = sh * px
            cash += val - FEE_PER_ORDER
            new_sh = cur_sh - sh
            orders.append(("SELL", t, sh, px, val, "rebalance"))
            append_trade(trades, {"action":"SELL","ticker":t,"date":d_str,"price":px,"shares":sh,"amount":val,"fee":FEE_PER_ORDER,"reason":"rebalance"})
            if new_sh <= 1e-10:
                portfolio["positions"].pop(t, None)
            else:
                portfolio["positions"][t]["shares"] = float(new_sh)
        elif diff > 0:
            # buy up (bounded by cash)
            max_buy = max(cash - FEE_PER_ORDER, 0.0)
            buy_val = min(diff, max_buy)
            if buy_val <= 1e-8:
                continue
            sh = buy_val / px
            cash -= buy_val + FEE_PER_ORDER
            orders.append(("BUY", t, sh, px, buy_val, "rebalance"))
            if t not in portfolio["positions"]:
                portfolio["positions"][t] = {"entry_date": d_str, "entry_price": px, "shares": 0.0}
            portfolio["positions"][t]["shares"] = float(portfolio["positions"][t].get("shares", 0.0) + sh)
            append_trade(trades, {"action":"BUY","ticker":t,"date":d_str,"price":px,"shares":sh,"amount":buy_val,"fee":FEE_PER_ORDER,"reason":"rebalance"})

    portfolio["cash"] = float(cash)

    # Summary + Telegram
    lines_pos = []
    for t, pos in portfolio.get("positions", {}).items():
        px = float(closes.get(t, np.nan))
        sh = float(pos.get("shares", 0.0))
        val = sh*px if (not np.isnan(px) and px>0) else np.nan
        lines_pos.append(f"{t}: {val:.0f}" if np.isfinite(val) else f"{t}: n/a")

    msg = []
    msg.append(f"APEX PROD — PACK_FINAL_V6C (A_rank5) — {d_str}")
    msg.append(f"Cash {portfolio['cash']:.2f} | Pos {pos_val:.2f} | Total {equity:.2f}")
    msg.append("")
    msg.append("TOP 5 MOMENTUM:")
    msg.extend([f"{line}" for line in top_lines] if top_lines else ["(none)"])
    msg.append("")
    msg.append(f"Desired: {', '.join(desired) if desired else '(none)'}")
    msg.append(f"CorrGate: {int(corr_gate_on)} | max_corr {max_corr:.3f}" if np.isfinite(max_corr) else f"CorrGate: {int(corr_gate_on)}")
    msg.append("")
    if orders:
        msg.append("ORDERS (signal close, exec open+1):")
        for a,t,sh,px,val,reason in orders:
            msg.append(f"- {a} {t} {val:.0f} ({reason})")
    else:
        msg.append("ORDERS: none")

    message = "\n".join(msg)
    print(message)
    send_telegram(message)

    save_portfolio(portfolio)
    save_trades(trades)


if __name__ == "__main__":
    main()
