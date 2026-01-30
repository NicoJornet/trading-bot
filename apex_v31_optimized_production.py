"""
APEX CHAMPION — Version Référence
==================================

Version Champion (référence figée) basée sur V33.
Différences principales vs V33:
- Universe U54 (strict)
- Fully invested (plus d'allocation 50/30/20)
- Score momentum: 0.5*R126 + 0.3*R252 + 0.2*R63
- Entry: breakout (close > High60) + trend (close > SMA200)
- Rotation: SwapEdge (EDGE_MULT=1.00, CONFIRM=3, COOLDOWN=1)
- Gates: breadth (55%), corr (63j, 0.65)
- Stops: hard 18%, trailing après MFE 15% puis -5%, trend break SMA200
- Execution: T+1 open, fee 20bps, slippage 5bps

Fichiers:
- portfolio.json
- trades_history.json

Telegram (optionnel):
- env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

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
# CONFIG CHAMPION
# =============================================================================

PARQUET_PATH = os.environ.get("APEX_OHLCV_PARQUET", "ohlcv_champion_u54.parquet")

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR = 100.0

# ===== CHAMPION: EXECUTION & COSTS =====
EXECUTION_MODE = "T+1_OPEN"  # T+1 open (vs paper close dans V33)
FEE_BPS = 20  # 20 bps
SLIPPAGE_BPS = 5  # 5 bps

# ===== CHAMPION: PORTFOLIO =====
MAX_POSITIONS = 3
FULLY_INVESTED = True  # Plus d'allocation 50/30/20

# ===== CHAMPION: ROTATION (SwapEdge) =====
EDGE_MULT = 1.00
CONFIRM_DAYS = 3
COOLDOWN_DAYS = 1

# ===== CHAMPION: STOPS =====
HARD_STOP_PCT = 0.18  # -18%
MFE_TRIGGER_PCT = 0.15  # +15% pour activer trailing
TRAIL_FROM_PEAK_PCT = 0.05  # -5% depuis peak

# ===== CHAMPION: MOMENTUM SCORE =====
# Score = 0.5*R126 + 0.3*R252 + 0.2*R63
R63_WINDOW = 63
R126_WINDOW = 126
R252_WINDOW = 252
SCORE_WEIGHTS = {
    R126_WINDOW: 0.5,
    R252_WINDOW: 0.3,
    R63_WINDOW: 0.2,
}

# ===== CHAMPION: ENTRY SIGNALS =====
SMA200_WINDOW = 200  # Trend filter
HIGH60_WINDOW = 60   # Breakout window

# ===== CHAMPION: GATES =====
BREADTH_THRESHOLD = 0.55  # 55% tickers above SMA200
CORR_WINDOW = 63
CORR_THRESHOLD = 0.65

# Indicators helpers
ATR_PERIOD = 14  # Pour calcul ATR (pas utilisé comme filtre)

# Download window
LOOKBACK_CAL_DAYS = 420

# ===== CHAMPION: UNIVERSE U54 (strict) =====
UNIVERSE_U54 = [
    # Tech Giants
    "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA",
    # Semiconductors
    "AMD", "MU", "ASML", "TSM", "LRCX", "AMAT", "AVGO", "QCOM",
    # Software / Cloud
    "PLTR", "APP", "CRWD", "NET", "DDOG", "ZS", "CRM", "ADBE", "NOW",
    # Emerging Tech
    "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER", "COIN",
    # Crypto exposure
    "MSTR", "MARA", "RIOT",
    # Energy / Nuclear
    "CEG", "VST",
    # Healthcare
    "LLY", "NVO", "UNH", "JNJ", "ABBV", "GILD",
    # Consumer
    "WMT", "COST", "PG", "KO", "MCD",
    # Energy traditional
    "XOM", "CVX",
    # ETFs
    "QQQ", "SPY", "GLD", "SLV",
]

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
        # Champion: swap tracking
        p.setdefault("swap_confirm_tracker", {})  # {ticker: days_confirmed}
        p.setdefault("last_swap_date", {})  # {ticker: last_swap_date} pour cooldown
        return p

    return {
        "currency": "EUR",
        "cash": float(INITIAL_CAPITAL_EUR),
        "initial_capital": float(INITIAL_CAPITAL_EUR),
        "monthly_dca": float(MONTHLY_DCA_EUR),
        "positions": {},
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "last_dca_month": None,
        "swap_confirm_tracker": {},
        "last_swap_date": {},
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
        pivot.columns = pivot.columns.swaplevel(0, 1)
        pivot = pivot.sort_index(axis=1)
        return pivot

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        if df.index.name == "date" or pd.api.types.is_datetime64_any_dtype(df.index):
            pass
        else:
            df.index = pd.to_datetime(df.index)
        
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if set(lvl0).issuperset({"open", "high", "low", "close", "volume"}):
            df.columns = df.columns.swaplevel(0, 1)
        
        subset_tickers = [t for t in tickers if t in df.columns.get_level_values(0)]
        df = df[[t for t in subset_tickers for field in ["open", "high", "low", "close", "volume"] if (t, field) in df.columns]]
        return df

    raise ValueError("Format parquet non reconnu")


def download_yfinance(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download via yfinance, returns MultiIndex (ticker, field).
    """
    if yf is None:
        raise ImportError("yfinance non disponible")
    
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("yfinance: aucune donnée")
    
    df = _standardize_ohlcv_columns(df)
    needed = ["open", "high", "low", "close", "volume"]
    
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if set(lvl0).issuperset(needed):
            df.columns = df.columns.swaplevel(0, 1)
    else:
        if len(tickers) == 1:
            t = tickers[0]
            df = df[needed].copy()
            df.columns = pd.MultiIndex.from_product([[t], needed])
    
    return df


def load_data(tickers: List[str]) -> pd.DataFrame:
    """
    Load OHLCV: parquet if exists, else yfinance.
    """
    if os.path.exists(PARQUET_PATH):
        try:
            return load_ohlcv_parquet(PARQUET_PATH, tickers)
        except Exception as e:
            print(f"Parquet error: {e}, fallback yfinance")
    
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_CAL_DAYS)
    return download_yfinance(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


# =============================================================================
# Indicators
# =============================================================================

def compute_returns(close: pd.Series, windows: List[int]) -> Dict[int, pd.Series]:
    """
    Returns dict: {window: return_series}
    """
    out = {}
    for w in windows:
        out[w] = close.pct_change(w)
    return out


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=1).mean()


def compute_high_rolling(high: pd.Series, window: int) -> pd.Series:
    """Rolling high over window (previous values, not including current)"""
    return high.shift(1).rolling(window, min_periods=1).max()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


# =============================================================================
# FX: EURUSD
# =============================================================================

def get_eurusd(df: pd.DataFrame) -> float:
    """
    Retrieve latest EURUSD=X close from df.
    """
    if ("EURUSD=X", "close") in df.columns:
        ser = df[("EURUSD=X", "close")].dropna()
        if not ser.empty:
            return float(ser.iloc[-1])
    return 1.0


def usd_to_eur(usd: float, eurusd: float) -> float:
    return usd / eurusd if eurusd > 0 else usd


# =============================================================================
# DCA
# =============================================================================

def apply_monthly_dca(portfolio: dict, today: datetime) -> None:
    last_dca_month = portfolio.get("last_dca_month", None)
    current_month = today.strftime("%Y-%m")
    if last_dca_month != current_month:
        dca = float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
        if dca > 0:
            portfolio["cash"] = float(portfolio.get("cash", 0.0)) + dca
            portfolio["last_dca_month"] = current_month


# =============================================================================
# CHAMPION: Momentum Score
# =============================================================================

def compute_momentum_score_champion(close: pd.Series) -> Tuple[float, dict]:
    """
    Champion score = 0.5*R126 + 0.3*R252 + 0.2*R63
    Returns (score, dict_of_returns)
    """
    windows = [R63_WINDOW, R126_WINDOW, R252_WINDOW]
    rets = compute_returns(close, windows)
    
    score = 0.0
    for w, weight in SCORE_WEIGHTS.items():
        r = rets.get(w, pd.Series([np.nan]))
        val = r.iloc[-1] if not r.empty else np.nan
        if not np.isnan(val):
            score += weight * val
    
    return score, {f"R{w}": rets[w].iloc[-1] if not rets[w].empty else np.nan for w in windows}


# =============================================================================
# CHAMPION: Entry Conditions (Breakout + Trend)
# =============================================================================

def check_entry_champion(close: pd.Series, high: pd.Series) -> Tuple[bool, dict]:
    """
    Champion entry:
    - close > SMA200
    - close > High60_prev (breakout)
    
    Returns (eligible, info_dict)
    """
    if len(close) < max(SMA200_WINDOW, HIGH60_WINDOW):
        return False, {"reason": "insufficient_data"}
    
    sma200 = compute_sma(close, SMA200_WINDOW)
    high60_prev = compute_high_rolling(high, HIGH60_WINDOW)
    
    c = close.iloc[-1]
    s200 = sma200.iloc[-1]
    h60 = high60_prev.iloc[-1]
    
    trend_ok = c > s200
    breakout_ok = c > h60
    
    info = {
        "close": c,
        "sma200": s200,
        "high60_prev": h60,
        "trend_ok": trend_ok,
        "breakout_ok": breakout_ok,
    }
    
    eligible = trend_ok and breakout_ok
    if not eligible:
        reasons = []
        if not trend_ok:
            reasons.append("trend_below_sma200")
        if not breakout_ok:
            reasons.append("no_breakout_high60")
        info["reason"] = "+".join(reasons)
    
    return eligible, info


# =============================================================================
# CHAMPION: Breadth Gate
# =============================================================================

def compute_breadth(df: pd.DataFrame, tickers: List[str]) -> Tuple[float, int, int]:
    """
    Breadth = % tickers with close > SMA200
    Returns (breadth_pct, count_above, total)
    """
    count_above = 0
    total = 0
    
    for t in tickers:
        if (t, "close") not in df.columns:
            continue
        close = df[(t, "close")].dropna()
        if len(close) < SMA200_WINDOW:
            continue
        sma200 = compute_sma(close, SMA200_WINDOW)
        if close.iloc[-1] > sma200.iloc[-1]:
            count_above += 1
        total += 1
    
    breadth = count_above / total if total > 0 else 0.0
    return breadth, count_above, total


# =============================================================================
# CHAMPION: Correlation Gate
# =============================================================================

def compute_correlation_matrix(df: pd.DataFrame, tickers: List[str], window: int = CORR_WINDOW) -> pd.DataFrame:
    """
    Compute correlation matrix on returns over window days.
    """
    rets = {}
    for t in tickers:
        if (t, "close") not in df.columns:
            continue
        close = df[(t, "close")].dropna()
        if len(close) < window + 1:
            continue
        r = close.pct_change().iloc[-window:]
        if len(r) >= window // 2:
            rets[t] = r
    
    if len(rets) < 2:
        return pd.DataFrame()
    
    ret_df = pd.DataFrame(rets)
    return ret_df.corr()


def check_correlation_gate(held_tickers: List[str], candidate: str, corr_matrix: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> Tuple[bool, dict]:
    """
    Check if candidate is too correlated with any held ticker.
    Returns (allowed, info)
    """
    if corr_matrix.empty or candidate not in corr_matrix.index:
        return True, {"reason": "no_corr_data"}
    
    max_corr = 0.0
    blocking_ticker = None
    
    for t in held_tickers:
        if t not in corr_matrix.columns:
            continue
        c = corr_matrix.loc[candidate, t]
        if not np.isnan(c) and abs(c) > abs(max_corr):
            max_corr = c
            if abs(c) > threshold:
                blocking_ticker = t
    
    allowed = blocking_ticker is None
    info = {
        "max_corr": max_corr,
        "threshold": threshold,
    }
    if not allowed:
        info["reason"] = f"corr_too_high_with_{blocking_ticker}"
    
    return allowed, info


# =============================================================================
# CHAMPION: SwapEdge Rotation
# =============================================================================

def check_swap_edge(
    portfolio: dict,
    ranked: List[Tuple[str, float]],
    score_map: Dict[str, float],
    today_str: str,
) -> List[Tuple[str, str, str]]:
    """
    Champion rotation: SwapEdge
    - swap if best_score >= worst_score * EDGE_MULT
    - need CONFIRM_DAYS consecutive days
    - COOLDOWN_DAYS after swap
    
    Returns list of (sell_ticker, buy_ticker, reason)
    """
    positions = portfolio.get("positions", {})
    if len(positions) >= MAX_POSITIONS:
        held = list(positions.keys())
        held_scores = [(t, score_map.get(t, -999)) for t in held]
        held_scores.sort(key=lambda x: x[1])
        worst_ticker, worst_score = held_scores[0]
        
        # Find best candidate not held
        best_ticker, best_score = None, -999
        for t, sc in ranked:
            if t not in held:
                best_ticker, best_score = t, sc
                break
        
        if best_ticker is None:
            return []
        
        # Check edge
        if best_score >= worst_score * EDGE_MULT:
            # Check cooldown
            last_swap = portfolio.get("last_swap_date", {}).get(worst_ticker, None)
            if last_swap:
                days_since = (pd.to_datetime(today_str) - pd.to_datetime(last_swap)).days
                if days_since < COOLDOWN_DAYS:
                    return []
            
            # Confirm tracker
            tracker = portfolio.get("swap_confirm_tracker", {})
            key = f"{worst_ticker}->{best_ticker}"
            tracker[key] = tracker.get(key, 0) + 1
            portfolio["swap_confirm_tracker"] = tracker
            
            if tracker[key] >= CONFIRM_DAYS:
                # Reset tracker
                tracker.pop(key, None)
                portfolio["last_swap_date"] = portfolio.get("last_swap_date", {})
                portfolio["last_swap_date"][worst_ticker] = today_str
                
                reason = f"SWAP_EDGE_{CONFIRM_DAYS}d_confirmed"
                return [(worst_ticker, best_ticker, reason)]
        else:
            # Reset tracker if edge no longer valid
            tracker = portfolio.get("swap_confirm_tracker", {})
            keys_to_remove = [k for k in tracker.keys() if k.startswith(f"{worst_ticker}->")]
            for k in keys_to_remove:
                tracker.pop(k, None)
    
    return []


# =============================================================================
# CHAMPION: Fully Invested Allocation
# =============================================================================

def allocate_cash_fully_invested(cash: float, num_slots: int) -> float:
    """
    Champion: fully invested, equal slots.
    """
    if num_slots <= 0:
        return 0.0
    return cash / num_slots


# =============================================================================
# CHAMPION: Compute Costs (bps)
# =============================================================================

def compute_trade_cost(amount_eur: float, fee_bps: float = FEE_BPS, slip_bps: float = SLIPPAGE_BPS) -> float:
    """
    Total cost in EUR = (fee_bps + slip_bps) * amount / 10000
    """
    return (fee_bps + slip_bps) * amount_eur / 10000.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("APEX CHAMPION — Version Référence")
    print("=" * 90)
    
    # Load data
    tickers_with_fx = UNIVERSE_U54 + ["EURUSD=X"]
    df = load_data(tickers_with_fx)
    
    if df.empty:
        print("Aucune donnée disponible")
        return
    
    df = df.sort_index()
    eurusd = get_eurusd(df)
    today = df.index[-1]
    today_str = pd.to_datetime(today).strftime("%Y-%m-%d")
    
    print(f"Date: {today_str}")
    print(f"EURUSD: {eurusd:.4f}")
    print(f"Universe: {len(UNIVERSE_U54)} tickers")
    print()
    
    # Load portfolio & trades
    portfolio = load_portfolio()
    trades = load_trades()
    
    # DCA
    apply_monthly_dca(portfolio, today)
    
    # =====================================================================
    # CHAMPION: Compute indicators & scores
    # =====================================================================
    score_map = {}
    entry_info_map = {}
    
    for t in UNIVERSE_U54:
        if (t, "close") not in df.columns or (t, "high") not in df.columns:
            continue
        
        close = df[(t, "close")].dropna()
        high = df[(t, "high")].dropna()
        
        if len(close) < max(R252_WINDOW, SMA200_WINDOW, HIGH60_WINDOW):
            continue
        
        # Score
        score, rets_dict = compute_momentum_score_champion(close)
        score_map[t] = score
        
        # Entry check
        eligible, info = check_entry_champion(close, high)
        entry_info_map[t] = (eligible, info)
    
    # Rank
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    rank_map = {t: i+1 for i, (t, _) in enumerate(ranked)}
    
    # Breadth gate
    breadth, breadth_above, breadth_total = compute_breadth(df, UNIVERSE_U54)
    breadth_ok = breadth >= BREADTH_THRESHOLD
    
    print(f"BREADTH: {breadth:.2%} ({breadth_above}/{breadth_total}) | Gate: {'PASS' if breadth_ok else 'FAIL'}")
    
    # Correlation matrix
    corr_matrix = compute_correlation_matrix(df, UNIVERSE_U54, CORR_WINDOW)
    
    # Last close USD
    last_close_usd = {}
    for t in UNIVERSE_U54:
        if (t, "close") in df.columns:
            ser = df[(t, "close")].dropna()
            if not ser.empty:
                last_close_usd[t] = float(ser.iloc[-1])
    
    # =====================================================================
    # CHAMPION: 1) Update positions (trailing, trend break)
    # =====================================================================
    positions = portfolio.get("positions", {})
    sells = []
    
    for t in list(positions.keys()):
        pos = positions[t]
        entry_price = float(pos.get("entry_price_eur", 1.0))
        shares = float(pos.get("shares", 0.0))
        entry_date = pos.get("entry_date", today_str)
        
        px_usd = last_close_usd.get(t, np.nan)
        if np.isnan(px_usd) or px_usd <= 0:
            continue
        
        px_eur = usd_to_eur(px_usd, eurusd)
        
        # Update peak/trough
        peak = float(pos.get("peak_price_eur", entry_price))
        trough = float(pos.get("trough_price_eur", entry_price))
        peak = max(peak, px_eur)
        trough = min(trough, px_eur)
        pos["peak_price_eur"] = peak
        pos["trough_price_eur"] = trough
        
        mfe_pct = (peak / entry_price - 1.0) * 100.0
        mae_pct = (trough / entry_price - 1.0) * 100.0
        pos["mfe_pct"] = mfe_pct
        pos["mae_pct"] = mae_pct
        
        pnl_eur = (px_eur - entry_price) * shares
        pnl_pct = (px_eur / entry_price - 1.0) * 100.0
        
        bh = (pd.to_datetime(today) - pd.to_datetime(entry_date)).days
        
        reason = None
        
        # Hard stop
        if pnl_pct <= -HARD_STOP_PCT * 100:
            reason = "HARD_STOP"
        
        # Trailing
        trailing_active = bool(pos.get("trailing_active", False))
        if reason is None and mfe_pct >= MFE_TRIGGER_PCT * 100:
            if not trailing_active:
                pos["trailing_active"] = True
                trailing_active = True
        
        if reason is None and trailing_active:
            dd_from_peak = (px_eur / peak - 1.0)
            if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
                reason = "TRAILING"
        
        # Trend break (close < SMA200)
        if reason is None and (t, "close") in df.columns:
            close_series = df[(t, "close")].dropna()
            if len(close_series) >= SMA200_WINDOW:
                sma200 = compute_sma(close_series, SMA200_WINDOW)
                if close_series.iloc[-1] < sma200.iloc[-1]:
                    reason = "TREND_BREAK_SMA200"
        
        if reason is not None:
            value_eur = px_eur * shares
            cost = compute_trade_cost(value_eur)
            net_pnl = pnl_eur - cost
            
            sells.append({
                "ticker": t,
                "price_eur": px_eur,
                "shares": shares,
                "value_eur": value_eur,
                "pnl_eur": net_pnl,
                "pnl_pct": pnl_pct,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "bars_held": bh,
                "reason": reason,
                "rank": rank_map.get(t, 999),
                "score": score_map.get(t, 0.0),
                "entry_date": entry_date,
                "entry_price_eur": entry_price,
            })
        
        positions[t] = pos
    
    # Execute sells
    for s in sells:
        t = s["ticker"]
        proceeds = float(s["value_eur"]) - compute_trade_cost(s["value_eur"])
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + proceeds
        
        append_trade(trades, {
            "action": "SELL",
            "ticker": t,
            "date": today_str,
            "price_eur": float(s["price_eur"]),
            "shares": float(s["shares"]),
            "amount_eur": float(s["value_eur"]),
            "fee_bps": float(FEE_BPS),
            "slippage_bps": float(SLIPPAGE_BPS),
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
    # CHAMPION: 2) SwapEdge Rotation
    # =====================================================================
    swaps = check_swap_edge(portfolio, ranked, score_map, today_str)
    
    for sell_ticker, buy_ticker, swap_reason in swaps:
        # Sell
        pos = portfolio["positions"][sell_ticker]
        entry_price = float(pos.get("entry_price_eur", 1.0))
        shares = float(pos.get("shares", 0.0))
        entry_date = pos.get("entry_date", today_str)
        
        px_usd = last_close_usd.get(sell_ticker, np.nan)
        if np.isnan(px_usd) or px_usd <= 0:
            continue
        px_eur = usd_to_eur(px_usd, eurusd)
        
        value_eur = px_eur * shares
        pnl_eur = (px_eur - entry_price) * shares
        pnl_pct = (px_eur / entry_price - 1.0) * 100.0
        mfe_pct = float(pos.get("mfe_pct", 0.0))
        mae_pct = float(pos.get("mae_pct", 0.0))
        bh = (pd.to_datetime(today) - pd.to_datetime(entry_date)).days
        
        cost = compute_trade_cost(value_eur)
        net_pnl = pnl_eur - cost
        proceeds = value_eur - cost
        
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + proceeds
        
        append_trade(trades, {
            "action": "SELL",
            "ticker": sell_ticker,
            "date": today_str,
            "price_eur": float(px_eur),
            "shares": float(shares),
            "amount_eur": float(value_eur),
            "fee_bps": float(FEE_BPS),
            "slippage_bps": float(SLIPPAGE_BPS),
            "reason": swap_reason,
            "pnl_eur": float(net_pnl),
            "pnl_pct": float(pnl_pct),
            "mfe_pct": float(mfe_pct),
            "mae_pct": float(mae_pct),
            "bars_held": int(bh),
            "rank": rank_map.get(sell_ticker, 999),
            "score": score_map.get(sell_ticker, 0.0),
        })
        
        del portfolio["positions"][sell_ticker]
        
        # Buy (will be executed below in normal buy logic)
        # We just freed a slot
    
    # =====================================================================
    # CHAMPION: 3) BUY signals (breakout + trend + gates)
    # =====================================================================
    buys = []
    held = set(portfolio.get("positions", {}).keys())
    slots = MAX_POSITIONS - len(held)
    
    cash = float(portfolio.get("cash", 0.0))
    
    if slots > 0 and cash > 50 and breadth_ok:
        for t, sc in ranked:
            if slots <= 0:
                break
            
            r = rank_map.get(t, 999)
            if r > MAX_POSITIONS:
                continue
            if t in held:
                continue
            
            # Entry check
            eligible, info = entry_info_map.get(t, (False, {}))
            if not eligible:
                continue
            
            # Corr gate
            corr_ok, corr_info = check_correlation_gate(list(held), t, corr_matrix, CORR_THRESHOLD)
            if not corr_ok:
                continue
            
            px_usd = last_close_usd.get(t, np.nan)
            if np.isnan(px_usd) or px_usd <= 0:
                continue
            px_eur = usd_to_eur(px_usd, eurusd)
            
            alloc = allocate_cash_fully_invested(cash, slots)
            alloc = min(alloc, max(0.0, cash - 10.0))
            if alloc < 50:
                continue
            
            shares = alloc / px_eur
            cost = alloc + compute_trade_cost(alloc)
            if cost > cash:
                continue
            
            buys.append({
                "ticker": t,
                "rank": r,
                "score": float(sc),
                "price_eur": px_eur,
                "shares": shares,
                "amount_eur": alloc,
                "trend_ok": info.get("trend_ok", False),
                "breakout_ok": info.get("breakout_ok", False),
            })
            
            # Execute buy
            cash -= cost
            portfolio["cash"] = cash
            portfolio["positions"][t] = {
                "entry_date": today_str,
                "entry_price_eur": float(px_eur),
                "shares": float(shares),
                "initial_amount_eur": float(alloc),
                "amount_invested_eur": float(alloc),
                "peak_price_eur": float(px_eur),
                "trough_price_eur": float(px_eur),
                "mfe_pct": 0.0,
                "mae_pct": 0.0,
                "trailing_active": False,
                "rank": int(r),
                "score": float(sc),
            }
            
            append_trade(trades, {
                "action": "BUY",
                "ticker": t,
                "date": today_str,
                "price_eur": float(px_eur),
                "shares": float(shares),
                "amount_eur": float(alloc),
                "fee_bps": float(FEE_BPS),
                "slippage_bps": float(SLIPPAGE_BPS),
                "reason": f"CHAMPION_RANK{r}_BREAKOUT+TREND",
                "rank": int(r),
                "score": float(sc),
            })
            
            held.add(t)
            slots -= 1
    
    # =====================================================================
    # 4) Portfolio summary + Telegram (Champion format)
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
    
    start_date = pd.to_datetime(portfolio.get("start_date", today_str))
    months = (today.year - start_date.year) * 12 + (today.month - start_date.month)
    invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL_EUR)) + max(0, months) * float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
    pnl_total = total - invested
    pnl_total_pct = (total / invested - 1.0) * 100.0 if invested > 0 else 0.0
    
    # Telegram message (Champion format)
    msg = []
    msg.append(f"APEX CHAMPION — {today_str}")
    msg.append(f"EURUSD {eurusd:.4f}")
    msg.append(f"Cash {cash:.2f}€ | Pos {pos_value:.2f}€ | Total {total:.2f}€")
    msg.append(f"Invested~ {invested:.2f}€ | PnL {pnl_total:+.2f}€ ({pnl_total_pct:+.1f}%)")
    msg.append("")
    
    msg.append("GATES STATUS:")
    msg.append(f"- Breadth: {breadth:.1%} (>={BREADTH_THRESHOLD:.0%}) {'✓' if breadth_ok else '✗'}")
    msg.append(f"- Corr: window={CORR_WINDOW}d, thr={CORR_THRESHOLD}")
    msg.append("")
    
    msg.append("ACTIONS:")
    if sells:
        for s in sells:
            msg.append(f"SELL {s['ticker']} — {s['reason']} | PnL {s['pnl_pct']:+.1f}% | MFE {s['mfe_pct']:+.1f}% | Hold {s['bars_held']}d")
    if swaps:
        for sell_t, buy_t, swap_r in swaps:
            msg.append(f"SWAP {sell_t} -> {buy_t} ({swap_r})")
    if buys:
        for b in buys:
            msg.append(f"BUY  {b['ticker']} (#{b['rank']}) amt {b['amount_eur']:.0f}€ | score {b['score']:.3f}")
    if not sells and not swaps and not buys:
        msg.append("HOLD — no action")
    msg.append("")
    
    msg.append("POSITIONS:")
    msg.extend(lines_pos if lines_pos else ["- (none)"])
    msg.append("")
    
    msg.append("TOP 5 MOMENTUM:")
    for i, (t, sc) in enumerate(ranked[:5], 1):
        px_usd = last_close_usd.get(t, np.nan)
        px_eur = usd_to_eur(px_usd, eurusd) if not np.isnan(px_usd) else np.nan
        eligible, info = entry_info_map.get(t, (False, {}))
        status = "✓" if eligible else f"✗({info.get('reason', 'unknown')})"
        msg.append(f"{i}. {t} score {sc:.3f} px {px_eur:.2f}€ {status}")
    
    message = "\n".join(msg)
    
    # Save
    save_portfolio(portfolio)
    save_trades(trades)
    
    # Output
    print(message)
    send_telegram(message)
    
    print("=" * 90)
    print("✅ CHAMPION Run terminé | portfolio.json + trades_history.json mis à jour")
    print("=" * 90)


if __name__ == "__main__":
    main()
