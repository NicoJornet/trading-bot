"""
APEX CHAMPION — Version Référence (YFINANCE ONLY)
=================================================

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

Persistance:
- portfolio.json
- trades_history.json

Telegram (optionnel):
- env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID

NOTE:
- Cette version ne lit PLUS aucun parquet : données via yfinance uniquement.
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
# CONFIG CHAMPION
# =============================================================================

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR = 100.0

# ===== CHAMPION: EXECUTION & COSTS =====
EXECUTION_MODE = "T+1_OPEN" # doc: T+1 open (vs paper close dans V33)
FEE_BPS = 20
SLIPPAGE_BPS = 5

# ===== CHAMPION: PORTFOLIO =====
MAX_POSITIONS = 3
FULLY_INVESTED = True

# ===== CHAMPION: ROTATION (SwapEdge) =====
EDGE_MULT = 1.00
CONFIRM_DAYS = 3
COOLDOWN_DAYS = 1

# ===== CHAMPION: STOPS =====
HARD_STOP_PCT = 0.18
MFE_TRIGGER_PCT = 0.15
TRAIL_FROM_PEAK_PCT = 0.05

# ===== CHAMPION: MOMENTUM SCORE =====
R63_WINDOW = 63
R126_WINDOW = 126
R252_WINDOW = 252
SCORE_WEIGHTS = {
 R126_WINDOW: 0.5,
 R252_WINDOW: 0.3,
 R63_WINDOW: 0.2,
}

# ===== CHAMPION: ENTRY SIGNALS =====
SMA200_WINDOW = 200
HIGH60_WINDOW = 60

# ===== CHAMPION: GATES =====
BREADTH_THRESHOLD = 0.55
CORR_WINDOW = 63
CORR_THRESHOLD = 0.65

# Download window
LOOKBACK_CAL_DAYS = 420 # ~ 20 mois calendaires, suffisant pour SMA200 + R252

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
 p.setdefault("swap_confirm_tracker", {}) # {pair: days_confirmed}
 p.setdefault("last_swap_date", {}) # {ticker: last_swap_date}
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
# Data loading (yfinance only)
# =============================================================================

def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
 """
 Normalize OHLCV columns to:
 - multiindex: (ticker, field) where field lower-case in open/high/low/close/volume
 Works with yfinance outputs (field,ticker) or (ticker,field).
 """
 out = df.copy()

 if isinstance(out.columns, pd.MultiIndex):
 lvl0 = [str(x).lower() for x in out.columns.get_level_values(0)]
 fields = {"open", "high", "low", "close", "volume", "adj close"}

 # If it is (field, ticker), swap to (ticker, field)
 if ({"open", "high", "low", "close", "volume"}.issubset(set(lvl0))
 or fields.issubset(set(lvl0))):
 out = out.swaplevel(0, 1, axis=1)

 out.columns = pd.MultiIndex.from_tuples(
 [(str(t), str(f).lower()) for (t, f) in out.columns],
 names=["ticker", "field"]
 )

 out = out.rename(columns={"adj close": "close"}, level="field")
 return out

 # Single index (rare when 1 ticker) -> keep & rename to lower, later we wrap in MultiIndex
 cols = {str(c).lower(): c for c in out.columns}
 ren = {}
 for target in ["open", "high", "low", "close", "volume"]:
 if target in cols:
 ren[cols[target]] = target
 elif target == "close" and "adj close" in cols:
 ren[cols["adj close"]] = "close"
 return out.rename(columns=ren)


def download_yfinance(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
 """
 Download via yfinance, returns MultiIndex (ticker, field).
 """
 if yf is None:
 raise ImportError("yfinance non disponible")

 df = yf.download(
 tickers=tickers,
 start=start_date,
 end=end_date,
 auto_adjust=False,
 progress=False,
 group_by="column",
 threads=True,
 )

 if df is None or df.empty:
 raise ValueError("yfinance: aucune donnée")

 df = _standardize_ohlcv_columns(df)

 # Force MultiIndex if single ticker returned single-index
 if not isinstance(df.columns, pd.MultiIndex):
 needed = ["open", "high", "low", "close", "volume"]
 df = df[needed].copy()
 df.columns = pd.MultiIndex.from_product([[tickers[0]], needed], names=["ticker", "field"])

 return df


def load_data(tickers: List[str]) -> pd.DataFrame:
 end = datetime.now()
 start = end - timedelta(days=LOOKBACK_CAL_DAYS)
 return download_yfinance(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


# =============================================================================
# Indicators
# =============================================================================

def compute_returns(close: pd.Series, windows: List[int]) -> Dict[int, pd.Series]:
 return {w: close.pct_change(w) for w in windows}


def compute_sma(close: pd.Series, window: int) -> pd.Series:
 return close.rolling(window, min_periods=1).mean()


def compute_high_rolling(high: pd.Series, window: int) -> pd.Series:
 return high.shift(1).rolling(window, min_periods=1).max()


# =============================================================================
# FX: EURUSD
# =============================================================================

def get_eurusd(df: pd.DataFrame) -> float:
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
# Momentum Score
# =============================================================================

def compute_momentum_score_champion(close: pd.Series) -> Tuple[float, dict]:
 windows = [R63_WINDOW, R126_WINDOW, R252_WINDOW]
 rets = compute_returns(close, windows)

 score = 0.0
 for w, weight in SCORE_WEIGHTS.items():
 r = rets.get(w, None)
 val = r.iloc[-1] if r is not None and len(r) else np.nan
 if not np.isnan(val):
 score += weight * float(val)

 last = {f"R{w}": (float(rets[w].iloc[-1]) if len(rets[w]) else np.nan) for w in windows}
 return float(score), last


# =============================================================================
# Entry: Breakout + Trend
# =============================================================================

def check_entry_champion(close: pd.Series, high: pd.Series) -> Tuple[bool, dict]:
 if len(close) < max(SMA200_WINDOW, HIGH60_WINDOW):
 return False, {"reason": "insufficient_data"}

 sma200 = compute_sma(close, SMA200_WINDOW)
 high60_prev = compute_high_rolling(high, HIGH60_WINDOW)

 c = float(close.iloc[-1])
 s200 = float(sma200.iloc[-1])
 h60 = float(high60_prev.iloc[-1])

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
# Breadth Gate
# =============================================================================

def compute_breadth(df: pd.DataFrame, tickers: List[str]) -> Tuple[float, int, int]:
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
 return float(breadth), int(count_above), int(total)


# =============================================================================
# Correlation Gate
# =============================================================================

def compute_correlation_matrix(df: pd.DataFrame, tickers: List[str], window: int = CORR_WINDOW) -> pd.DataFrame:
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

 return pd.DataFrame(rets).corr()


def check_correlation_gate(
 held_tickers: List[str],
 candidate: str,
 corr_matrix: pd.DataFrame,
 threshold: float = CORR_THRESHOLD
) -> Tuple[bool, dict]:
 if corr_matrix.empty or candidate not in corr_matrix.index:
 return True, {"reason": "no_corr_data"}

 max_corr = 0.0
 blocking = None

 for t in held_tickers:
 if t not in corr_matrix.columns:
 continue
 c = corr_matrix.loc[candidate, t]
 if not np.isnan(c) and abs(c) > abs(max_corr):
 max_corr = float(c)
 if abs(c) > threshold:
 blocking = t

 allowed = blocking is None
 info = {"max_corr": float(max_corr), "threshold": float(threshold)}
 if not allowed:
 info["reason"] = f"corr_too_high_with_{blocking}"
 return allowed, info


# =============================================================================
# SwapEdge Rotation
# =============================================================================

def check_swap_edge(
 portfolio: dict,
 ranked: List[Tuple[str, float]],
 score_map: Dict[str, float],
 today_str: str,
) -> List[Tuple[str, str, str]]:
 positions = portfolio.get("positions", {})
 if len(positions) < MAX_POSITIONS:
 return []

 held = list(positions.keys())
 held_scores = [(t, score_map.get(t, -999.0)) for t in held]
 held_scores.sort(key=lambda x: x[1])
 worst_ticker, worst_score = held_scores[0]

 best_ticker, best_score = None, -999.0
 for t, sc in ranked:
 if t not in positions:
 best_ticker, best_score = t, sc
 break

 if best_ticker is None:
 return []

 if best_score >= worst_score * EDGE_MULT:
 last_swap = portfolio.get("last_swap_date", {}).get(worst_ticker, None)
 if last_swap:
 days_since = (pd.to_datetime(today_str) - pd.to_datetime(last_swap)).days
 if days_since < COOLDOWN_DAYS:
 return []

 tracker = portfolio.get("swap_confirm_tracker", {})
 key = f"{worst_ticker}->{best_ticker}"
 tracker[key] = tracker.get(key, 0) + 1
 portfolio["swap_confirm_tracker"] = tracker

 if tracker[key] >= CONFIRM_DAYS:
 tracker.pop(key, None)
 portfolio["last_swap_date"] = portfolio.get("last_swap_date", {})
 portfolio["last_swap_date"][worst_ticker] = today_str
 return [(worst_ticker, best_ticker, f"SWAP_EDGE_{CONFIRM_DAYS}d_confirmed")]
 else:
 tracker = portfolio.get("swap_confirm_tracker", {})
 keys_to_remove = [k for k in tracker.keys() if k.startswith(f"{worst_ticker}->")]
 for k in keys_to_remove:
 tracker.pop(k, None)

 return []


# =============================================================================
# Allocation fully invested (equal slots)
# =============================================================================

def allocate_cash_fully_invested(cash: float, num_slots: int) -> float:
 if num_slots <= 0:
 return 0.0
 return cash / num_slots


# =============================================================================
# Costs
# =============================================================================

def compute_trade_cost(amount_eur: float, fee_bps: float = FEE_BPS, slip_bps: float = SLIPPAGE_BPS) -> float:
 return (fee_bps + slip_bps) * amount_eur / 10000.0


# =============================================================================
# MAIN
# =============================================================================

def main():
 print("=" * 90)
 print("APEX CHAMPION — Version Référence (YFINANCE ONLY)")
 print("=" * 90)

 tickers_with_fx = UNIVERSE_U54 + ["EURUSD=X"]
 df = load_data(tickers_with_fx)

 if df is None or df.empty:
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

 portfolio = load_portfolio()
 trades = load_trades()

 apply_monthly_dca(portfolio, today)

 # Scores + entry eligibility
 score_map: Dict[str, float] = {}
 entry_info_map: Dict[str, Tuple[bool, dict]] = {}

 for t in UNIVERSE_U54:
 if (t, "close") not in df.columns or (t, "high") not in df.columns:
 continue

 close = df[(t, "close")].dropna()
 high = df[(t, "high")].dropna()

 if len(close) < max(R252_WINDOW, SMA200_WINDOW, HIGH60_WINDOW):
 continue

 score, _ = compute_momentum_score_champion(close)
 score_map[t] = float(score)

 eligible, info = check_entry_champion(close, high)
 entry_info_map[t] = (eligible, info)

 ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
 rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}

 # Breadth gate
 breadth, breadth_above, breadth_total = compute_breadth(df, UNIVERSE_U54)
 breadth_ok = breadth >= BREADTH_THRESHOLD
 print(f"BREADTH: {breadth:.2%} ({breadth_above}/{breadth_total}) | Gate: {'PASS' if breadth_ok else 'FAIL'}")

 # Corr matrix
 corr_matrix = compute_correlation_matrix(df, UNIVERSE_U54, CORR_WINDOW)

 # Last close USD
 last_close_usd: Dict[str, float] = {}
 for t in UNIVERSE_U54:
 if (t, "close") in df.columns:
 ser = df[(t, "close")].dropna()
 if not ser.empty:
 last_close_usd[t] = float(ser.iloc[-1])

 # =====================================================================
 # 1) Update positions (hard stop / trailing / trend break)
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

 peak = float(pos.get("peak_price_eur", entry_price))
 trough = float(pos.get("trough_price_eur", entry_price))
 peak = max(peak, px_eur)
 trough = min(trough, px_eur)
 pos["peak_price_eur"] = peak
 pos["trough_price_eur"] = trough

 mfe_pct = (peak / entry_price - 1.0) * 100.0
 mae_pct = (trough / entry_price - 1.0) * 100.0
 pos["mfe_pct"] = float(mfe_pct)
 pos["mae_pct"] = float(mae_pct)

 pnl_eur = (px_eur - entry_price) * shares
 pnl_pct = (px_eur / entry_price - 1.0) * 100.0

 bh = (pd.to_datetime(today) - pd.to_datetime(entry_date)).days

 reason = None

 # Hard stop
 if pnl_pct <= -HARD_STOP_PCT * 100:
 reason = "HARD_STOP"

 # Trailing activation
 trailing_active = bool(pos.get("trailing_active", False))
 if reason is None and mfe_pct >= MFE_TRIGGER_PCT * 100:
 if not trailing_active:
 pos["trailing_active"] = True
 trailing_active = True

 # Trailing exit
 if reason is None and trailing_active:
 dd_from_peak = (px_eur / peak - 1.0)
 if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
 reason = "TRAILING"

 # Trend break SMA200
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
 "price_eur": float(px_eur),
 "shares": float(shares),
 "value_eur": float(value_eur),
 "pnl_eur": float(net_pnl),
 "pnl_pct": float(pnl_pct),
 "mfe_pct": float(mfe_pct),
 "mae_pct": float(mae_pct),
 "bars_held": int(bh),
 "reason": reason,
 "rank": int(rank_map.get(t, 999)),
 "score": float(score_map.get(t, 0.0)),
 })

 positions[t] = pos

 # Execute sells
 for s in sells:
 t = s["ticker"]
 proceeds = float(s["value_eur"]) - compute_trade_cost(float(s["value_eur"]))
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
 # 2) SwapEdge (sell only; buy handled later)
 # =====================================================================
 swaps = check_swap_edge(portfolio, ranked, score_map, today_str)

 for sell_ticker, buy_ticker, swap_reason in swaps:
 if sell_ticker not in portfolio.get("positions", {}):
 continue

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
 "rank": int(rank_map.get(sell_ticker, 999)),
 "score": float(score_map.get(sell_ticker, 0.0)),
 })

 del portfolio["positions"][sell_ticker]
 # buy handled by standard buy loop

 # =====================================================================
 # 3) BUY signals
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

 eligible, info = entry_info_map.get(t, (False, {}))
 if not eligible:
 continue

 corr_ok, _ = check_correlation_gate(list(held), t, corr_matrix, CORR_THRESHOLD)
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
 "rank": int(r),
 "score": float(sc),
 "price_eur": float(px_eur),
 "shares": float(shares),
 "amount_eur": float(alloc),
 })

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
 "reason": f"CHAMPION_RANK{int(r)}_BREAKOUT+TREND",
 "rank": int(r),
 "score": float(sc),
 })

 held.add(t)
 slots -= 1

 # =====================================================================
 # 4) Portfolio summary + Telegram
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
 msg.append(f"BUY {b['ticker']} (#{b['rank']}) amt {b['amount_eur']:.0f}€ | score {b['score']:.3f}")
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

 save_portfolio(portfolio)
 save_trades(trades)

 print(message)
 send_telegram(message)

 print("=" * 90)
 print(" CHAMPION Run terminé | portfolio.json + trades_history.json mis à jour")
 print("=" * 90)


if __name__ == "__main__":
 main()

Palmino NicolasTél. : 06 73 20 76 70
