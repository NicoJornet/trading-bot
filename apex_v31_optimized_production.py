"""
APEX CHAMPION — Version Référence (YFINANCE ONLY) — FIXED
========================================================

 Objectif: script PROD stable (GitHub Actions) pour signal/rotation daily.
 Données: yfinance uniquement (plus de parquet)
 Fixes: compat MultiIndex yfinance + aucune indentation fragile

Persistance:
- portfolio.json
- trades_history.json

Telegram (optionnel):
- env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
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

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR = 100.0

# Execution & costs (bps)
EXECUTION_MODE = "T+1_OPEN"
FEE_BPS = 20.0
SLIPPAGE_BPS = 5.0

# Portfolio
MAX_POSITIONS = 3
FULLY_INVESTED = True

# Rotation (SwapEdge)
EDGE_MULT = 1.00
CONFIRM_DAYS = 3
COOLDOWN_DAYS = 1

# Stops
HARD_STOP_PCT = 0.18
MFE_TRIGGER_PCT = 0.15
TRAIL_FROM_PEAK_PCT = 0.05

# Momentum score windows
R63_WINDOW = 63
R126_WINDOW = 126
R252_WINDOW = 252
SCORE_WEIGHTS = {R126_WINDOW: 0.5, R252_WINDOW: 0.3, R63_WINDOW: 0.2}

# Entry filters
SMA200_WINDOW = 200
HIGH60_WINDOW = 60

# Gates
BREADTH_THRESHOLD = 0.55
CORR_WINDOW = 63
CORR_THRESHOLD = 0.65

# Download history (cal days)
LOOKBACK_CAL_DAYS = 420

UNIVERSE_U54 = [
 "NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA",
 "AMD", "MU", "ASML", "TSM", "LRCX", "AMAT", "AVGO", "QCOM",
 "PLTR", "APP", "CRWD", "NET", "DDOG", "ZS", "CRM", "ADBE", "NOW",
 "RKLB", "SHOP", "ABNB", "VRT", "SMCI", "UBER", "COIN",
 "MSTR", "MARA", "RIOT",
 "CEG", "VST",
 "LLY", "NVO", "UNH", "JNJ", "ABBV", "GILD",
 "WMT", "COST", "PG", "KO", "MCD",
 "XOM", "CVX",
 "QQQ", "SPY", "GLD", "SLV",
]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# =============================================================================
# UTIL: safe time
# =============================================================================

def _now_str() -> str:
 return datetime.now().strftime("%Y-%m-%d %H:%M")


# =============================================================================
# IO: portfolio + trades (robuste, pas d'indentation fragile)
# =============================================================================

def load_portfolio() -> dict:
 if os.path.exists(PORTFOLIO_FILE):
 with open(PORTFOLIO_FILE, "r") as f:
 p = json.load(f)

 p.setdefault("currency", "EUR")
 p.setdefault("cash", float(INITIAL_CAPITAL_EUR))
 p.setdefault("initial_capital", float(INITIAL_CAPITAL_EUR))
 p.setdefault("monthly_dca", float(MONTHLY_DCA_EUR))
 p.setdefault("positions", {})
 p.setdefault("start_date", datetime.now().strftime("%Y-%m-%d"))
 p.setdefault("last_dca_month", None)
 p.setdefault("swap_confirm_tracker", {})
 p.setdefault("last_swap_date", {})
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
# Data: yfinance only (MultiIndex safe)
# =============================================================================

def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
 """
 Convert yfinance output to MultiIndex (ticker, field) with field in:
 open/high/low/close/volume (lowercase).
 Works for (field,ticker) or (ticker,field).
 """
 out = df.copy()

 if isinstance(out.columns, pd.MultiIndex):
 lvl0 = [str(x).lower() for x in out.columns.get_level_values(0)]
 fields = {"open", "high", "low", "close", "volume", "adj close"}

 # If (field, ticker) => swap to (ticker, field)
 if {"open", "high", "low", "close", "volume"}.issubset(set(lvl0)) or fields.issubset(set(lvl0)):
 out = out.swaplevel(0, 1, axis=1)

 out.columns = pd.MultiIndex.from_tuples(
 [(str(t), str(f).lower()) for (t, f) in out.columns],
 names=["ticker", "field"]
 )

 out = out.rename(columns={"adj close": "close"}, level="field")
 return out

 # Single ticker sometimes returns single-level columns
 cols = {str(c).lower(): c for c in out.columns}
 ren = {}
 for target in ["open", "high", "low", "close", "volume"]:
 if target in cols:
 ren[cols[target]] = target
 elif target == "close" and "adj close" in cols:
 ren[cols["adj close"]] = "close"
 return out.rename(columns=ren)


def download_yfinance(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
 if yf is None:
 raise ImportError("yfinance non disponible. Ajoute yfinance dans requirements.txt")

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
 raise ValueError("yfinance: aucune donnée téléchargée")

 df = _standardize_ohlcv_columns(df)

 # Force MultiIndex if single-index
 if not isinstance(df.columns, pd.MultiIndex):
 needed = ["open", "high", "low", "close", "volume"]
 if not all(c in df.columns for c in needed):
 raise ValueError(f"Colonnes manquantes (single-ticker): {df.columns}")
 t = tickers[0]
 df = df[needed].copy()
 df.columns = pd.MultiIndex.from_product([[t], needed], names=["ticker", "field"])

 df = df.sort_index()
 return df


def load_data(tickers: List[str]) -> pd.DataFrame:
 end = datetime.now()
 start = end - timedelta(days=LOOKBACK_CAL_DAYS)
 return download_yfinance(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


# =============================================================================
# Indicators
# =============================================================================

def compute_sma(close: pd.Series, window: int) -> pd.Series:
 return close.rolling(window, min_periods=1).mean()


def compute_high_rolling(high: pd.Series, window: int) -> pd.Series:
 return high.shift(1).rolling(window, min_periods=1).max()


def compute_momentum_score(close: pd.Series) -> Tuple[float, dict]:
 r63 = close.pct_change(R63_WINDOW)
 r126 = close.pct_change(R126_WINDOW)
 r252 = close.pct_change(R252_WINDOW)

 v63 = float(r63.iloc[-1]) if len(r63) and not np.isnan(r63.iloc[-1]) else np.nan
 v126 = float(r126.iloc[-1]) if len(r126) and not np.isnan(r126.iloc[-1]) else np.nan
 v252 = float(r252.iloc[-1]) if len(r252) and not np.isnan(r252.iloc[-1]) else np.nan

 score = 0.0
 if not np.isnan(v126):
 score += SCORE_WEIGHTS[R126_WINDOW] * v126
 if not np.isnan(v252):
 score += SCORE_WEIGHTS[R252_WINDOW] * v252
 if not np.isnan(v63):
 score += SCORE_WEIGHTS[R63_WINDOW] * v63

 return float(score), {"R63": v63, "R126": v126, "R252": v252}


def check_entry(close: pd.Series, high: pd.Series) -> Tuple[bool, dict]:
 if len(close) < max(SMA200_WINDOW, HIGH60_WINDOW):
 return False, {"reason": "insufficient_data"}

 sma200 = compute_sma(close, SMA200_WINDOW)
 high60_prev = compute_high_rolling(high, HIGH60_WINDOW)

 c = float(close.iloc[-1])
 s200 = float(sma200.iloc[-1])
 h60 = float(high60_prev.iloc[-1])

 trend_ok = c > s200
 breakout_ok = c > h60
 ok = trend_ok and breakout_ok

 info = {
 "close": c,
 "sma200": s200,
 "high60_prev": h60,
 "trend_ok": trend_ok,
 "breakout_ok": breakout_ok,
 }
 if not ok:
 reasons = []
 if not trend_ok:
 reasons.append("trend_below_sma200")
 if not breakout_ok:
 reasons.append("no_breakout_high60")
 info["reason"] = "+".join(reasons)

 return ok, info


# =============================================================================
# Gates
# =============================================================================

def compute_breadth(df: pd.DataFrame, tickers: List[str]) -> Tuple[float, int, int]:
 above = 0
 total = 0
 for t in tickers:
 if (t, "close") not in df.columns:
 continue
 close = df[(t, "close")].dropna()
 if len(close) < SMA200_WINDOW:
 continue
 sma200 = compute_sma(close, SMA200_WINDOW)
 if close.iloc[-1] > sma200.iloc[-1]:
 above += 1
 total += 1
 breadth = above / total if total > 0 else 0.0
 return float(breadth), int(above), int(total)


def compute_correlation_matrix(df: pd.DataFrame, tickers: List[str], window: int) -> pd.DataFrame:
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


def corr_ok(held: List[str], candidate: str, corr_matrix: pd.DataFrame, threshold: float) -> bool:
 if corr_matrix.empty or candidate not in corr_matrix.index:
 return True
 for h in held:
 if h not in corr_matrix.columns:
 continue
 c = corr_matrix.loc[candidate, h]
 if not np.isnan(c) and abs(float(c)) > threshold:
 return False
 return True


# =============================================================================
# Rotation: SwapEdge
# =============================================================================

def check_swap_edge(
 portfolio: dict,
 ranked: List[Tuple[str, float]],
 score_map: Dict[str, float],
 today_str: str
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
 last_swap = portfolio.get("last_swap_date", {}).get(worst_ticker)
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
 portfolio.setdefault("last_swap_date", {})
 portfolio["last_swap_date"][worst_ticker] = today_str
 return [(worst_ticker, best_ticker, f"SWAP_EDGE_{CONFIRM_DAYS}d_confirmed")]
 else:
 tracker = portfolio.get("swap_confirm_tracker", {})
 for k in list(tracker.keys()):
 if k.startswith(f"{worst_ticker}->"):
 tracker.pop(k, None)

 return []


# =============================================================================
# Allocation + costs
# =============================================================================

def allocate_cash_equal(cash: float, slots: int) -> float:
 return cash / slots if slots > 0 else 0.0


def trade_cost(amount_eur: float) -> float:
 return (FEE_BPS + SLIPPAGE_BPS) * amount_eur / 10000.0


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

def apply_monthly_dca(portfolio: dict, today: pd.Timestamp) -> None:
 last = portfolio.get("last_dca_month")
 cur = today.strftime("%Y-%m")
 if last != cur:
 dca = float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
 if dca > 0:
 portfolio["cash"] = float(portfolio.get("cash", 0.0)) + dca
 portfolio["last_dca_month"] = cur


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
 print("=" * 90)
 print("APEX CHAMPION — PROD (YFINANCE ONLY) — FIXED")
 print("=" * 90)

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

 portfolio = load_portfolio()
 trades = load_trades()

 apply_monthly_dca(portfolio, pd.to_datetime(today))

 # Scores + entry checks
 score_map: Dict[str, float] = {}
 entry_map: Dict[str, Tuple[bool, dict]] = {}

 for t in UNIVERSE_U54:
 if (t, "close") not in df.columns or (t, "high") not in df.columns:
 continue

 close = df[(t, "close")].dropna()
 high = df[(t, "high")].dropna()

 if len(close) < max(R252_WINDOW, SMA200_WINDOW, HIGH60_WINDOW):
 continue

 sc, _ = compute_momentum_score(close)
 score_map[t] = sc

 ok, info = check_entry(close, high)
 entry_map[t] = (ok, info)

 ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
 rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}

 breadth, above, total = compute_breadth(df, UNIVERSE_U54)
 breadth_ok = breadth >= BREADTH_THRESHOLD

 print(f"BREADTH: {breadth:.2%} ({above}/{total}) | Gate: {'PASS' if breadth_ok else 'FAIL'}")

 corr_matrix = compute_correlation_matrix(df, UNIVERSE_U54, CORR_WINDOW)

 # Last close USD
 last_close_usd: Dict[str, float] = {}
 for t in UNIVERSE_U54:
 if (t, "close") in df.columns:
 ser = df[(t, "close")].dropna()
 if not ser.empty:
 last_close_usd[t] = float(ser.iloc[-1])

 # 1) Update positions (stops)
 sells = []
 for t in list(portfolio.get("positions", {}).keys()):
 pos = portfolio["positions"][t]
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
 pos["peak_price_eur"] = float(peak)
 pos["trough_price_eur"] = float(trough)

 mfe_pct = (peak / entry_price - 1.0) * 100.0
 mae_pct = (trough / entry_price - 1.0) * 100.0
 pos["mfe_pct"] = float(mfe_pct)
 pos["mae_pct"] = float(mae_pct)

 pnl_pct = (px_eur / entry_price - 1.0) * 100.0
 pnl_eur = (px_eur - entry_price) * shares
 hold_days = (pd.to_datetime(today) - pd.to_datetime(entry_date)).days

 reason = None

 if pnl_pct <= -HARD_STOP_PCT * 100:
 reason = "HARD_STOP"

 trailing_active = bool(pos.get("trailing_active", False))
 if reason is None and mfe_pct >= MFE_TRIGGER_PCT * 100:
 if not trailing_active:
 pos["trailing_active"] = True
 trailing_active = True

 if reason is None and trailing_active:
 dd_from_peak = (px_eur / peak - 1.0)
 if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
 reason = "TRAILING"

 if reason is None:
 close_series = df[(t, "close")].dropna()
 if len(close_series) >= SMA200_WINDOW:
 sma200 = compute_sma(close_series, SMA200_WINDOW)
 if close_series.iloc[-1] < sma200.iloc[-1]:
 reason = "TREND_BREAK_SMA200"

 if reason:
 value_eur = px_eur * shares
 c = trade_cost(value_eur)
 proceeds = value_eur - c
 net_pnl = pnl_eur - c

 sells.append({
 "ticker": t,
 "price_eur": float(px_eur),
 "shares": float(shares),
 "value_eur": float(value_eur),
 "proceeds": float(proceeds),
 "pnl_eur": float(net_pnl),
 "pnl_pct": float(pnl_pct),
 "mfe_pct": float(mfe_pct),
 "mae_pct": float(mae_pct),
 "hold_days": int(hold_days),
 "reason": reason,
 "rank": int(rank_map.get(t, 999)),
 "score": float(score_map.get(t, 0.0)),
 })

 portfolio["positions"][t] = pos

 for s in sells:
 portfolio["cash"] = float(portfolio.get("cash", 0.0)) + float(s["proceeds"])

 append_trade(trades, {
 "action": "SELL",
 "ticker": s["ticker"],
 "date": today_str,
 "price_eur": s["price_eur"],
 "shares": s["shares"],
 "amount_eur": s["value_eur"],
 "fee_bps": FEE_BPS,
 "slippage_bps": SLIPPAGE_BPS,
 "reason": s["reason"],
 "pnl_eur": s["pnl_eur"],
 "pnl_pct": s["pnl_pct"],
 "mfe_pct": s["mfe_pct"],
 "mae_pct": s["mae_pct"],
 "bars_held": s["hold_days"],
 "rank": s["rank"],
 "score": s["score"],
 })

 portfolio["positions"].pop(s["ticker"], None)

 # 2) SwapEdge (sell only here)
 swaps = check_swap_edge(portfolio, ranked, score_map, today_str)
 for sell_ticker, buy_ticker, reason in swaps:
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
 c = trade_cost(value_eur)
 proceeds = value_eur - c
 pnl_eur = (px_eur - entry_price) * shares - c
 pnl_pct = (px_eur / entry_price - 1.0) * 100.0
 hold_days = (pd.to_datetime(today) - pd.to_datetime(entry_date)).days

 portfolio["cash"] = float(portfolio.get("cash", 0.0)) + float(proceeds)

 append_trade(trades, {
 "action": "SELL",
 "ticker": sell_ticker,
 "date": today_str,
 "price_eur": float(px_eur),
 "shares": float(shares),
 "amount_eur": float(value_eur),
 "fee_bps": FEE_BPS,
 "slippage_bps": SLIPPAGE_BPS,
 "reason": reason,
 "pnl_eur": float(pnl_eur),
 "pnl_pct": float(pnl_pct),
 "mfe_pct": float(pos.get("mfe_pct", 0.0)),
 "mae_pct": float(pos.get("mae_pct", 0.0)),
 "bars_held": int(hold_days),
 "rank": int(rank_map.get(sell_ticker, 999)),
 "score": float(score_map.get(sell_ticker, 0.0)),
 })

 portfolio["positions"].pop(sell_ticker, None)

 # 3) Buys (top MAX_POSITIONS ranks, entry + breadth + corr)
 buys = []
 held = set(portfolio.get("positions", {}).keys())
 slots = MAX_POSITIONS - len(held)
 cash = float(portfolio.get("cash", 0.0))

 if slots > 0 and cash > 50 and breadth_ok:
 for t, sc in ranked:
 if slots <= 0:
 break
 if rank_map.get(t, 999) > MAX_POSITIONS:
 continue
 if t in held:
 continue

 ok, info = entry_map.get(t, (False, {}))
 if not ok:
 continue

 if not corr_ok(list(held), t, corr_matrix, CORR_THRESHOLD):
 continue

 px_usd = last_close_usd.get(t, np.nan)
 if np.isnan(px_usd) or px_usd <= 0:
 continue
 px_eur = usd_to_eur(px_usd, eurusd)

 alloc = allocate_cash_equal(cash, slots)
 alloc = min(alloc, max(0.0, cash - 10.0))
 if alloc < 50:
 continue

 shares = alloc / px_eur
 total_cost = alloc + trade_cost(alloc)
 if total_cost > cash:
 continue

 cash -= total_cost
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
 "rank": int(rank_map.get(t, 999)),
 "score": float(sc),
 }

 append_trade(trades, {
 "action": "BUY",
 "ticker": t,
 "date": today_str,
 "price_eur": float(px_eur),
 "shares": float(shares),
 "amount_eur": float(alloc),
 "fee_bps": FEE_BPS,
 "slippage_bps": SLIPPAGE_BPS,
 "reason": f"CHAMPION_RANK{int(rank_map.get(t, 999))}_BREAKOUT+TREND",
 "rank": int(rank_map.get(t, 999)),
 "score": float(sc),
 })

 buys.append({"ticker": t, "rank": int(rank_map.get(t, 999)), "score": float(sc), "amount": float(alloc)})
 held.add(t)
 slots -= 1

 # 4) Summary + Telegram
 pos_value = 0.0
 pos_lines = []

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
 pos_lines.append(f"- {t} (#{rk}) PnL {pnl_pct:+.1f}% | MFE {mfe:+.1f}% | Trail {trail}")

 cash = float(portfolio.get("cash", 0.0))
 total_val = cash + pos_value

 start_date = pd.to_datetime(portfolio.get("start_date", today_str))
 months = (pd.to_datetime(today).year - start_date.year) * 12 + (pd.to_datetime(today).month - start_date.month)
 invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL_EUR)) + max(0, months) * float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
 pnl_total = total_val - invested
 pnl_total_pct = (total_val / invested - 1.0) * 100.0 if invested > 0 else 0.0

 msg = []
 msg.append(f"APEX CHAMPION — {today_str}")
 msg.append(f"EURUSD {eurusd:.4f}")
 msg.append(f"Cash {cash:.2f}€ | Pos {pos_value:.2f}€ | Total {total_val:.2f}€")
 msg.append(f"Invested~ {invested:.2f}€ | PnL {pnl_total:+.2f}€ ({pnl_total_pct:+.1f}%)")
 msg.append("")
 msg.append("GATES STATUS:")
 msg.append(f"- Breadth: {breadth:.1%} (>={BREADTH_THRESHOLD:.0%}) {'✓' if breadth_ok else '✗'}")
 msg.append(f"- Corr: window={CORR_WINDOW}d, thr={CORR_THRESHOLD}")
 msg.append("")
 msg.append("ACTIONS:")

 if sells:
 for s in sells:
 msg.append(f"SELL {s['ticker']} — {s['reason']} | PnL {s['pnl_pct']:+.1f}% | MFE {s['mfe_pct']:+.1f}% | Hold {s['hold_days']}d")
 if swaps:
 for sell_t, buy_t, r in swaps:
 msg.append(f"SWAP {sell_t} -> {buy_t} ({r})")
 if buys:
 for b in buys:
 msg.append(f"BUY {b['ticker']} (#{b['rank']}) amt {b['amount']:.0f}€ | score {b['score']:.3f}")
 if not sells and not swaps and not buys:
 msg.append("HOLD — no action")

 msg.append("")
 msg.append("POSITIONS:")
 msg.extend(pos_lines if pos_lines else ["- (none)"])

 msg.append("")
 msg.append("TOP 5 MOMENTUM:")
 for i, (t, sc) in enumerate(ranked[:5], 1):
 ok, info = entry_map.get(t, (False, {}))
 status = "✓" if ok else f"✗({info.get('reason', 'unknown')})"
 msg.append(f"{i}. {t} score {float(sc):.3f} {status}")

 message = "\n".join(msg)

 save_portfolio(portfolio)
 save_trades(trades)

 print(message)
 send_telegram(message)

 print("=" * 90)
 print(" Run terminé | portfolio.json + trades_history.json mis à jour")
 print("=" * 90)


if __name__ == "__main__":
 main()

Palmino NicolasTél. : 06 73 20 76 70
