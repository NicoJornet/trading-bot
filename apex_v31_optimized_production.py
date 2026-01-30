"""
APEX CHAMPION — PROD (YFINANCE ONLY) — STABLE / INDENT-SAFE
===========================================================

 Script unique à mettre tel quel dans GitHub (remplace 100% le fichier).
 Données: yfinance uniquement (pas de parquet)
 Colonnes yfinance MultiIndex: gérées (field/ticker ou ticker/field)
 Persistance:
 - portfolio.json
 - trades_history.json
 Telegram (optionnel):
 - TELEGRAM_BOT_TOKEN
 - TELEGRAM_CHAT_ID
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

# Costs (bps)
FEE_BPS = 20.0
SLIPPAGE_BPS = 5.0

# Portfolio
MAX_POSITIONS = 3
FULLY_INVESTED = True # equal slots

# Rotation (SwapEdge)
EDGE_MULT = 1.00
CONFIRM_DAYS = 3
COOLDOWN_DAYS = 1

# Stops
HARD_STOP_PCT = 0.18 # -18%
MFE_TRIGGER_PCT = 0.15 # activate trailing after +15%
TRAIL_FROM_PEAK_PCT = 0.05 # -5% from peak once trailing active

# Momentum
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

# yfinance history (calendar days)
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
# UTIL
# =============================================================================

def _now_str() -> str:
 return datetime.now().strftime("%Y-%m-%d %H:%M")


def trade_cost(amount_eur: float) -> float:
 return (FEE_BPS + SLIPPAGE_BPS) * float(amount_eur) / 10000.0


def usd_to_eur(usd: float, eurusd: float) -> float:
 return float(usd) / float(eurusd) if eurusd and eurusd > 0 else float(usd)


# =============================================================================
# IO (portfolio / trades) — NO INDENT TRAPS
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

 # Rotation tracking
 p.setdefault("swap_confirm_tracker", {}) # key "worst->best" -> count
 p.setdefault("last_swap_date", {}) # worst ticker -> date str

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
 r = dict(row)
 r["id"] = len(trades.get("trades", [])) + 1
 r["ts"] = _now_str()
 trades.setdefault("trades", []).append(r)


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
# Data (yfinance only) — MultiIndex safe
# =============================================================================

def _standardize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
 """
 Returns MultiIndex columns: (ticker, field) with field in lowercase:
 open/high/low/close/volume
 Handles yfinance outputs:
 - (field, ticker) OR (ticker, field)
 - sometimes single-ticker output (non MultiIndex)
 """
 out = df.copy()

 if isinstance(out.columns, pd.MultiIndex):
 lvl0 = [str(x).lower() for x in out.columns.get_level_values(0)]
 fields = {"open", "high", "low", "close", "volume", "adj close"}

 # If (field, ticker), swap
 if {"open", "high", "low", "close", "volume"}.issubset(set(lvl0)) or fields.issubset(set(lvl0)):
 out = out.swaplevel(0, 1, axis=1)

 out.columns = pd.MultiIndex.from_tuples(
 [(str(t), str(f).lower()) for (t, f) in out.columns],
 names=["ticker", "field"]
 )

 # Use close instead of adj close
 out = out.rename(columns={"adj close": "close"}, level="field")
 return out

 # Single-index columns (rare)
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
 raise ImportError("yfinance non disponible (ajoute yfinance dans requirements.txt)")

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

 df = _standardize_yf_columns(df)

 # Force MultiIndex if still not MultiIndex (single ticker case)
 if not isinstance(df.columns, pd.MultiIndex):
 needed = ["open", "high", "low", "close", "volume"]
 for c in needed:
 if c not in df.columns:
 raise ValueError(f"Colonnes manquantes: {c} (got {list(df.columns)})")
 t = tickers[0]
 df = df[needed].copy()
 df.columns = pd.MultiIndex.from_product([[t], needed], names=["ticker", "field"])

 df = df.sort_index()
 return df


def load_data(tickers: List[str]) -> pd.DataFrame:
 end = datetime.now()
 start = end - timedelta(days=LOOKBACK_CAL_DAYS)
 return download_yfinance(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def get_eurusd(df: pd.DataFrame) -> float:
 if ("EURUSD=X", "close") in df.columns:
 ser = df[("EURUSD=X", "close")].dropna()
 if not ser.empty:
 return float(ser.iloc[-1])
 return 1.0


# =============================================================================
# Indicators
# =============================================================================

def sma(series: pd.Series, window: int) -> pd.Series:
 return series.rolling(window, min_periods=1).mean()


def rolling_high_prev(high: pd.Series, window: int) -> pd.Series:
 return high.shift(1).rolling(window, min_periods=1).max()


def momentum_score(close: pd.Series) -> float:
 r63 = close.pct_change(R63_WINDOW).iloc[-1]
 r126 = close.pct_change(R126_WINDOW).iloc[-1]
 r252 = close.pct_change(R252_WINDOW).iloc[-1]

 score = 0.0
 if not pd.isna(r126):
 score += SCORE_WEIGHTS[R126_WINDOW] * float(r126)
 if not pd.isna(r252):
 score += SCORE_WEIGHTS[R252_WINDOW] * float(r252)
 if not pd.isna(r63):
 score += SCORE_WEIGHTS[R63_WINDOW] * float(r63)
 return float(score)


def entry_ok(close: pd.Series, high: pd.Series) -> Tuple[bool, str]:
 if len(close) < max(SMA200_WINDOW, HIGH60_WINDOW):
 return False, "insufficient_data"

 s200 = sma(close, SMA200_WINDOW).iloc[-1]
 h60 = rolling_high_prev(high, HIGH60_WINDOW).iloc[-1]
 c = close.iloc[-1]

 if pd.isna(s200) or pd.isna(h60) or pd.isna(c):
 return False, "nan_data"

 if not (c > s200):
 return False, "trend_below_sma200"
 if not (c > h60):
 return False, "no_breakout_high60"

 return True, "ok"


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
 if close.iloc[-1] > sma(close, SMA200_WINDOW).iloc[-1]:
 above += 1
 total += 1
 pct = above / total if total > 0 else 0.0
 return float(pct), int(above), int(total)


def corr_matrix(df: pd.DataFrame, tickers: List[str], window: int) -> pd.DataFrame:
 rets = {}
 for t in tickers:
 if (t, "close") not in df.columns:
 continue
 c = df[(t, "close")].dropna()
 if len(c) < window + 1:
 continue
 r = c.pct_change().iloc[-window:]
 if len(r) >= window // 2:
 rets[t] = r
 if len(rets) < 2:
 return pd.DataFrame()
 return pd.DataFrame(rets).corr()


def corr_gate_ok(held: List[str], cand: str, cm: pd.DataFrame, thr: float) -> bool:
 if cm.empty or cand not in cm.index:
 return True
 for h in held:
 if h not in cm.columns:
 continue
 v = cm.loc[cand, h]
 if not pd.isna(v) and abs(float(v)) > thr:
 return False
 return True


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
# SwapEdge
# =============================================================================

def check_swap_edge(portfolio: dict, ranked: List[Tuple[str, float]], score_map: Dict[str, float], today_str: str) -> List[Tuple[str, str, str]]:
 pos = portfolio.get("positions", {})
 if len(pos) < MAX_POSITIONS:
 return []

 held = list(pos.keys())
 held_scores = [(t, float(score_map.get(t, -999.0))) for t in held]
 held_scores.sort(key=lambda x: x[1])
 worst_t, worst_s = held_scores[0]

 best_t, best_s = None, -999.0
 for t, sc in ranked:
 if t not in pos:
 best_t, best_s = t, float(sc)
 break

 if best_t is None:
 return []

 # Edge check
 if best_s >= worst_s * EDGE_MULT:
 # cooldown check
 last_swap = portfolio.get("last_swap_date", {}).get(worst_t)
 if last_swap:
 days_since = (pd.to_datetime(today_str) - pd.to_datetime(last_swap)).days
 if days_since < COOLDOWN_DAYS:
 return []

 tracker = portfolio.get("swap_confirm_tracker", {})
 key = f"{worst_t}->{best_t}"
 tracker[key] = int(tracker.get(key, 0)) + 1
 portfolio["swap_confirm_tracker"] = tracker

 if tracker[key] >= CONFIRM_DAYS:
 tracker.pop(key, None)
 portfolio.setdefault("last_swap_date", {})
 portfolio["last_swap_date"][worst_t] = today_str
 return [(worst_t, best_t, f"SWAP_EDGE_{CONFIRM_DAYS}d_confirmed")]
 else:
 tracker = portfolio.get("swap_confirm_tracker", {})
 for k in list(tracker.keys()):
 if k.startswith(f"{worst_t}->"):
 tracker.pop(k, None)

 return []


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
 print("=" * 90)
 print("APEX CHAMPION — PROD (YFINANCE ONLY) — STABLE")
 print("=" * 90)

 tickers = UNIVERSE_U54 + ["EURUSD=X"]
 df = load_data(tickers)
 if df is None or df.empty:
 print("Aucune donnée")
 return

 df = df.sort_index()
 today = pd.to_datetime(df.index[-1])
 today_str = today.strftime("%Y-%m-%d")
 eurusd = get_eurusd(df)

 print(f"Date: {today_str} | EURUSD {eurusd:.4f} | Universe {len(UNIVERSE_U54)}")

 portfolio = load_portfolio()
 trades = load_trades()

 apply_monthly_dca(portfolio, today)

 # Compute scores + entry
 score_map: Dict[str, float] = {}
 entry_map: Dict[str, Tuple[bool, str]] = {}

 for t in UNIVERSE_U54:
 if (t, "close") not in df.columns or (t, "high") not in df.columns:
 continue
 close = df[(t, "close")].dropna()
 high = df[(t, "high")].dropna()
 if len(close) < max(R252_WINDOW, SMA200_WINDOW, HIGH60_WINDOW):
 continue
 sc = momentum_score(close)
 score_map[t] = float(sc)
 ok, reason = entry_ok(close, high)
 entry_map[t] = (ok, reason)

 ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
 rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}

 # Breadth
 breadth, above, total = compute_breadth(df, UNIVERSE_U54)
 breadth_ok = breadth >= BREADTH_THRESHOLD
 print(f"BREADTH: {breadth:.2%} ({above}/{total}) | Gate: {'PASS' if breadth_ok else 'FAIL'}")

 # Correlation matrix
 cm = corr_matrix(df, UNIVERSE_U54, CORR_WINDOW)

 # Last closes USD
 last_close_usd: Dict[str, float] = {}
 for t in UNIVERSE_U54:
 if (t, "close") in df.columns:
 s = df[(t, "close")].dropna()
 if not s.empty:
 last_close_usd[t] = float(s.iloc[-1])

 # -------------------------------------------------------------------------
 # 1) Manage exits (hard stop / trailing / trend break)
 # -------------------------------------------------------------------------
 sells: List[dict] = []
 positions = portfolio.get("positions", {})

 for t in list(positions.keys()):
 pos = positions[t]
 entry_price = float(pos.get("entry_price_eur", 0.0))
 shares = float(pos.get("shares", 0.0))
 entry_date = str(pos.get("entry_date", today_str))

 px_usd = last_close_usd.get(t, np.nan)
 if pd.isna(px_usd) or px_usd <= 0:
 continue
 px_eur = usd_to_eur(px_usd, eurusd)

 if entry_price <= 0 or shares <= 0:
 continue

 peak = float(pos.get("peak_price_eur", entry_price))
 trough = float(pos.get("trough_price_eur", entry_price))
 peak = max(peak, px_eur)
 trough = min(trough, px_eur)
 pos["peak_price_eur"] = float(peak)
 pos["trough_price_eur"] = float(trough)

 mfe_pct = (peak / entry_price - 1.0) * 100.0
 pnl_pct = (px_eur / entry_price - 1.0) * 100.0
 pnl_eur = (px_eur - entry_price) * shares

 pos["mfe_pct"] = float(mfe_pct)
 pos["mae_pct"] = float((trough / entry_price - 1.0) * 100.0)

 hold_days = (today - pd.to_datetime(entry_date)).days

 reason = None

 # hard stop
 if pnl_pct <= -HARD_STOP_PCT * 100.0:
 reason = "HARD_STOP"

 # trailing activation
 trailing_active = bool(pos.get("trailing_active", False))
 if reason is None and mfe_pct >= MFE_TRIGGER_PCT * 100.0:
 if not trailing_active:
 pos["trailing_active"] = True
 trailing_active = True

 # trailing exit
 if reason is None and trailing_active:
 dd_from_peak = (px_eur / peak - 1.0)
 if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
 reason = "TRAILING"

 # trend break SMA200
 if reason is None and (t, "close") in df.columns:
 close_series = df[(t, "close")].dropna()
 if len(close_series) >= SMA200_WINDOW:
 if close_series.iloc[-1] < sma(close_series, SMA200_WINDOW).iloc[-1]:
 reason = "TREND_BREAK_SMA200"

 if reason is not None:
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
 "hold_days": int(hold_days),
 "reason": reason,
 "rank": int(rank_map.get(t, 999)),
 "score": float(score_map.get(t, 0.0)),
 })

 positions[t] = pos

 # Execute sells
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
 "bars_held": s["hold_days"],
 "rank": s["rank"],
 "score": s["score"],
 })

 portfolio["positions"].pop(s["ticker"], None)

 # -------------------------------------------------------------------------
 # 2) SwapEdge (sell worst if confirmed)
 # -------------------------------------------------------------------------
 swaps = check_swap_edge(portfolio, ranked, score_map, today_str)
 for sell_t, buy_t, swap_reason in swaps:
 if sell_t not in portfolio.get("positions", {}):
 continue

 pos = portfolio["positions"][sell_t]
 entry_price = float(pos.get("entry_price_eur", 0.0))
 shares = float(pos.get("shares", 0.0))
 entry_date = str(pos.get("entry_date", today_str))

 px_usd = last_close_usd.get(sell_t, np.nan)
 if pd.isna(px_usd) or px_usd <= 0:
 continue
 px_eur = usd_to_eur(px_usd, eurusd)

 value_eur = px_eur * shares
 c = trade_cost(value_eur)
 proceeds = value_eur - c
 pnl_eur = (px_eur - entry_price) * shares - c
 pnl_pct = (px_eur / entry_price - 1.0) * 100.0
 hold_days = (today - pd.to_datetime(entry_date)).days

 portfolio["cash"] = float(portfolio.get("cash", 0.0)) + float(proceeds)

 append_trade(trades, {
 "action": "SELL",
 "ticker": sell_t,
 "date": today_str,
 "price_eur": float(px_eur),
 "shares": float(shares),
 "amount_eur": float(value_eur),
 "fee_bps": FEE_BPS,
 "slippage_bps": SLIPPAGE_BPS,
 "reason": swap_reason,
 "pnl_eur": float(pnl_eur),
 "pnl_pct": float(pnl_pct),
 "mfe_pct": float(pos.get("mfe_pct", 0.0)),
 "bars_held": int(hold_days),
 "rank": int(rank_map.get(sell_t, 999)),
 "score": float(score_map.get(sell_t, 0.0)),
 })

 portfolio["positions"].pop(sell_t, None)

 # -------------------------------------------------------------------------
 # 3) Buys (top ranks, entry + breadth + corr)
 # -------------------------------------------------------------------------
 buys: List[dict] = []
 held = set(portfolio.get("positions", {}).keys())
 slots = int(MAX_POSITIONS - len(held))
 cash = float(portfolio.get("cash", 0.0))

 if slots > 0 and cash > 50.0 and breadth_ok:
 for t, sc in ranked:
 if slots <= 0:
 break

 # Buy only top MAX_POSITIONS ranks
 if int(rank_map.get(t, 999)) > int(MAX_POSITIONS):
 continue
 if t in held:
 continue

 ok, reason = entry_map.get(t, (False, "no_entry_info"))
 if not ok:
 continue

 if not corr_gate_ok(list(held), t, cm, CORR_THRESHOLD):
 continue

 px_usd = last_close_usd.get(t, np.nan)
 if pd.isna(px_usd) or px_usd <= 0:
 continue
 px_eur = usd_to_eur(px_usd, eurusd)

 alloc = (cash / slots) if slots > 0 else 0.0
 alloc = min(alloc, max(0.0, cash - 10.0))
 if alloc < 50.0:
 continue

 total_cost = alloc + trade_cost(alloc)
 if total_cost > cash:
 continue

 shares = alloc / px_eur

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

 buys.append({"ticker": t, "rank": int(rank_map.get(t, 999)), "score": float(sc), "amount": float(alloc), "entry": reason})
 held.add(t)
 slots -= 1

 # -------------------------------------------------------------------------
 # 4) Summary + Telegram
 # -------------------------------------------------------------------------
 pos_value = 0.0
 pos_lines: List[str] = []

 for t, pos in portfolio.get("positions", {}).items():
 px_usd = last_close_usd.get(t, np.nan)
 if pd.isna(px_usd) or px_usd <= 0:
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
 months = (today.year - start_date.year) * 12 + (today.month - start_date.month)
 invested = float(portfolio.get("initial_capital", INITIAL_CAPITAL_EUR)) + max(0, months) * float(portfolio.get("monthly_dca", MONTHLY_DCA_EUR))
 pnl_total = total_val - invested
 pnl_total_pct = (total_val / invested - 1.0) * 100.0 if invested > 0 else 0.0

 msg: List[str] = []
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
 ok, r = entry_map.get(t, (False, "no_info"))
 status = "✓" if ok else f"✗({r})"
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
