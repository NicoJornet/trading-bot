# File: APEX_CHAMPION_PROD_GITHUB_MIN.py
from __future__ import annotations

"""
APEX v33.x — PROD (GitHub) — Champion-params (MIN DIFF)
======================================================

⚠️ Objectif
- Repartir du script PROD V33 qui fonctionne, et ne changer que le minimum.
- Garder la plomberie: portfolio.json, trades_history.json, Telegram optionnel, parquet>yfinance.
- Passer sur les paramètres "algo champion" demandés.

Changements (diff checklist)
- FULLY_INVESTED: plus d'allocation 50/30/20 => allocation equal-split du cash dispo sur les slots restants.
- Rotation portfolio (prod):
    MAX_POSITIONS=3
    EDGE_MULT=1.00 (non utilisé ici: pas de swap edge dans le script prod; laissé en constante pour traçabilité)
    CONFIRM_DAYS=3 (achat seulement après N confirmations consécutives en top ranks)
    COOLDOWN_DAYS=1 (interdit de racheter un ticker pendant N jours après une vente)
- Signals:
    score momentum cross-sectional = w*r63 + w*r126 + w*r252
    entries filtrées par: SMA200 trend + breakout High60
- Gates:
    breadth >= 0.55 (univers: tickers éligibles avec données) => bloque les achats si gate OFF
    corr gate (win=63, thr=0.65) => empêche d'acheter trop corrélé aux positions existantes
- Exits:
    HARD_STOP -18% (sur close "paper" comme V33)
    Trailing: armé si MFE >= +15%, sortie si -5% depuis peak
    Trend break: close < SMA200 => sortie
    (désactive dead trade / duration / force rotation de V33)

Notes réalistes
- Le script prod travaille sur le dernier Close disponible (comme V33), pour piloter une exécution manuelle/plateforme.
- Si tu veux une exécution strict T+1 open pour le message Telegram, on peut l'ajouter, mais ce serait une modif plus intrusive.

"""

import os
import json
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
# CONFIG (Champion params)
# =============================================================================

PARQUET_PATH = os.environ.get("APEX_OHLCV_PARQUET", "ohlcv_44tickers_2015_2025.parquet")

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

INITIAL_CAPITAL_EUR = 2000.0
MONTHLY_DCA_EUR = 100.0
FEE_EUR = 1.0

# Portfolio/Rotation
MAX_POSITIONS = 3
FULLY_INVESTED = True
EDGE_MULT = 1.00          # (trace only; not used in this prod script)
CONFIRM_DAYS = 3
COOLDOWN_DAYS = 1

# Signals / Momentum
SMA200_WIN = 200
HIGH60_WIN = 60
R63 = 63
R126 = 126
R252 = 252
WEIGHTS = {"r126": 0.5, "r252": 0.3, "r63": 0.2}

# Gates
BREADTH_THR = 0.55
CORR_WIN = 63
CORR_THR = 0.65

# Exits
HARD_STOP_PCT = 0.18
MFE_TRIGGER_PCT = 0.15
TRAIL_FROM_PEAK_PCT = 0.05

# Download window for indicators (calendar days)
LOOKBACK_CAL_DAYS = 520  # enough for SMA200 + R252 with some margin

# Universe (44 tickers) — unchanged (V33)
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
# IO: portfolio + trades
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


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
        # New state for confirm / cooldown (minimal additions)
        p.setdefault("confirm_state", {})   # ticker -> consecutive confirms
        p.setdefault("cooldown_until", {})  # ticker -> YYYY-MM-DD
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
    # normalize to (ticker, field)
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
    # Prefer parquet cache
    if os.path.exists(PARQUET_PATH):
        ohlcv = load_ohlcv_parquet(PARQUET_PATH, UNIVERSE)
        return ohlcv.sort_index()

    # Fallback: yfinance (last LOOKBACK_CAL_DAYS calendar days)
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_CAL_DAYS)
    return load_ohlcv_yfinance(UNIVERSE, start=start, end=end)


# =============================================================================
# FX (USD -> EUR)
# =============================================================================

def get_eur_usd_rate() -> float:
    """Returns EURUSD (USD per 1 EUR). If unavailable, returns 1.0."""
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
# Indicators / Signal helpers
# =============================================================================

def safe_last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def sma(series: pd.Series, win: int) -> float:
    s = series.dropna()
    if len(s) < win:
        return np.nan
    return float(s.rolling(win).mean().iloc[-1])


def breakout_high(series_high: pd.Series, win: int) -> float:
    h = series_high.dropna()
    if len(h) < win:
        return np.nan
    return float(h.rolling(win).max().iloc[-1])


def ret_window(close: pd.Series, win: int) -> float:
    c = close.dropna()
    if len(c) < win + 1:
        return np.nan
    prev = c.iloc[-win-1]
    last = c.iloc[-1]
    if prev <= 0:
        return np.nan
    return float(last / prev - 1.0)


def momentum_score_cs(close: pd.Series) -> float:
    r63 = ret_window(close, R63)
    r126 = ret_window(close, R126)
    r252 = ret_window(close, R252)
    if any(pd.isna(x) for x in [r63, r126, r252]):
        return np.nan
    return (
        WEIGHTS["r63"] * r63 +
        WEIGHTS["r126"] * r126 +
        WEIGHTS["r252"] * r252
    )


def corr_with_held(ohlcv: pd.DataFrame, d: pd.Timestamp, candidate: str, held: List[str]) -> float:
    if not held:
        return 0.0
    # daily returns over CORR_WIN
    def _rets(ticker: str) -> pd.Series:
        c = ohlcv.get((ticker, "close"), pd.Series(dtype=float)).loc[:d].dropna()
        if len(c) < CORR_WIN + 2:
            return pd.Series(dtype=float)
        r = c.pct_change().dropna().tail(CORR_WIN)
        return r

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


# =============================================================================
# Allocation (FULLY_INVESTED)
# =============================================================================

def allocate_cash_equal(cash: float, slots_left: int) -> float:
    if slots_left <= 0:
        return 0.0
    return float(cash) / float(slots_left)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 90)
    print("🚀 APEX v33.x — PROD (Champion-params, MIN DIFF)")
    print("=" * 90)
    print(f"🕒 {_now_str()}")

    portfolio = load_portfolio()
    trades = load_trades()

    # Monthly DCA (unchanged)
    today = datetime.now()
    month_key = f"{today.year}-{today.month:02d}"
    if portfolio.get("last_dca_month") != month_key:
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + MONTHLY_DCA_EUR
        portfolio["last_dca_month"] = month_key
        print(f"💰 DCA: +{MONTHLY_DCA_EUR:.2f}€ (month={month_key})")

    # Load data (unchanged)
    ohlcv = load_data()
    if ohlcv.empty:
        raise RuntimeError("OHLCV vide.")

    # Use latest available date as "signal day"
    d = ohlcv.index.max()
    d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
    print(f"📅 Dernière date OHLCV: {d_str}")

    eurusd = get_eur_usd_rate()
    print(f"💱 EURUSD=X: {eurusd:.4f} (prixUSD -> prixEUR = USD / eurusd)")

    # ---------------------------------------------------------
    # Precompute per-ticker: close/high/low + SMA200 + High60 + score
    # ---------------------------------------------------------
    score_map: Dict[str, float] = {}
    sma200_map: Dict[str, float] = {}
    high60_map: Dict[str, float] = {}
    close_map: Dict[str, float] = {}
    hi_map: Dict[str, float] = {}
    lo_map: Dict[str, float] = {}

    eligible_univ = []
    for t in UNIVERSE:
        c = ohlcv.get((t, "close"), pd.Series(dtype=float)).loc[:d]
        h = ohlcv.get((t, "high"), pd.Series(dtype=float)).loc[:d]
        l = ohlcv.get((t, "low"), pd.Series(dtype=float)).loc[:d]
        if c.dropna().shape[0] < (SMA200_WIN + R252 + 5):
            continue

        last_c = safe_last(c)
        last_h = safe_last(h)
        last_l = safe_last(l)

        sm = sma(c, SMA200_WIN)
        hh = breakout_high(h, HIGH60_WIN)
        sc = momentum_score_cs(c)

        if pd.isna(last_c) or pd.isna(sm) or pd.isna(hh) or pd.isna(sc):
            continue

        close_map[t] = float(last_c)
        hi_map[t] = float(last_h)
        lo_map[t] = float(last_l)
        sma200_map[t] = float(sm)
        high60_map[t] = float(hh)
        score_map[t] = float(sc)
        eligible_univ.append(t)

    # Breadth gate: % close > SMA200
    if eligible_univ:
        above = sum(1 for t in eligible_univ if close_map.get(t, np.nan) > sma200_map.get(t, np.nan))
        breadth = above / max(1, len(eligible_univ))
    else:
        breadth = 0.0

    breadth_ok = (breadth >= BREADTH_THR)
    print(f"📊 Breadth: {breadth:.2%} (thr={BREADTH_THR:.0%}) => {'ON' if breadth_ok else 'OFF'}")

    # Rank (cross-sectional)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    rank_map = {t: (i + 1) for i, (t, _) in enumerate(ranked)}

    # Latest prices (USD->EUR) for mark-to-market / trades
    last_close_eur = {t: usd_to_eur(close_map[t], eurusd) for t in close_map}
    last_high_eur = {t: usd_to_eur(hi_map[t], eurusd) for t in hi_map}
    last_low_eur = {t: usd_to_eur(lo_map[t], eurusd) for t in lo_map}

    # =====================================================================
    # 1) Evaluate positions -> SELL signals (HARD_STOP / TRAIL / TREND_BREAK)
    # =====================================================================
    sells = []
    positions = portfolio.get("positions", {})

    for t, pos in list(positions.items()):
        px_eur = last_close_eur.get(t, np.nan)
        hi_eur = last_high_eur.get(t, np.nan)
        lo_eur = last_low_eur.get(t, np.nan)
        if np.isnan(px_eur):
            continue

        entry_price = float(pos.get("entry_price_eur", pos.get("entry_price", px_eur)))
        shares = float(pos.get("shares", 0.0))
        if shares <= 0 or entry_price <= 0:
            continue

        # Update peak/trough (for mfe/mae)
        peak = float(pos.get("peak_price_eur", entry_price))
        trough = float(pos.get("trough_price_eur", entry_price))
        peak = max(peak, hi_eur if not np.isnan(hi_eur) else px_eur)
        trough = min(trough, lo_eur if not np.isnan(lo_eur) else px_eur)

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

        pnl_eur = (px_eur - entry_price) * shares
        pnl_pct = (px_eur / entry_price - 1.0) * 100.0

        # SELL rules (champion)
        reason = None

        # Trend break (SMA200)
        sm = sma200_map.get(t, np.nan)
        if reason is None and not np.isnan(sm) and px_eur <= usd_to_eur(sm, eurusd):
            reason = "TREND_BREAK_SMA200"

        # Hard stop
        stop_price = entry_price * (1.0 - HARD_STOP_PCT)
        if reason is None and px_eur <= stop_price:
            reason = "HARD_STOP"

        # Trailing
        if reason is None and trailing_active:
            dd_from_peak = (px_eur / peak - 1.0)
            if dd_from_peak <= -TRAIL_FROM_PEAK_PCT:
                reason = "MFE_TRAILING"

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
                "reason": reason,
                "rank": cur_rank,
                "score": cur_score,
                "entry_price_eur": entry_price,
            })

        positions[t] = pos

    # Execute sells (paper)
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
            "rank": int(s["rank"]),
            "score": float(s["score"]),
        })

        # cooldown
        cd_until = (pd.to_datetime(d_str) + pd.Timedelta(days=COOLDOWN_DAYS)).strftime("%Y-%m-%d")
        portfolio.setdefault("cooldown_until", {})[t] = cd_until
        portfolio.setdefault("confirm_state", {}).pop(t, None)

        # remove position
        if t in portfolio.get("positions", {}):
            del portfolio["positions"][t]

    # =====================================================================
    # 2) BUY signals for top ranks 1..MAX_POSITIONS
    # =====================================================================
    buys = []
    held = set(portfolio.get("positions", {}).keys())
    slots_left = MAX_POSITIONS - len(held)

    cash = float(portfolio.get("cash", 0.0))

    # Update confirm_state based on current top ranks eligibility
    confirm_state = portfolio.get("confirm_state", {})
    today_str = d_str

    top_set = set()
    for t, sc in ranked:
        r = int(rank_map.get(t, 999))
        if r > MAX_POSITIONS:
            continue
        if close_map.get(t, np.nan) <= sma200_map.get(t, np.nan):
            continue
        if close_map.get(t, np.nan) < high60_map.get(t, np.nan):
            continue
        top_set.add(t)

    for t in list(confirm_state.keys()):
        if t not in top_set:
            confirm_state[t] = 0
    for t in top_set:
        confirm_state[t] = int(confirm_state.get(t, 0)) + 1

    portfolio["confirm_state"] = confirm_state

    if slots_left > 0 and cash > 50 and breadth_ok:
        for t, sc in ranked:
            if slots_left <= 0:
                break
            r = int(rank_map.get(t, 999))
            if r > MAX_POSITIONS:
                continue
            if t in held:
                continue
            if is_cooldown_active(portfolio, t, today_str):
                continue

            if close_map.get(t, np.nan) <= sma200_map.get(t, np.nan):
                continue
            if close_map.get(t, np.nan) < high60_map.get(t, np.nan):
                continue

            if int(confirm_state.get(t, 0)) < CONFIRM_DAYS:
                continue

            cmax = corr_with_held(ohlcv, d, t, list(held))
            if cmax >= CORR_THR:
                continue

            px_eur = last_close_eur.get(t, np.nan)
            if np.isnan(px_eur) or px_eur <= 0:
                continue

            alloc = allocate_cash_equal(cash, slots_left)
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
                "price_eur": float(px_eur),
                "shares": float(shares),
                "amount_eur": float(alloc),
                "confirm": int(confirm_state.get(t, 0)),
                "corr_max": float(cmax),
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
                "fee_eur": float(FEE_EUR),
                "reason": f"BUY_RANK{r}_CONF{CONFIRM_DAYS}",
                "rank": int(r),
                "score": float(sc),
            })

            held.add(t)
            slots_left -= 1

    # =====================================================================
    # 3) Portfolio summary + Telegram
    # =====================================================================
    pos_value = 0.0
    lines_pos = []
    for t, pos in portfolio.get("positions", {}).items():
        px_eur = last_close_eur.get(t, np.nan)
        if np.isnan(px_eur):
            continue
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
    msg.append(f"APEX PROD — {today_str}")
    msg.append(f"EURUSD=X {eurusd:.4f}")
    msg.append(f"Cash {cash:.2f}€ | Pos {pos_value:.2f}€ | Total {total:.2f}€")
    msg.append(f"Invested~ {invested:.2f}€ | PnL {pnl_total:+.2f}€ ({pnl_total_pct:+.1f}%)")
    msg.append(f"Breadth {breadth:.1%} (thr {BREADTH_THR:.0%}) => {'ON' if breadth_ok else 'OFF'}")
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
    for i, (t, sc) in enumerate(ranked[:5], 1):
        px_eur = last_close_eur.get(t, np.nan)
        rk = rank_map.get(t, 999)
        conf = int(confirm_state.get(t, 0))
        flt = []
        if close_map.get(t, np.nan) > sma200_map.get(t, np.nan):
            flt.append("SMA200")
        if close_map.get(t, np.nan) >= high60_map.get(t, np.nan):
            flt.append("HIGH60")
        msg.append(f"{i}. {t} rank {rk} score {sc:+.3f} px {px_eur:.2f}€ conf {conf} [{','.join(flt) if flt else '-'}]")

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
