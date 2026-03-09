from __future__ import annotations

"""
==========================================================================================
APEX PROD — PACK_FINAL_V6C (A_rank5) — YFINANCE ONLY (U127/130)
==========================================================================================
Base canonique:
- Ranking CSMOM: R63=0.20, R126=0.40, R252=0.40
- Filtre tendance: SMA220 (par actif)
- Corr-guard: window=63, gate=0.92, pick=0.80, scan=10
- Portefeuille: TopK=3, rank_pool=15, keep_rank=5, inv-vol (vol20)
- Rebalance: tous les 10 jours, delta_rebalance=10% (anti-churn)
- Execution: signal Close J, execution Open J+1 (T+1 open)
- Coûts: 1€ par ordre
- Capital: initial 2000€, DCA 100€/mois (1er jour de bourse du mois sur calendrier SPY)
- Données: 100% yfinance (aucun CSV/parquet)
==========================================================================================

✅ PATCH V6C — Allocation lisible (CORRIGÉ v2)
- Affiche HOLDINGS (positions actuelles) pour éviter la confusion avec "Desired"
- Calcule les TARGET € sur la valeur totale du portefeuille (cash + positions) même si cash=0
- Réserve les frais estimés seulement pour les BUY manquants (desired - held)
- ✅ FIX: last_date = dernier jour réellement tradé (anchor SPY si dispo)
- ✅ FIX: valorisation positions = close_ffill (évite Pos=0 à cause de NaN de close brut)
- Ajoute ces infos au message Telegram
"""

import os
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# SETTINGS
# =============================================================================

UNIVERSE: List[str] = [
    "AAPL","ABBV","ABNB","ABT","AEM","AIR.PA","ALB","AMAT","AMD","AMGN","AMZN","ANET","APP","ASML",
    "AVAV","AVGO","AXON","BA","BA.L","BHP","BMY","BP","BWXT","CCJ","CEG","CHTR","CL","CMCSA","CNI",
    "COP","COST","CP","CRWD","CSU.TO","CVX","DHR","DIA","DIS","DLR","DOV","DUK","ENI.MI","EQIX",
    "EQR","EWW","FNV","FSLR","FTNT","GD","GDX","GE","GILD","GLD","GOOGL","GS","HAL","HAG.DE",
    "HD","HO.PA","HON","HUM","IEF","INTC","ISRG","IWM","JNJ","JPM","KLAC","KO","LDO.MI","LLY",
    "LMT","LRCX","LULU","MA","MCD","MELI","META","MC.PA","MMM","MO","MRK","MSTR","MSFT","MU",
    "NEE","NET","NKE","NOC","NVO","NVDA","O","ORCL","PAAS","PANW","PEP","PFE","PG","PLTR","PM",
    "PNC","QCOM","QQQ","RACE","RHM.DE","RIO","RMS.PA","ROK","RTX","SAAB-B.ST","SAF.PA","SBUX",
    "SCHW","SHOP","SLV","SMCI","SO","SPY","SU.PA","T","TGT","TM","TMO","TSLA","TSM","TTE.PA","TXN",
    "UNH","UPS","V","VRTX","VZ","WFC","WM","WMT","XOM","ZS","SNDK","HOOD","BE","WDC","URNM",
]

YF_TICKER_MAP: Dict[str, str] = {
    "CAC40": "^FCHI",
    "DAX": "^GDAXI",
    "EUROSTOXX50": "^STOXX50E",
    "FTSE100": "^FTSE",
}

HISTORY_START = "2014-01-01"
YF_END: Optional[str] = None

# If True: on exec_date, replace missing Open with close_ffill (proxy) so all tickers are tradable.
# WARNING: This relaxes strict 'T+1 open' for tickers with missing open data from yfinance.
FORCE_TRADABLE_OPEN_FALLBACK = True

W_R63, W_R126, W_R252 = 0.20, 0.40, 0.40
# =============================================================================
# CHAMPION ENGINE2 FEATURES (ported from zip, yfinance-compatible)
# =============================================================================
RISK_SET_15 = {
    # Tail-risk set (edit if needed)
    "MARA","RIOT","LEU","MSTR","SMCI","RKLB","APP","NET","COIN","TSLA","NVDA","AMD","PLTR","SMH","TQQQ"
}

# OEG2 conditional (risk_set): veto entry if overextended vs SMA220 AND breaks SMA20
OEG2_ENABLE = 1
OEG2_DIST_TH = 0.85        # distSMA220 > 0.85
OEG2_SMA20_WIN = 20

# TailVeto spike (risk_set): veto entry if ATR% high AND vol spike high
TAILVETO_ENABLE = 1
TAIL_ATR_TH = 0.07         # 7%
TAIL_SPIKE_TH = 1.60       # vol20/vol60

# MIE: Momentum Invalidation Exit (risk_set & non-risk), position-based
MIE_ENABLE = 1
MIE_RS63_TH = -0.03        # exit if R63 < -3%
MIE_MIN_HOLD_DAYS = 9

# ExitSmooth3: smooth exit for names leaving target (3-step)
EXITSMOOTH_ENABLE = 1
EXITSMOOTH_STEPS = 3       # 3 tranches

# Leader Overweight (A022): boost top name weight, renormalize
LEADER_OVW_ENABLE = 1
LEADER_ALPHA = 0.22
SMA_WIN = 220
VOL_WIN = 20

TOPK = 3
RANK_POOL = 15
KEEP_RANK = 5

REB_EVERY_N_DAYS = 10
DELTA_REBAL = 0.10  # 10%

CORR_WIN = 63
CORR_GATE = 0.92
CORR_PICK = 0.75  # champion
CORR_SCAN = 10

FEE_PER_ORDER = 1.0

INITIAL_CASH = 2000.0
MONTHLY_DCA = 100.0

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE = "trades_history.json"

DEBUG_DATA_COVERAGE = True
MIN_BARS_REQUIRED = 260

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OHLCV:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    close_ffill: pd.DataFrame


# =============================================================================
# Utilities
# =============================================================================

def _now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def extract_shares(pos_val) -> float:
    '''
    Robust shares extractor for portfolio formats.

    Supported:
      - float/int : shares directly
      - dict      : {"shares": x} or {"qty": x} or {"quantity": x} or {"units": x}
    '''
    if pos_val is None:
        return 0.0
    if isinstance(pos_val, (int, float, np.integer, np.floating)):
        return float(pos_val)
    if isinstance(pos_val, dict):
        for k in ("shares", "qty", "quantity", "units", "sh", "size"):
            if k in pos_val:
                return safe_float(pos_val.get(k), 0.0)
        return 0.0
    return safe_float(pos_val, 0.0)


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def map_to_yfinance_symbols(tickers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    yf_symbols = []
    rev = {}
    for t in tickers:
        yf_t = YF_TICKER_MAP.get(t, t)
        yf_symbols.append(yf_t)
        rev[yf_t] = t
    return yf_symbols, rev


# =============================================================================
# YFINANCE LOADER (multi-region safe)
# =============================================================================

def load_data_yfinance(tickers: List[str]) -> OHLCV:
    yf_symbols, rev_map = map_to_yfinance_symbols(tickers)

    kwargs = dict(
        tickers=yf_symbols,
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
        interval="1d",
    )
    kwargs["start"] = HISTORY_START
    if YF_END:
        kwargs["end"] = YF_END

    data = yf.download(**kwargs)

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty dataset.")

    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("Unexpected yfinance format: expected MultiIndex columns.")

    lvl0 = list(data.columns.get_level_values(0).unique())
    if "Open" not in lvl0 and "Close" not in lvl0:
        data = data.swaplevel(axis=1).sort_index(axis=1)

    def _get_field(field: str) -> pd.DataFrame:
        if field not in data.columns.get_level_values(0):
            return pd.DataFrame(index=data.index)
        df_f = data[field].copy()
        df_f.columns = [rev_map.get(c, c) for c in df_f.columns]
        return df_f

    o = _get_field("Open")
    h = _get_field("High")
    l = _get_field("Low")
    c = _get_field("Close")
    v = _get_field("Volume")

    cols = sorted(list(set(o.columns) | set(c.columns) | set(h.columns) | set(l.columns) | set(v.columns)))
    o = o.reindex(columns=cols)
    h = h.reindex(columns=cols)
    l = l.reindex(columns=cols)
    c = c.reindex(columns=cols)
    v = v.reindex(columns=cols)

    c_ff = c.ffill()

    if DEBUG_DATA_COVERAGE:
        bars = c_ff.notna().sum().sort_values(ascending=False)
        min_bars = int(bars.min()) if len(bars) else 0
        print(f"🧪 Coverage bars (min={min_bars}, required={MIN_BARS_REQUIRED}) | tickers={len(bars)}")
        bad = bars[bars < MIN_BARS_REQUIRED]
        if len(bad) > 0:
            print("⚠️ Tickers with insufficient history (excluded from ranking):")
            for t, n in bad.items():
                print(f"  - {t}: {int(n)} bars")

    return OHLCV(open=o, high=h, low=l, close=c, volume=v, close_ffill=c_ff)


# =============================================================================
# Signals
# =============================================================================

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_signals(ohlcv: OHLCV) -> dict:
    c = ohlcv.close_ffill
    ret1 = c.pct_change(fill_method=None)

    r63 = c / c.shift(63) - 1.0
    r126 = c / c.shift(126) - 1.0
    r252 = c / c.shift(252) - 1.0

    # Base score (champion backbone)
    score = W_R63 * r63 + W_R126 * r126 + W_R252 * r252

    sma220 = c.rolling(SMA_WIN, min_periods=SMA_WIN).mean()
    sma20 = c.rolling(OEG2_SMA20_WIN, min_periods=OEG2_SMA20_WIN).mean()

    # Volatility (std) and spike
    vol20 = ret1.rolling(VOL_WIN, min_periods=VOL_WIN).std()
    vol60 = ret1.rolling(60, min_periods=60).std()
    vol_spike = (vol20 / (vol60 + 1e-12)).replace([np.inf, -np.inf], np.nan)

    # ATR% proxy (mean absolute return)
    atrp20 = ret1.abs().rolling(20, min_periods=20).mean()

    # Drawdown vs high60 (negative/0)
    high60 = c.rolling(60, min_periods=60).max()
    dd60 = (c / (high60 + 1e-12) - 1.0).clip(upper=0.0)

    # Overextension vs SMA220
    dist_sma220 = (c / (sma220 + 1e-12) - 1.0)

    # RSI red flag
    rsi14 = c.apply(lambda s: _rsi(s, 14))

    enough_history = (c.notna().sum() >= MIN_BARS_REQUIRED)

    return dict(
        score=score,
        sma220=sma220,
        sma20=sma20,
        vol20=vol20,
        vol60=vol60,
        vol_spike=vol_spike,
        atrp20=atrp20,
        dd60=dd60,
        dist_sma220=dist_sma220,
        r63=r63,
        rsi14=rsi14,
        ret1=ret1,
        enough_history=enough_history
    )
def corr_matrix(window_returns: np.ndarray) -> np.ndarray:
    m = window_returns.astype(float)
    m = m - np.nanmean(m, axis=0, keepdims=True)
    s = np.nanstd(m, axis=0, keepdims=True) + 1e-12
    m = m / s
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (m.T @ m) / max(m.shape[0] - 1, 1)
    return np.clip(corr, -1, 1)


def corr_cap_pick(
    ranked: List[str],
    win_slice: np.ndarray,
    topk: int,
    thr: float,
    max_scan: int,
    held: Optional[List[str]] = None
) -> List[str]:
    held = held or []
    corr = corr_matrix(win_slice)
    idx = {t: i for i, t in enumerate(ranked)}

    chosen: List[str] = []
    for h in held:
        if h in idx and h not in chosen:
            chosen.append(h)
            if len(chosen) >= topk:
                return chosen[:topk]

    for t in ranked[:max_scan]:
        if t in chosen:
            continue
        ti = idx.get(t)
        if ti is None:
            continue
        ok = True
        for c in chosen:
            ci = idx.get(c)
            if ci is None:
                continue
            if corr[ti, ci] >= thr:
                ok = False
                break
        if ok:
            chosen.append(t)
        if len(chosen) >= topk:
            break

    if len(chosen) < topk:
        for t in ranked:
            if t not in chosen:
                chosen.append(t)
            if len(chosen) >= topk:
                break

    return chosen[:topk]


def apply_keep_rank(current: List[str], ranked: List[str], topk: int, keep_rank: int) -> List[str]:
    if not ranked:
        return []
    rankpos = {t: i + 1 for i, t in enumerate(ranked)}
    kept = [t for t in current if rankpos.get(t, 10**9) <= keep_rank][:topk]
    out = list(kept)
    for t in ranked:
        if t in out:
            continue
        out.append(t)
        if len(out) >= topk:
            break
    return out[:topk]


def invvol_weights(vol_row: pd.Series, tickers: List[str]) -> Dict[str, float]:
    v = vol_row.reindex(tickers).replace(0, np.nan)
    inv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty:
        return {}
    inv = inv / inv.sum()
    return {t: float(inv.loc[t]) for t in inv.index}


def pretty_weights_and_targets(
    total_equity: float,
    cash: float,
    desired: List[str],
    held: List[str],
    vol_row: pd.Series,
    fee_per_order: float
) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Returns:
      weights: {ticker: w}
      targets_eur: {ticker: target€} based on TOTAL_EQUITY (cash + positions)
      investable_equity: total_equity - fees_reserved (floored at 0)
      fees_reserved: fee_per_order * #missing_buys
    Notes:
      - fee reservation assumes 1 BUY per missing desired name (desired - held).
      - This is an "approx allocation view" (uses CLOSE-date vol, not next open).
    """
    w = invvol_weights(vol_row, desired)
    if not w:
        return {}, {}, 0.0, 0.0

    missing_buys = [t for t in desired if t not in set(held)]
    fees_reserved = fee_per_order * len(missing_buys)

    investable_equity = max(total_equity - fees_reserved, 0.0)
    targets = {t: float(w.get(t, 0.0) * investable_equity) for t in desired}
    return w, targets, investable_equity, fees_reserved


# =============================================================================
# Portfolio IO
# =============================================================================

def load_portfolio() -> dict:
    p = load_json(PORTFOLIO_FILE)
    if not p:
        p = {"cash": INITIAL_CASH, "positions": {}, "entry_date": {}, "exit_stage": {}, "last_rebalance_idx": None, "last_dca_month": None}
    p["cash"] = safe_float(p.get("cash", INITIAL_CASH), INITIAL_CASH)
    p["positions"] = p.get("positions", {}) or {}
    p["entry_date"] = p.get("entry_date", {}) or {}
    p["exit_stage"] = p.get("exit_stage", {}) or {}
    if "last_rebalance_idx" not in p:
        p["last_rebalance_idx"] = None
    if "last_dca_month" not in p:
        p["last_dca_month"] = None
    return p


def save_portfolio(p: dict) -> None:
    save_json(PORTFOLIO_FILE, p)


def append_trades(rows: List[dict]) -> None:
    hist = load_json(TRADES_FILE)
    if not isinstance(hist, list):
        hist = []
    hist.extend(rows)
    save_json(TRADES_FILE, hist)


# =============================================================================
# Engine (prod)
# =============================================================================

def get_month_key(d: pd.Timestamp) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def first_trading_day_each_month(calendar: pd.DatetimeIndex) -> Dict[str, pd.Timestamp]:
    out = {}
    for d in calendar:
        mk = get_month_key(d)
        if mk not in out:
            out[mk] = d
    return out


def main():
    print("=" * 90)
    print("APEX PROD — PACK_FINAL_V6C (A_rank5) — YFINANCE ONLY")
    print("=" * 90)
    print(f"🕒 {_now_str()}")

    port = load_portfolio()

    ohlcv = load_data_yfinance(UNIVERSE)

    # ✅ FIX: last_date = dernier jour réellement tradé (anchor SPY si dispo)
    if "SPY" in ohlcv.close.columns and ohlcv.close["SPY"].notna().sum() > 0:
        cal = ohlcv.close.index[ohlcv.close["SPY"].notna()]
    else:
        cal = ohlcv.close.index
    last_date = cal.max()

    print(f"📅 Dernière date OHLCV (cal): {last_date.date()}")

    month_first = first_trading_day_each_month(cal)
    mk = get_month_key(last_date)
    last_dca_month = port.get("last_dca_month")

    # DCA only on first trading day of month
    if mk != last_dca_month and mk in month_first and month_first[mk] == last_date:
        port["cash"] += MONTHLY_DCA
        port["last_dca_month"] = mk
        print(f"💰 DCA: +{MONTHLY_DCA:.2f} (month={mk})")

    sig = compute_signals(ohlcv)
    score = sig["score"]
    sma220 = sig["sma220"]
    sma20 = sig["sma20"]
    vol20 = sig["vol20"]
    vol_spike = sig["vol_spike"]
    atrp20 = sig["atrp20"]
    dd60 = sig["dd60"]
    dist_sma220 = sig["dist_sma220"]
    r63 = sig["r63"]
    rsi14 = sig["rsi14"]
    ret1 = sig["ret1"]
    enough_history = sig["enough_history"]

    if DEBUG_DATA_COVERAGE:
        try:
            score_non_nan = int(score.loc[last_date].notna().sum())
            above_sma = int((ohlcv.close_ffill.loc[last_date] > sma220.loc[last_date]).sum())
            print(f"🧪 elig debug @ {last_date.date()} | score_non_nan={score_non_nan} | above_SMA={above_sma}")
        except Exception as e:
            print("elig debug failed:", e)

    # IMPORTANT: use calendar index, not raw OHLCV index max
    idx_map = {d: i for i, d in enumerate(cal)}
    last_idx = idx_map.get(last_date, None)
    if last_idx is None:
        raise RuntimeError("last_date not in calendar index map (unexpected).")

    last_reb = port.get("last_rebalance_idx")
    if last_reb is None:
        do_rebalance = True
    else:
        do_rebalance = (last_idx - int(last_reb)) >= REB_EVERY_N_DAYS

    # --- Rebalance calendar diagnostics ---
    if last_reb is None:
        next_rebalance_date = last_date
        trading_days_to_rebalance = 0
        last_rebalance_date = None
    else:
        last_reb = int(last_reb)
        last_rebalance_date = cal[last_reb]
        next_reb_idx = last_reb + REB_EVERY_N_DAYS
        if next_reb_idx < len(cal):
            next_rebalance_date = cal[next_reb_idx]
            trading_days_to_rebalance = max(0, next_reb_idx - last_idx)
        else:
            next_rebalance_date = None
            trading_days_to_rebalance = None

    print(f"LAST_REBALANCE_IDX: {last_reb}")
    print(f"LAST_REBALANCE_DATE: {last_rebalance_date.date() if last_rebalance_date is not None else 'None'}")
    print(f"CURRENT_IDX: {last_idx}")
    print(f"CURRENT_DATE: {last_date.date()}")
    print(f"REB_EVERY_N_DAYS: {REB_EVERY_N_DAYS}")
    print(f"NEXT_REBALANCE_DATE: {next_rebalance_date.date() if next_rebalance_date is not None else 'None'}")
    print(f"TRADING_DAYS_TO_REBALANCE: {trading_days_to_rebalance}")

    positions = port.get("positions", {}) or {}
    entry_date = port.get("entry_date", {}) or {}
    exit_stage = port.get("exit_stage", {}) or {}
    cash = float(port.get("cash", 0.0))

    def pos_value_at_close(d: pd.Timestamp) -> float:
        """
        ✅ FIX: valorise sur close_ffill (évite Pos=0 quand close brut est NaN sur last_date)
        """
        v_ = 0.0
        for t, pos_val in positions.items():
            sh = extract_shares(pos_val)
            if sh <= 0:
                continue
            px = safe_float(ohlcv.close_ffill.loc[d].get(t, np.nan))
            if np.isfinite(px):
                v_ += sh * px
        return v_

    total_pos = pos_value_at_close(last_date)
    total = cash + total_pos

    header = f"APEX PROD — PACK_FINAL_V6C (A_rank5) — {last_date.date()}"
    print(header)
    print(f"Cash {cash:.2f} | Pos {total_pos:.2f} | Total {total:.2f}\n")

    elig = (
        (ohlcv.close_ffill.loc[last_date] > sma220.loc[last_date]) &
        score.loc[last_date].notna() &
        vol20.loc[last_date].notna()
    )
    elig = elig & enough_history.reindex(elig.index, fill_value=False)

    srow = score.loc[last_date].where(elig, np.nan).dropna()
    ranked = list(srow.sort_values(ascending=False).head(RANK_POOL).index)
    # Champion guards (risk_set only): OEG2 conditional + TailVeto spike
    if ranked:
        close_row = ohlcv.close_ffill.loc[last_date]
        filt: List[str] = []
        for t in ranked:
            if t in RISK_SET_15:
                if OEG2_ENABLE and (safe_float(dist_sma220.loc[last_date].get(t, np.nan)) > OEG2_DIST_TH) and (
                    safe_float(close_row.get(t, np.nan)) < safe_float(sma20.loc[last_date].get(t, np.nan))
                ):
                    continue  # OEG2 veto
                if TAILVETO_ENABLE and (safe_float(atrp20.loc[last_date].get(t, np.nan)) > TAIL_ATR_TH) and (
                    safe_float(vol_spike.loc[last_date].get(t, np.nan)) > TAIL_SPIKE_TH
                ):
                    continue  # TailVeto spike
            filt.append(t)
        ranked = filt

    print("TOP 15 MOMENTUM:")
    if len(ranked) == 0:
        print("(none)")
    else:
        for i, t in enumerate(ranked[:15], 1):
            print(f"{i}. {t} score {float(srow.loc[t]):.4f}")
    print("")

    desired_ranked = ranked[:TOPK]

    current = list(positions.keys())

    corr_gate_hit = 0
    if len(ranked) >= 2:
        # Use calendar location in the full ret1 index
        loc_full = ohlcv.close.index.get_loc(last_date)
        w0 = max(0, loc_full - CORR_WIN + 1)
        win_slice = ret1.iloc[w0:loc_full + 1][ranked].to_numpy()
        if win_slice.shape[0] >= int(0.8 * CORR_WIN):
            corr = corr_matrix(win_slice)
            max_corr = float(np.nanmax(np.where(np.eye(corr.shape[0]), np.nan, corr)))
            if np.isfinite(max_corr) and max_corr > CORR_GATE:
                corr_gate_hit = 1
                desired_ranked = corr_cap_pick(
                    ranked=ranked,
                    win_slice=win_slice,
                    topk=TOPK,
                    thr=CORR_PICK,
                    max_scan=CORR_SCAN,
                    held=current if current else None
                )

    current = list(positions.keys())
    desired = apply_keep_rank(current=current, ranked=desired_ranked, topk=TOPK, keep_rank=KEEP_RANK)

    held = sorted(list(positions.keys()))
    print(f"HELD: {held if held else '(none)'}")
    print(f"Desired: {desired if desired else '(none)'}")
    print(f"CorrGate: {corr_gate_hit}\n")

    # show weights and target € based on TOTAL equity (cash + positions),
    # and reserve fees only for missing BUYs (desired - held).
    total_equity_close = cash + total_pos
    w_dbg, targets_dbg, investable_eq_dbg, fees_reserved_dbg = pretty_weights_and_targets(
        total_equity=total_equity_close,
        cash=cash,
        desired=desired,
        held=held,
        vol_row=vol20.loc[last_date],
        fee_per_order=FEE_PER_ORDER
    )

    if desired and w_dbg:
        print("WEIGHTS (inv-vol20):", {t: round(w_dbg.get(t, 0.0), 4) for t in desired})
        print("TARGET € (based on TOTAL, fees reserved for missing BUYs):",
              {t: round(targets_dbg.get(t, 0.0), 2) for t in desired})
        print(f"TOTAL_EQUITY(close): {total_equity_close:.2f}")
        print(f"FEES_RESERVED (missing buys): {fees_reserved_dbg:.2f}")
        print(f"INVESTABLE_EQUITY: {investable_eq_dbg:.2f}\n")
    else:
        print("WEIGHTS/TARGETS: (unavailable)\n")

    if not do_rebalance:
        print("ORDERS: none (not a rebalance day)")
        save_portfolio(port)

        msg_lines = [
            header,
            f"Cash {cash:.2f} | Pos {total_pos:.2f} | Total {total:.2f}",
            "",
            "TOP 15 MOMENTUM:",
        ]
        if len(ranked) == 0:
            msg_lines.append("(none)")
        else:
            for i, t in enumerate(ranked[:15], 1):
                msg_lines.append(f"{i}. {t} score {float(srow.loc[t]):.4f}")

        msg_lines.append("")
        msg_lines.append(f"HELD: {held if held else '(none)'}")
        msg_lines.append(f"Desired: {desired if desired else '(none)'}")
        msg_lines.append(f"CorrGate: {corr_gate_hit}")

        if desired and w_dbg:
            msg_lines.append("")
            msg_lines.append("WEIGHTS (inv-vol20): " + str({t: round(w_dbg.get(t, 0.0), 4) for t in desired}))
            msg_lines.append("TARGET € (approx): " + str({t: round(targets_dbg.get(t, 0.0), 2) for t in desired}))
            msg_lines.append(f"TOTAL_EQUITY(close): {total_equity_close:.2f}")
            msg_lines.append(f"FEES_RESERVED: {fees_reserved_dbg:.2f}")
            msg_lines.append(f"INVESTABLE_EQUITY: {investable_eq_dbg:.2f}")

        msg_lines.append("")
        msg_lines.append(f"LAST_REBALANCE_DATE: {last_rebalance_date.date() if last_rebalance_date is not None else 'None'}")
        msg_lines.append(f"CURRENT_DATE: {last_date.date()}")
        msg_lines.append(f"NEXT_REBALANCE_DATE: {next_rebalance_date.date() if next_rebalance_date is not None else 'None'}")
        msg_lines.append(f"TRADING_DAYS_TO_REBALANCE: {trading_days_to_rebalance}")

        msg_lines.append("")
        msg_lines.append("ORDERS: none (not a rebalance day)")
        send_telegram("\n".join(msg_lines))
        return

    # Rebalance day execution @ next open (T+1 open)
    # Need next trading day in the FULL index
    loc_full = ohlcv.open.index.get_loc(last_date)
    if loc_full + 1 >= len(ohlcv.open.index):
        print("ORDERS: none (no next open available)")
        save_portfolio(port)
        return

    exec_date = ohlcv.open.index[loc_full + 1]
    px_open = ohlcv.open.loc[exec_date]

    def _px_for_valuation(t: str) -> float:
        """Use exec_date open if available, else fallback to last_date close_ffill for valuation only."""
        op = safe_float(px_open.get(t, np.nan))
        if np.isfinite(op):
            return float(op)
        cl = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
        return float(cl) if np.isfinite(cl) else float("nan")

    needed = set(current) | set(desired)

    # yfinance can have missing "Open" on exec_date for some tickers (holes/late refresh/market specifics).
    # If FORCE_TRADABLE_OPEN_FALLBACK is True, we force tradability by substituting missing Open with a proxy:
    #  - primary: close_ffill on exec_date (same-day close proxy)
    #  - fallback: close_ffill on last_date (previous close proxy)
    # This keeps the engine operational on day J, but relaxes strict "T+1 open" for those tickers only.
    fallback_used = []
    if FORCE_TRADABLE_OPEN_FALLBACK:
        px_exec = px_open.copy()
        # First proxy: close_ffill at exec_date
        if exec_date in ohlcv.close_ffill.index:
            px_exec = px_exec.fillna(ohlcv.close_ffill.loc[exec_date])
        # Second proxy: previous close_ffill (last_date)
        px_exec = px_exec.fillna(ohlcv.close_ffill.loc[last_date])

        for t in needed:
            if not np.isfinite(safe_float(px_open.get(t, np.nan))) and np.isfinite(safe_float(px_exec.get(t, np.nan))):
                fallback_used.append(t)

        if fallback_used:
            print(f"⚠️ Missing next open for {fallback_used} on {exec_date.date()} — using CLOSE proxy for execution price (forced tradable).")

        # overwrite execution prices used downstream
        px_open = px_exec

    # Hard check: all needed tickers must have an execution price
    missing_price = [t for t in needed if not np.isfinite(safe_float(px_open.get(t, np.nan)))]
    if missing_price:
        print(f"ORDERS: none (missing execution price for {missing_price} on {exec_date.date()})")
        save_portfolio(port)
        return

    port_val_open = cash
    for t, pos_val in positions.items():
        sh = extract_shares(pos_val)
        if sh <= 0:
            continue
        pxv = _px_for_valuation(t)
        if np.isfinite(pxv):
            port_val_open += sh * float(pxv)

    w = invvol_weights(vol20.loc[last_date], desired)
    targets_val = {t: w[t] * port_val_open for t in w if t in needed}

    orders: List[dict] = []

    # MIE (Momentum Invalidation Exit): force exit if R63 below threshold and min-hold satisfied
    if MIE_ENABLE and positions:
        for t in list(positions.keys()):
            rs = safe_float(r63.loc[last_date].get(t, np.nan))
            if not np.isfinite(rs) or rs >= MIE_RS63_TH:
                continue

            ent = entry_date.get(t)
            hold_ok = True
            if ent:
                try:
                    ent_dt = pd.Timestamp(ent)
                    ent_idx = idx_map.get(ent_dt, None)
                    if ent_idx is not None:
                        hold_days = max(0, last_idx - ent_idx)
                        hold_ok = (hold_days >= MIE_MIN_HOLD_DAYS)
                except Exception:
                    pass
            if not hold_ok:
                continue

            if t not in needed:
                continue  # cannot trade today (missing next open)

            sh = extract_shares(positions.pop(t, None))
            if sh <= 0:
                continue
            exit_stage.pop(t, None)
            entry_date.pop(t, None)
            cash += sh * float(px_open[t]) - FEE_PER_ORDER
            orders.append({
                "Date": str(exec_date.date()),
                "Side": "SELL",
                "Ticker": t,
                "Shares": sh,
                "Price": float(px_open[t]),
                "Fee": FEE_PER_ORDER,
                "Reason": "MOM_INV_RS63",
            })

    # Sell names not in target (ExitSmooth3)
    for t in list(positions.keys()):
        if t in targets_val:
            exit_stage.pop(t, None)
            continue

        cur_sh = extract_shares(positions.get(t, 0.0))
        if cur_sh <= 0:
            positions.pop(t, None)
            exit_stage.pop(t, None)
            entry_date.pop(t, None)
            continue

        if EXITSMOOTH_ENABLE:
            st = int(exit_stage.get(t, 0)) + 1
            st = min(st, EXITSMOOTH_STEPS)
            exit_stage[t] = st
            frac = 1.0 / EXITSMOOTH_STEPS if st < EXITSMOOTH_STEPS else 1.0
            sell_sh = cur_sh * frac
            reason = f"EXITSMOOTH_{st}"
        else:
            sell_sh = cur_sh
            reason = "EXIT_NOT_TARGET"

        if t not in needed:
            continue  # cannot trade today (missing next open)

        cash += sell_sh * float(px_open[t]) - FEE_PER_ORDER
        new_sh = cur_sh - sell_sh
        if new_sh <= 1e-10:
            positions.pop(t, None)
            exit_stage.pop(t, None)
            entry_date.pop(t, None)
        else:
            positions[t] = new_sh

        orders.append({
            "Date": str(exec_date.date()),
            "Side": "SELL",
            "Ticker": t,
            "Shares": sell_sh,
            "Price": float(px_open[t]),
            "Fee": FEE_PER_ORDER,
            "Reason": reason,
        })

    # Recompute portfolio value at open after exits (used for delta-rebalance threshold)
    port_val_open = cash
    for t_pos, pos_val in positions.items():
        sh = extract_shares(pos_val)
        if sh <= 0:
            continue
        pxv = _px_for_valuation(t_pos)
        if np.isfinite(pxv):
            port_val_open += sh * float(pxv)

    # Delta rebalance
    for t, tgt_val in targets_val.items():
        if t not in needed:
            continue  # cannot trade today (missing next open)
        cur_sh = extract_shares(positions.get(t, 0.0))
        price = float(px_open[t])
        cur_val = cur_sh * price
        diff = tgt_val - cur_val

        if abs(diff) < DELTA_REBAL * port_val_open:
            continue

        if diff < 0 and cur_sh > 0:
            sell_sh = min((-diff) / price, cur_sh)
            cash += sell_sh * price - FEE_PER_ORDER
            new_sh = cur_sh - sell_sh
            if new_sh <= 1e-10:
                positions.pop(t, None)
            else:
                positions[t] = new_sh
            orders.append({
                "Date": str(exec_date.date()),
                "Side": "SELL",
                "Ticker": t,
                "Shares": sell_sh,
                "Price": price,
                "Fee": FEE_PER_ORDER,
                "Reason": "DELTA_REBAL_SELL",
            })

        elif diff > 0:
            max_buy_val = max(cash - FEE_PER_ORDER, 0.0)
            buy_val = min(diff, max_buy_val)
            if buy_val > 1e-8:
                buy_sh = buy_val / price
                cash -= buy_val + FEE_PER_ORDER
                new_total = cur_sh + buy_sh
                positions[t] = new_total
                if cur_sh <= 1e-12 and new_total > 1e-12:
                    entry_date[t] = str(exec_date.date())
                orders.append({
                    "Date": str(exec_date.date()),
                    "Side": "BUY",
                    "Ticker": t,
                    "Shares": buy_sh,
                    "Price": price,
                    "Fee": FEE_PER_ORDER,
                    "Reason": "DELTA_REBAL_BUY",
                })



    port["cash"] = cash
    port["positions"] = positions
    port["entry_date"] = entry_date
    port["exit_stage"] = exit_stage
    port["last_rebalance_idx"] = int(last_idx)
    save_portfolio(port)

    if orders:
        append_trades(orders)

    if not orders:
        print("ORDERS: none")
    else:
        print("ORDERS:")
        for o in orders:
            print(f" - {o['Side']} {o['Ticker']} sh={o['Shares']:.6f} @ {o['Price']:.2f} fee={o['Fee']:.2f} ({o['Reason']})")

    # Telegram message on rebalance days
    msg_lines = [
        header,
        f"Cash {cash:.2f} | Pos {pos_value_at_close(last_date):.2f} | Total {cash + pos_value_at_close(last_date):.2f}",
        "",
        "TOP 5 MOMENTUM:",
    ]
    if len(ranked) == 0:
        msg_lines.append("(none)")
    else:
        for i, t in enumerate(ranked[:5], 1):
            msg_lines.append(f"{i}. {t} score {float(srow.loc[t]):.4f}")

    held_after = sorted(list(positions.keys()))
    msg_lines.append("")
    msg_lines.append(f"HELD: {held_after if held_after else '(none)'}")
    msg_lines.append(f"Desired: {desired if desired else '(none)'}")
    msg_lines.append(f"CorrGate: {corr_gate_hit}")

    if desired and w:
        msg_lines.append("")
        msg_lines.append("WEIGHTS (inv-vol20): " + str({t: round(w.get(t, 0.0), 4) for t in desired}))
        msg_lines.append("TARGET € (open-based): " + str({t: round(targets_val.get(t, 0.0), 2) for t in desired}))

    msg_lines.append("")
    msg_lines.append(f"LAST_REBALANCE_DATE: {last_rebalance_date.date() if last_rebalance_date is not None else 'None'}")
    msg_lines.append(f"CURRENT_DATE: {last_date.date()}")
    msg_lines.append(f"NEXT_REBALANCE_DATE: {next_rebalance_date.date() if next_rebalance_date is not None else 'None'}")
    msg_lines.append(f"TRADING_DAYS_TO_REBALANCE: {trading_days_to_rebalance}")

    msg_lines.append("")
    if not orders:
        msg_lines.append("ORDERS: none")
    else:
        msg_lines.append("ORDERS:")
        for o in orders:
            msg_lines.append(f"- {o['Side']} {o['Ticker']} @ {o['Price']:.2f} (fee {o['Fee']:.2f})")

    send_telegram("\n".join(msg_lines))


if __name__ == "__main__":
    main()
