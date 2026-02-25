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
    "SCHW","SHOP","SLV","SMCI","SO","SPY","SU.PA","T","TGT","TM","TMO","TSLA","TSM","TXN",
    "UNH","UPS","V","VRTX","VZ","WFC","WM","WMT","XOM","ZS","SNDK","HOOD","BE","WDC",
    "KTOS",
    "LEU",
    "DNN",
    "UEC"
]

YF_TICKER_MAP: Dict[str, str] = {
    "CAC40": "^FCHI",
    "DAX": "^GDAXI",
    "EUROSTOXX50": "^STOXX50E",
    "FTSE100": "^FTSE",
}

HISTORY_START = "2014-01-01"
YF_END: Optional[str] = None

W_R63, W_R126, W_R252 = 0.20, 0.40, 0.40
SMA_WIN = 220
VOL_WIN = 20

TOPK = 3
RANK_POOL = 15
KEEP_RANK = 5

REB_EVERY_N_DAYS = 10
DELTA_REBAL = 0.10  # 10%

CORR_WIN = 63
CORR_GATE = 0.92
CORR_PICK = 0.80
CORR_SCAN = 10

FEE_PER_ORDER = 1.0
# =============================================================================
# CHAMPION V7.1 (EXITSMOOTH3 + TAILVETO + OEG2_COND + PP + MIE + LeaderOverweight A022)
# =============================================================================

# TailVeto Spike (risk_set only) — veto ENTRY if tail risk is detected
TAILVETO_ENABLE = True
TAILVETO_ATRP20_TH = 0.06
TAILVETO_VOLSPIKE_TH = 1.10  # vol20 / vol60

# OEG2_COND (risk_set only, entry only)
OEG2_ENABLE = True
OEG2_DIST_TH = 0.85          # distSMA220 = Close/SMA220 - 1
OEG2_SMA_SHORT = 20          # SMA20

# Risk set (15 tickers) — frozen from RUN_FULL V7.1 config
RISK_SET = {
    "RKLB","MSTR","KTOS","LEU","SMCI","APP","MARA","RIOT","DNN","VRT","RHM.DE","PLTR","COP","UEC","TSLA"
}

# EXITSMOOTH3 — defer SELL of losers not-in-target if still trending (Close>SMA220)
EXIT_SMOOTH_ENABLE = True
EXIT_SMOOTH_MAX_DEFERS = 2
EXIT_SMOOTH_REQUIRE_TREND = True

# Profit Protection (PP) — armed after MFE trigger, then trailing DD from peak
PP_ENABLE = True
PP_MFE_TRIGGER = 0.38
PP_TRAIL_DD = 0.10
PP_MIN_DAYS_AFTER_ARM = 5

# MIE (Momentum Invalidation Exit) — checked on close, executed next open (PP-safe)
MIE_ENABLE = True
MIE_RS63_TH = 0.03
MIE_MIN_HOLD = 9

# Leader Overweight (A022) — blend inv-vol weights with a leader one-hot
LEADER_OVW_ENABLE = True
LEADER_ALPHA = 0.22
LEADER_W_CAP = 0.50


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

def compute_signals(ohlcv: OHLCV) -> dict:
    c = ohlcv.close_ffill
    o = ohlcv.open
    h = ohlcv.high
    l = ohlcv.low

    ret1 = c.pct_change(fill_method=None)

    r63 = c / c.shift(63) - 1.0
    r126 = c / c.shift(126) - 1.0
    r252 = c / c.shift(252) - 1.0
    r5 = c / c.shift(5) - 1.0

    # Cross-sectional momentum score
    score = W_R63 * r63 + W_R126 * r126 + W_R252 * r252

    sma220 = c.rolling(SMA_WIN, min_periods=SMA_WIN).mean()
    sma20 = c.rolling(OEG2_SMA_SHORT, min_periods=OEG2_SMA_SHORT).mean()

    vol20 = ret1.rolling(VOL_WIN, min_periods=VOL_WIN).std()
    vol60 = ret1.rolling(60, min_periods=60).std()

    # ATR%20 (true range based)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=0).groupby(level=0).max()
    atr20 = tr.rolling(20, min_periods=20).mean()
    atrp20 = atr20 / c

    # RS63 vs SPY (fallback neutral if SPY missing)
    if "SPY" in c.columns and c["SPY"].notna().any():
        spy_r63 = c["SPY"] / c["SPY"].shift(63) - 1.0
        rs63 = (1.0 + r63).div(1.0 + spy_r63, axis=0) - 1.0
    else:
        rs63 = r63 * 0.0

    dist_sma220 = c / sma220 - 1.0

    enough_history = (c.notna().sum() >= MIN_BARS_REQUIRED)

    return dict(
        score=score,
        sma220=sma220,
        sma20=sma20,
        dist_sma220=dist_sma220,
        vol20=vol20,
        vol60=vol60,
        atrp20=atrp20,
        rs63=rs63,
        r5=r5,
        ret1=ret1,
        enough_history=enough_history,
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
        p = {"cash": INITIAL_CASH, "positions": {}, "last_rebalance_idx": None, "last_dca_month": None}
    p["cash"] = safe_float(p.get("cash", INITIAL_CASH), INITIAL_CASH)
    p["positions"] = p.get("positions", {}) or {}
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
    dist_sma220 = sig["dist_sma220"]
    vol20 = sig["vol20"]
    vol60 = sig["vol60"]
    atrp20 = sig["atrp20"]
    rs63 = sig["rs63"]
    r5 = sig["r5"]
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

    do_rebalance = (last_idx % REB_EVERY_N_DAYS == 0)

    positions = port.get("positions", {}) or {}
    cash = float(port.get("cash", 0.0))
    pp_state = port.get("pp_state", {}) or {}
    exit_defer = port.get("exit_defer", {}) or {}

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

    # Champion V7.1 entry vetoes (risk_set only)
    if TAILVETO_ENABLE:
        tail = (atrp20.loc[last_date] >= TAILVETO_ATRP20_TH) & ((vol20.loc[last_date] / (vol60.loc[last_date] + 1e-12)) >= TAILVETO_VOLSPIKE_TH)
        tail = tail & pd.Series([t in RISK_SET for t in elig.index], index=elig.index)
        elig = elig & (~tail.fillna(False))

    if OEG2_ENABLE:
        oeg2 = (dist_sma220.loc[last_date] > OEG2_DIST_TH) & (ohlcv.close_ffill.loc[last_date] < sma20.loc[last_date])
        oeg2 = oeg2 & pd.Series([t in RISK_SET for t in elig.index], index=elig.index)
        elig = elig & (~oeg2.fillna(False))

    srow = score.loc[last_date].where(elig, np.nan).dropna()
    ranked = list(srow.sort_values(ascending=False).head(RANK_POOL).index)

    print("TOP 5 MOMENTUM:")
    if len(ranked) == 0:
        print("(none)")
    else:
        for i, t in enumerate(ranked[:5], 1):
            print(f"{i}. {t} score {float(srow.loc[t]):.4f}")
    print("")

    desired_ranked = ranked[:TOPK]

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
                    held=None
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




    # Next open date for any execution (PP/MIE exits and/or rebalance) — T+1 open
    loc_full = ohlcv.open.index.get_loc(last_date)
    exec_date = None
    if loc_full + 1 < len(ohlcv.open.index):
        exec_date = ohlcv.open.index[loc_full + 1]
        px_open_exec = ohlcv.open.loc[exec_date]
    else:
        px_open_exec = None

    orders: List[dict] = []

    # -------------------------------------------------------------
    # Daily risk exits (checked on close last_date, executed next open)
    # -------------------------------------------------------------
    if exec_date is not None and px_open_exec is not None:
        # Ensure state exists for held tickers
        for t in list(positions.keys()):
            if t not in pp_state:
                entry_px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
                pp_state[t] = {"entry_px": entry_px, "entry_idx": int(last_idx), "peak": entry_px, "armed": False, "arm_idx": None}

        pp_exit = []
        if PP_ENABLE:
            for t in list(positions.keys()):
                st = pp_state.get(t, None)
                if st is None:
                    continue
                px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
                if not np.isfinite(px):
                    continue
                st["peak"] = float(max(float(st.get("peak", px)), px))
                entry_px = safe_float(st.get("entry_px", px))
                mfe = (float(st["peak"]) / entry_px - 1.0) if entry_px > 0 else 0.0
                if (not bool(st.get("armed", False))) and mfe >= PP_MFE_TRIGGER:
                    st["armed"] = True
                    st["arm_idx"] = int(last_idx)
                if bool(st.get("armed", False)) and (int(last_idx) - int(st.get("arm_idx", last_idx))) >= PP_MIN_DAYS_AFTER_ARM:
                    dd = 1.0 - (px / float(st["peak"])) if float(st["peak"]) > 0 else 0.0
                    if dd >= PP_TRAIL_DD:
                        pp_exit.append(t)

        mie_exit = []
        if MIE_ENABLE and ("SPY" in ohlcv.close_ffill.columns):
            for t in list(positions.keys()):
                st = pp_state.get(t, None)
                if st is None:
                    continue
                if bool(st.get("armed", False)):
                    continue  # PP-safe
                entry_px = safe_float(st.get("entry_px", np.nan))
                entry_idx = int(st.get("entry_idx", last_idx))
                age = int(last_idx) - entry_idx
                if age < MIE_MIN_HOLD:
                    continue
                px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
                if not (np.isfinite(px) and np.isfinite(entry_px) and entry_px > 0):
                    continue
                live_pnl = px / entry_px - 1.0
                if live_pnl > 0:
                    continue
                if (safe_float(rs63.loc[last_date].get(t, 0.0)) <= -MIE_RS63_TH) and (px < safe_float(sma220.loc[last_date].get(t, np.inf))) and (safe_float(r5.loc[last_date].get(t, 0.0)) < 0.0):
                    mie_exit.append(t)

        forced_exit = {t: "PP" for t in pp_exit}
        forced_exit.update({t: "MIE" for t in mie_exit})

        for t, reason in list(forced_exit.items()):
            if t not in positions:
                continue
            price = safe_float(px_open_exec.get(t, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            sh = extract_shares(positions.get(t))
            if sh <= 0:
                continue
            cash += sh * price - FEE_PER_ORDER
            positions.pop(t, None)
            pp_state.pop(t, None)
            exit_defer.pop(t, None)
            orders.append({
                "Date": str(exec_date.date()),
                "Side": "SELL",
                "Ticker": t,
                "Shares": sh,
                "Price": price,
                "Fee": FEE_PER_ORDER,
                "Reason": reason,
            })

    # Persist state after possible exits
    port["pp_state"] = pp_state
    port["exit_defer"] = exit_defer
    if not do_rebalance:
        print("ORDERS:")
        if len(orders)==0:
            print("none (not a rebalance day)")
        else:
            for o_ in orders:
                print(o_)
        if orders:
            append_trades(orders)
        save_portfolio(port)

        msg_lines = [
            header,
            f"Cash {cash:.2f} | Pos {total_pos:.2f} | Total {total:.2f}",
            "",
            "TOP 5 MOMENTUM:",
        ]
        if len(ranked) == 0:
            msg_lines.append("(none)")
        else:
            for i, t in enumerate(ranked[:5], 1):
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
        if len(orders)==0:
            msg_lines.append("ORDERS: none (not a rebalance day)")
        else:
            msg_lines.append("ORDERS:")
            for o_ in orders:
                msg_lines.append(f"{o_['Side']} {o_['Ticker']} sh={o_['Shares']:.4f} @ {o_['Price']:.4f} ({o_['Reason']})")
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

    needed = set(current) | set(desired)
    for t in needed:
        if not np.isfinite(safe_float(px_open.get(t, np.nan))):
            print(f"ORDERS: none (missing next open for {t} on {exec_date.date()})")
            save_portfolio(port)
            return

    port_val_open = cash
    for t, pos_val in positions.items():
        sh = extract_shares(pos_val)
        if sh <= 0:
            continue
        port_val_open += sh * float(px_open[t])

    w = invvol_weights(vol20.loc[last_date], desired)

    # Leader Overweight (A022)
    if LEADER_OVW_ENABLE and desired:
        leader = desired[0]
        if leader in w:
            for _t in list(w.keys()):
                w[_t] = (1.0 - LEADER_ALPHA) * w[_t] + (LEADER_ALPHA if _t == leader else 0.0)
            # cap leader
            if w.get(leader, 0.0) > LEADER_W_CAP:
                excess = w[leader] - LEADER_W_CAP
                w[leader] = LEADER_W_CAP
                others = [k for k in w.keys() if k != leader]
                s_oth = sum(w[k] for k in others)
                if s_oth > 0:
                    for k in others:
                        w[k] += excess * (w[k] / s_oth)
            # renorm
            ssum = float(sum(w.values()))
            if ssum > 0:
                for k in w:
                    w[k] /= ssum

    targets_val = {t: w[t] * port_val_open for t in w}

    # (orders list may already contain PP/MIE exits earlier in the day)

    

    # Sell names not in target
    for t in list(positions.keys()):
        if t not in targets_val:
            sh = extract_shares(positions.get(t))
            if sh <= 0:
                continue

            # EXITSMOOTH3: defer selling losers that still pass trend (Close>SMA220), max 2 defers
            if EXIT_SMOOTH_ENABLE:
                entry_px = safe_float(pp_state.get(t, {}).get("entry_px", np.nan))
                cl_px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
                upnl = (cl_px / entry_px - 1.0) if (np.isfinite(entry_px) and entry_px > 0 and np.isfinite(cl_px)) else 0.0
                ok = (upnl < 0.0)
                if EXIT_SMOOTH_REQUIRE_TREND:
                    ok = ok and (cl_px > safe_float(sma220.loc[last_date].get(t, -np.inf)))
                if ok and int(exit_defer.get(t, 0)) < EXIT_SMOOTH_MAX_DEFERS:
                    exit_defer[t] = int(exit_defer.get(t, 0)) + 1
                    continue

            price = safe_float(px_open.get(t, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            sh = extract_shares(positions.pop(t))
            cash += sh * float(price) - FEE_PER_ORDER
            pp_state.pop(t, None)
            exit_defer.pop(t, None)
            orders.append({
                "Date": str(exec_date.date()),
                "Side": "SELL",
                "Ticker": t,
                "Shares": sh,
                "Price": float(price),
                "Fee": FEE_PER_ORDER,
                "Reason": "EXIT_NOT_TARGET",
            })
    port_val_open = cash
    for t, pos_val in positions.items():
        sh = extract_shares(pos_val)
        if sh <= 0:
            continue
        port_val_open += sh * float(px_open[t])

    # Delta rebalance
    for t, tgt_val in targets_val.items():
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
                pp_state.pop(t, None)
                exit_defer.pop(t, None)
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
                positions[t] = cur_sh + buy_sh

                if t not in pp_state:
                    pp_state[t] = {"entry_px": price, "entry_idx": int(last_idx)+1, "peak": price, "armed": False, "arm_idx": None}
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
    port["pp_state"] = pp_state
    port["exit_defer"] = exit_defer
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
    if not orders:
        msg_lines.append("ORDERS: none")
    else:
        msg_lines.append("ORDERS:")
        for o in orders:
            msg_lines.append(f"- {o['Side']} {o['Ticker']} @ {o['Price']:.2f} (fee {o['Fee']:.2f})")

    send_telegram("\n".join(msg_lines))


if __name__ == "__main__":
    main()
