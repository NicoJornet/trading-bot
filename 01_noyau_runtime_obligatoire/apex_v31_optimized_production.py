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
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# SETTINGS
# =============================================================================

# Live master universe excludes the 2 hard exclusions (MRNA, FTNT) on purpose.
MASTER_UNIVERSE: List[str] = [
    "AAPL","ABBV","ABNB","ABT","ADBE","ADI","AEM","AIR.PA","ALB","AMAT","AMD","AMGN","AMZN","ANET",
    "AON","APD","APH","APP","ASML","ATO.PA","AVAV","AVGO","AXON","BA","BA.L","BHP","BKNG","BLK",
    "BE","BMY","BN.PA","BNP.PA","BP","BRK-B","BSX","BWXT","BX","CA.PA","CAP.PA","CARR","CAT","CCJ","CDNS",
    "CEG","CMG","CMI","COP","COST","CRM","CRWD","CS.PA","CVX","DASH","DBC","DDOG","DE","DG.PA",
    "DHR","DNN","DSY.PA","DVN","EL.PA","EMR","ENGI.PA","ENI.MI","EOG","EQNR","ETN","FAST","FCX","FER",
    "FNV","GD","GE","GILD","GLD","GOLD","GOOGL","HAG.DE","HAL","HD","HEI","HII","HO.PA","HON",
    "HWM","INTU","ISRG","ITA","JNJ","JPM","KER.PA","KKR","KLAC","KMI","KO","KTOS","LDO.MI","LEU",
    "LHX","LIN","LITE","LLY","LMT","LNG","LOW","LRCX","LULU","MA","MARA","MC.PA","MCK","MDT","UI",
    "META","MMC","MPC","MRK","MRVL","MS","MSFT","MSTR","MTD","MU","NEM","NET","NFLX","NKE",
    "NOC","NOW","NVDA","NVO","NXE","NXPI","ORA.PA","ORCL","OXY","FIEE","PANW","PFE","PG","PH",
    "PLTR","PM","PWR","QCOM","QQQ","RACE","REGN","REMX","RHM.DE","RI.PA","RIO","RIOT","RKLB","RMS.PA",
    "ROP","RTX","SAAB-B.ST","SAF.PA","SAN.PA","SBUX","SCCO","SCHW","SGO.PA","SHEL","SHOP","SLB","SLV","SMCI",
    "SNDK","SPGI","SPOT","SPY","SQM","STMPA.PA","SU.PA","SYK","TDG","TDY","TECK","TJX","TMO","TMUS","TSLA","TSM",
    "TT","TTE","TTE.PA","TXN","UBER","UEC","UNH","URA","V","VALE","VIE.PA","VLO","VRT","VRTX",
    "WDC","WDAY","WELL","WLN.PA","WM","WMB","WMT","WPM","XAR","XLE","XLP","XLU","XLV","XME","XOM",
    "ZS","ZTS",
]

# Current reserve list: names kept in the data/master universe but not traded in
# the active live basket because they have shown no top-15 trend relevance in the
# research window and do not change the baseline when removed.
RESERVE_UNIVERSE: Tuple[str, ...] = (
    "BN.PA","CA.PA","MDT","AMGN","V","ROP","MMC","UNH","ABNB","ORA.PA","SYK","BMY","LIN","JNJ",
    "GD","EL.PA","SU.PA","LMT","BRK-B","HD","ADI","TXN","CS.PA","TTE","HON","KMI","BLK","ENGI.PA",
    "JPM","SHEL","EMR","RIO","CMI",
)

ROOT_DIR = Path(__file__).resolve().parent
ACTIVE_UNIVERSE_CSV = ROOT_DIR / "data" / "extracts" / "apex_tickers_active.csv"
DYNAMIC_APPROVED_CSV = ROOT_DIR / "data" / "dynamic_universe" / "dynamic_universe_approved_additions.csv"
DYNAMIC_SELECTED_ADDS_CSV = ROOT_DIR / "data" / "dynamic_universe" / "dynamic_universe_selected_additions.csv"
DYNAMIC_SELECTED_DEMS_CSV = ROOT_DIR / "data" / "dynamic_universe" / "dynamic_universe_selected_demotions.csv"


def load_ticker_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "ticker" not in df.columns:
        return []
    return [str(x).strip() for x in df["ticker"].dropna().astype(str) if str(x).strip()]


def dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_live_universe() -> List[str]:
    fallback_active = [ticker for ticker in MASTER_UNIVERSE if ticker not in RESERVE_UNIVERSE]
    active_from_csv = load_ticker_csv(ACTIVE_UNIVERSE_CSV)
    dynamic_selected_adds = load_ticker_csv(DYNAMIC_SELECTED_ADDS_CSV)
    dynamic_selected_dems = set(load_ticker_csv(DYNAMIC_SELECTED_DEMS_CSV))
    dynamic_approved = load_ticker_csv(DYNAMIC_APPROVED_CSV)
    base = active_from_csv or fallback_active
    overlay_adds = dynamic_selected_adds or dynamic_approved
    out = dedupe_keep_order(base + overlay_adds)
    return [ticker for ticker in out if ticker not in dynamic_selected_dems]


UNIVERSE: List[str] = build_live_universe()

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
    "RKLB","MSTR","KTOS","LEU","SMCI","APP","MARA","RIOT","DNN","VRT","RHM.DE","PLTR","COP","UEC","TSLA"
}

# OEG2 conditional (risk_set): veto entry if overextended vs SMA220 AND breaks SMA20
OEG2_ENABLE = 1
OEG2_DIST_TH = 0.85        # distSMA220 > 0.85
OEG2_SMA20_WIN = 20

# TailVeto spike (risk_set): veto entry if ATR% high AND vol spike high
TAILVETO_ENABLE = 1
TAIL_ATR_TH = 0.055        # 5.5%
TAIL_SPIKE_TH = 1.10       # vol20/vol60

# MIE: Momentum Invalidation Exit
# Disabled in the current baseline because the research replay shows that the
# block systematically exits many names into rebounds rather than protecting
# capital in the current system stack.
MIE_ENABLE = 0
MIE_RS63_TH = -0.01        # exit if R63 < -1% after the minimum hold
MIE_MIN_HOLD_DAYS = 5
MIE_EXEMPT_RANK_MAX = 8    # do not invalidate names that are still clear leaders
MIE_DD60_TH = 0.08         # require at least -8% off the 60d high before MIE can fire

# Profit protection: arm on MFE, then trail from peak after a short delay.
PP_ENABLE = 1
PP_MFE_TRIGGER = 0.34
PP_TRAIL_DD = 0.08
PP_MIN_DAYS_AFTER_ARM = 3

# Exit smoothing: defer the exit of losing names that still keep trend support.
EXITSMOOTH_ENABLE = 1
EXITSMOOTH_MAX_DEFERS = 3
EXITSMOOTH_REQUIRE_TREND = 1
EXITSMOOTH_REQUIRE_POS_RS63 = 0

# Leader Overweight (A022): boost top name weight, renormalize
LEADER_OVW_ENABLE = 1
LEADER_ALPHA = 0.22
LEADER_W_CAP = 0.55
SMA_WIN = 220
VOL_WIN = 20

# =============================================================================
# CHAMPION D ADDITIONS
# =============================================================================
SLOT3_GATE_ENABLE = 1
SLOT3_MAX_RANK = 4
SLOT3_LEADER_EXEMPT_TOPN = 12

QUALITY_FILTER_ENABLE = 1
Q_SLOT2_ENABLE = 1
Q_SLOT3_ENABLE = 1
Q_PERSIST_WIN = 5
Q_SLOT2_RANK_TH = 2
Q_SLOT3_RANK_TH = 5
Q_MIN_COUNT = 5
Q_LEADER_EXEMPT_TOPN = 12
Q_REQUIRE_POS_RS63 = 1


TOPK = 3
RANK_POOL = 15
KEEP_RANK = 5
REGIME_ENABLE = 1
REGIME_MKTVOL_ENABLE = 1
REGIME_MKTVOL_WIN = 20
REGIME_MKTVOL_TH = 0.40
REGIME_BREADTH_TH = 0.40
REGIME_CONFIRM_ALL = 1
REGIME_WEAK_TOPK = 2
REGIME_SPY_SMA_FILTER = 0

REB_EVERY_N_DAYS = 10
DELTA_REBAL = 0.10  # 10%

CORR_WIN = 63
CORR_GATE = 0.92
CORR_PICK = 0.80
CORR_SCAN = 10
CORR_HELD_ENABLE = 1

# Soft guard from negative-PnL correlation analysis: avoid stacking multiple
# names from the same fragile cluster in a 3-slot portfolio.
# Enabled in BEST_ALGO_183: avoid stacking fragile correlated losers in 3 slots.
LOSS_CLUSTER_GUARD_ENABLE = 1
LOSS_CLUSTER_MAX_PER_CLUSTER = 1
LOSS_CLUSTERS: Tuple[Tuple[str, ...], ...] = (
    ("AEM", "PAAS", "WPM", "SLV"),
    ("COP", "OXY", "ENI.MI"),
    ("NET", "ZS"),
    ("AVAV", "AXON"),
    ("DNN", "UEC"),
)
LOSS_ENTRY_GUARD_ENABLE = 1
LOSS_ENTRY_DIST_TH = 0.90
LOSS_ENTRY_DD60_TH = 0.05
LOSS_REENTRY_COOLDOWN_DAYS = 29
HELD_RELEASE_ENABLE = 1
HELD_RELEASE_LOSS_CLUSTER_ONLY = 0
HELD_RELEASE_WEAK_RANK_TH = 8
HELD_RELEASE_NEW_TOPM = 5
HELD_RELEASE_MIN_NEW = 1

# Re-entry on healthy pullbacks for established long-term leaders.
REENTRY_ENABLE = 1
REENTRY_LT_TOPN = 12
REENTRY_DD60_MIN = 0.05
REENTRY_DD60_MAX = 0.25
REENTRY_R21_MAX = 0.03
REENTRY_BONUS = 0.15
REENTRY_BREADTH_MIN = 0.50
REENTRY_RS63_MIN = 0.01

# Dynamic concentration guard: when a fragile cluster turns weak on a short
# lookback, keep only one name from that cluster instead of stacking copies.
STATE_CLUSTER_GUARD_ENABLE = 1
STATE_CLUSTER_MAX_PER_CLUSTER = 1
STATE_CLUSTER_GUARD_ONLY_DUP = 1
STATE_CLUSTER_R21_MAX = 0.0
STATE_CLUSTER_LOOKBACK = 21
STATE_CLUSTER_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("MSTR", "MARA", "RIOT"),
)

# Dynamic tech-sector sizing cap: when the technology basket is weak on 21d,
# avoid carrying more than 70% total weight in stacked technology names.
STATE_WEIGHT_CAP_ENABLE = 1
STATE_WEIGHT_TOTAL_CAP = 0.70
STATE_WEIGHT_INDIV_CAP = 1.0
STATE_WEIGHT_ONLY_DUP = 1
STATE_WEIGHT_R21_MAX = 0.0
STATE_WEIGHT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    (
        "AAPL","ADBE","AMAT","AMD","ANET","APH","ASML","ATO.PA","AVGO","AXTI","CAP.PA","CDNS",
        "CRM","CRWD","DDOG","DSY.PA","INTU","KLAC","LITE","LRCX","MRVL","MSFT","MSTR","MU",
        "NET","NOW","NVDA","NXPI","ORCL","PANW","PLTR","QCOM","SHOP","SMCI","SNDK","STMPA.PA",
        "TDY","TSM","UBER","UI","WDAY","WDC","WLN.PA","ZS",
    ),
)

# Narrow convex non-PP exit guard: allow deeper defer only for a few uranium
# names that historically suffered from premature EXIT_NOT_TARGET sells.
CONVEX_EXIT_GUARD_ENABLE = 1
CONVEX_EXIT_GUARD_TICKERS: Tuple[str, ...] = ("CCJ", "DNN", "LEU", "NXE", "UEC")
CONVEX_EXIT_GUARD_MAX_DEFERS = 8
CONVEX_EXIT_GUARD_DD60_TH = 0.05
CONVEX_EXIT_GUARD_RS63_MIN = -999.0
CONVEX_EXIT_GUARD_LT_TOPN = 0

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


def normalize_position_record(pos_val, default_entry_date: Optional[str] = None) -> dict:
    sh = extract_shares(pos_val)
    if isinstance(pos_val, dict):
        rec = dict(pos_val)
        rec["shares"] = sh
    else:
        rec = {"shares": sh}
    if default_entry_date and not rec.get("entry_date"):
        rec["entry_date"] = default_entry_date
    return rec


def extract_cost_basis(pos_val) -> float:
    if not isinstance(pos_val, dict):
        return np.nan
    sh = extract_shares(pos_val)
    if sh <= 0:
        return np.nan
    cost = safe_float(pos_val.get("cost_eur", np.nan))
    if np.isfinite(cost):
        return cost
    avg = safe_float(pos_val.get("avg_price_eur", np.nan))
    if np.isfinite(avg):
        return avg * sh
    return np.nan


def enrich_buy_position(pos_val, buy_sh: float, price: float, exec_date: pd.Timestamp) -> dict:
    rec = normalize_position_record(pos_val)
    cur_sh = extract_shares(rec)
    cur_cost = extract_cost_basis(rec)
    new_sh = cur_sh + float(buy_sh)
    if new_sh <= 0:
        return {"shares": 0.0}

    new_cost = float(buy_sh) * float(price)
    if np.isfinite(cur_cost):
        new_cost += cur_cost
    elif cur_sh > 0:
        avg = safe_float(rec.get("avg_price_eur", np.nan))
        if np.isfinite(avg):
            new_cost += cur_sh * avg

    rec["shares"] = new_sh
    if np.isfinite(new_cost):
        rec["cost_eur"] = new_cost
        rec["avg_price_eur"] = new_cost / new_sh
    elif cur_sh <= 0:
        rec["cost_eur"] = float(buy_sh) * float(price)
        rec["avg_price_eur"] = float(price)

    if cur_sh <= 1e-12:
        rec["entry_date"] = str(exec_date.date())
    elif not rec.get("entry_date"):
        rec["entry_date"] = str(exec_date.date())
    return rec


def reduce_position_after_sell(pos_val, sell_sh: float) -> Optional[dict]:
    rec = normalize_position_record(pos_val)
    cur_sh = extract_shares(rec)
    prior_cost = extract_cost_basis(rec)
    new_sh = cur_sh - float(sell_sh)
    if new_sh <= 1e-10:
        return None

    rec["shares"] = new_sh
    avg = safe_float(rec.get("avg_price_eur", np.nan))
    if np.isfinite(avg):
        rec["cost_eur"] = avg * new_sh
    elif np.isfinite(prior_cost) and cur_sh > 0:
        rec["cost_eur"] = prior_cost * (new_sh / cur_sh)
    return rec


def build_sell_audit(pos_val, sell_sh: float, sell_price: float, fee: float) -> dict:
    out = {}
    rec = normalize_position_record(pos_val)
    cur_sh = extract_shares(rec)
    if cur_sh <= 0:
        return out

    avg = safe_float(rec.get("avg_price_eur", np.nan))
    if not np.isfinite(avg):
        cost = extract_cost_basis(rec)
        if np.isfinite(cost):
            avg = cost / cur_sh

    if np.isfinite(avg):
        sold_cost = avg * float(sell_sh)
        realized_pnl = float(sell_sh) * float(sell_price) - sold_cost - float(fee)
        out["RealizedPnL"] = realized_pnl
        out["RealizedPnLPct"] = ((realized_pnl / sold_cost) * 100.0) if sold_cost > 0 else np.nan

    if rec.get("entry_date"):
        out["OpenDate"] = rec.get("entry_date")
    return out


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
    r5 = c / c.shift(5) - 1.0
    r21 = c / c.shift(21) - 1.0

    r63 = c / c.shift(63) - 1.0
    r126 = c / c.shift(126) - 1.0
    r252 = c / c.shift(252) - 1.0
    if "SPY" in c.columns:
        spy_r63 = c["SPY"] / c["SPY"].shift(63) - 1.0
        rs63 = (1.0 + r63).div(1.0 + spy_r63, axis=0) - 1.0
    else:
        rs63 = r63 * 0.0

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
    ranks = score.rank(axis=1, ascending=False, method="min")
    rank126 = r126.rank(axis=1, ascending=False, method="min")
    rank252 = r252.rank(axis=1, ascending=False, method="min")

    return dict(
        score=score,
        ranks=ranks,
        rank126=rank126,
        rank252=rank252,
        sma220=sma220,
        sma20=sma20,
        vol20=vol20,
        vol60=vol60,
        vol_spike=vol_spike,
        atrp20=atrp20,
        dd60=dd60,
        dist_sma220=dist_sma220,
        r63=r63,
        rs63=rs63,
        r5=r5,
        r21=r21,
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
    held: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
) -> List[str]:
    held = held or []
    tickers = tickers or ranked
    corr = corr_matrix(win_slice)
    idx = {t: i for i, t in enumerate(tickers)}

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


def build_cluster_map(clusters: Tuple[Tuple[str, ...], ...]) -> Dict[str, int]:
    cluster_map: Dict[str, int] = {}
    for idx, cluster in enumerate(clusters or ()):
        for ticker in cluster:
            cluster_map[str(ticker)] = idx
    return cluster_map


def flatten_clusters(clusters: Tuple[Tuple[str, ...], ...]) -> set[str]:
    names: set[str] = set()
    for cluster in clusters or ():
        for ticker in cluster:
            names.add(str(ticker))
    return names


def has_group_duplicates(selected: List[str], group_map: Dict[str, int]) -> bool:
    seen: set[int] = set()
    for ticker in selected:
        group_id = group_map.get(ticker)
        if group_id is None:
            continue
        if group_id in seen:
            return True
        seen.add(group_id)
    return False


def active_state_group_map_for_date(
    tickers: List[str],
    clusters: Tuple[Tuple[str, ...], ...],
    r_frame: pd.DataFrame,
    date: pd.Timestamp,
    max_state: float,
) -> Dict[str, int]:
    if r_frame.empty or date not in r_frame.index:
        return {}
    group_map = build_cluster_map(clusters)
    if not group_map:
        return {}
    row = r_frame.loc[date]
    active: Dict[str, int] = {}
    for ticker in tickers:
        group_id = group_map.get(ticker)
        if group_id is None:
            continue
        members = [name for name in clusters[group_id] if name in row.index]
        if not members:
            continue
        state = pd.to_numeric(row.reindex(members), errors="coerce").mean()
        if pd.notna(state) and float(state) <= max_state:
            active[ticker] = group_id
    return active


def apply_held_release_rule(
    held: List[str],
    ranked: List[str],
    loss_names: set[str],
    weak_rank_th: int,
    newcomer_topm: int,
    min_new: int,
    loss_only: bool = True,
) -> Tuple[List[str], List[str]]:
    if not held or not ranked or weak_rank_th <= 0 or newcomer_topm <= 0 or min_new <= 0:
        return list(held), []

    rankpos = {t: i + 1 for i, t in enumerate(ranked)}
    weakest = max(held, key=lambda h: rankpos.get(h, 10**9))
    if loss_only and weakest not in loss_names:
        return list(held), []
    if rankpos.get(weakest, 10**9) <= weak_rank_th:
        return list(held), []

    newcomers = [t for t in ranked[:newcomer_topm] if t not in held]
    if len(newcomers) < min_new:
        return list(held), []

    return [h for h in held if h != weakest], [weakest]


def apply_cluster_limit(
    primary: List[str],
    fallback: List[str],
    topk: int,
    cluster_map: Dict[str, int],
    max_per_cluster: int = 1,
) -> List[str]:
    chosen: List[str] = []
    cluster_counts: Dict[int, int] = {}

    def can_take(ticker: str) -> bool:
        cluster_id = cluster_map.get(ticker)
        if cluster_id is None:
            return True
        return cluster_counts.get(cluster_id, 0) < max_per_cluster

    def take(ticker: str) -> None:
        cluster_id = cluster_map.get(ticker)
        if cluster_id is not None:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        chosen.append(ticker)

    for ticker in primary:
        if ticker in chosen:
            continue
        if not can_take(ticker):
            continue
        take(ticker)
        if len(chosen) >= topk:
            return chosen[:topk]

    for ticker in fallback:
        if ticker in chosen:
            continue
        if not can_take(ticker):
            continue
        take(ticker)
        if len(chosen) >= topk:
            break

    return chosen[:topk]


def invvol_weights(
    vol_row: pd.Series,
    tickers: List[str],
    leader_ticker: Optional[str] = None,
    leader_alpha: float = 0.0,
    leader_w_cap: float = 1.0,
) -> Dict[str, float]:
    v = vol_row.reindex(tickers).replace(0, np.nan)
    inv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty:
        return {}
    inv = inv / inv.sum()
    w = {t: float(inv.loc[t]) for t in inv.index}
    if leader_ticker and leader_ticker in w and leader_alpha > 0:
        w[leader_ticker] *= (1.0 + leader_alpha)
        s = sum(w.values())
        if s > 0:
            w = {k: v / s for k, v in w.items()}
        if leader_w_cap < 1.0:
            capped = min(float(w[leader_ticker]), float(leader_w_cap))
            excess = float(w[leader_ticker]) - capped
            if excess > 0:
                w[leader_ticker] = capped
                others = [k for k in w if k != leader_ticker]
                other_sum = sum(w[k] for k in others)
                if other_sum > 0:
                    for k in others:
                        w[k] += excess * (w[k] / other_sum)
                else:
                    s = sum(w.values())
                    if s > 0:
                        w = {k: v / s for k, v in w.items()}
    return w


def _candidate_quality_ok(ranks: pd.DataFrame, rank126: pd.DataFrame, rank252: pd.DataFrame, rs63: pd.DataFrame, d: pd.Timestamp, cand: str, slot_num: int) -> bool:
    if not QUALITY_FILTER_ENABLE:
        return True
    if slot_num == 2 and not Q_SLOT2_ENABLE:
        return True
    if slot_num == 3 and not Q_SLOT3_ENABLE:
        return True
    try:
        r126 = rank126.loc[d, cand]
        r252 = rank252.loc[d, cand]
        if pd.notna(r126) and pd.notna(r252) and int(r126) <= Q_LEADER_EXEMPT_TOPN and int(r252) <= Q_LEADER_EXEMPT_TOPN:
            return True
    except Exception:
        pass
    try:
        end_loc = ranks.index.get_loc(d)
    except KeyError:
        return True
    start_loc = max(0, end_loc - Q_PERSIST_WIN + 1)
    wdates = ranks.index[start_loc:end_loc + 1]
    rank_th = Q_SLOT2_RANK_TH if slot_num == 2 else Q_SLOT3_RANK_TH
    rr = ranks.loc[wdates, cand]
    count = int((rr <= rank_th).sum())
    if count < Q_MIN_COUNT:
        return False
    if Q_REQUIRE_POS_RS63:
        try:
            if float(rs63.loc[d, cand]) <= 0.0:
                return False
        except Exception:
            return False
    return True


def pretty_weights_and_targets(
    total_equity: float,
    cash: float,
    desired: List[str],
    held: List[str],
    vol_row: pd.Series,
    fee_per_order: float,
    leader_ticker: Optional[str] = None,
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
    w = invvol_weights(
        vol_row,
        desired,
        leader_ticker=leader_ticker if LEADER_OVW_ENABLE else None,
        leader_alpha=LEADER_ALPHA if LEADER_OVW_ENABLE else 0.0,
        leader_w_cap=LEADER_W_CAP if LEADER_OVW_ENABLE else 1.0,
    )
    if not w:
        return {}, {}, 0.0, 0.0

    missing_buys = [t for t in desired if t not in set(held)]
    fees_reserved = fee_per_order * len(missing_buys)

    investable_equity = max(total_equity - fees_reserved, 0.0)
    targets = {t: float(w.get(t, 0.0) * investable_equity) for t in desired}
    return w, targets, investable_equity, fees_reserved


def fmt_eur(x: float) -> str:
    try:
        return f"{float(x):,.2f} EUR".replace(",", " ")
    except Exception:
        return "n/a"


def fmt_name_list(names: List[str]) -> str:
    return ", ".join(names) if names else "(none)"


def build_rank_lines(ranked: List[str], srow: pd.Series, limit: int) -> List[str]:
    if not ranked:
        return ["- (none)"]
    lines = []
    for i, t in enumerate(ranked[:limit], 1):
        try:
            score_txt = f"{float(srow.loc[t]):.4f}"
        except Exception:
            score_txt = "n/a"
        lines.append(f"- {i}. {t} | score {score_txt}")
    return lines


def build_target_lines(desired: List[str], weights: Dict[str, float], targets: Dict[str, float]) -> List[str]:
    if not desired:
        return ["- (none)"]
    lines = []
    for i, t in enumerate(desired, 1):
        w = 100.0 * float(weights.get(t, 0.0))
        tgt = float(targets.get(t, 0.0))
        lines.append(f"- {i}. {t} | {w:.1f}% | target {fmt_eur(tgt)}")
    return lines


def build_order_lines(orders: List[dict]) -> List[str]:
    if not orders:
        return ["- none"]
    lines = []
    for o in orders:
        side = str(o.get("Side", "?")).upper()
        ticker = str(o.get("Ticker", "?"))
        shares = safe_float(o.get("Shares", np.nan))
        price = safe_float(o.get("Price", np.nan))
        reason = str(o.get("Reason", ""))
        parts = [f"- {side} {ticker} {shares:.6f} @ {price:.2f}"]
        if reason:
            parts.append(reason)
        pnl = safe_float(o.get("RealizedPnL", np.nan))
        pnl_pct = safe_float(o.get("RealizedPnLPct", np.nan))
        if np.isfinite(pnl):
            pnl_txt = f"PnL {pnl:+,.2f} EUR".replace(",", " ")
            if np.isfinite(pnl_pct):
                pnl_txt += f" ({pnl_pct:+.1f}%)"
            parts.append(pnl_txt)
        amount = safe_float(o.get("Amount", np.nan))
        if side == "BUY" and np.isfinite(amount):
            parts.append(f"amount {fmt_eur(amount)}")
        lines.append(" | ".join(parts))
    return lines


def build_telegram_snapshot(
    *,
    header: str,
    cash: float,
    pos_value: float,
    held_before: List[str],
    desired: List[str],
    ranked: List[str],
    srow: pd.Series,
    corr_gate_hit: bool,
    weights: Dict[str, float],
    targets: Dict[str, float],
    fees_reserved: float,
    investable_equity: float,
    do_rebalance: bool,
    orders: List[dict],
    rebalance_days_since: Optional[int] = None,
    rebalance_interval: int = REB_EVERY_N_DAYS,
    last_rebalance_label: Optional[str] = None,
    exec_date: Optional[pd.Timestamp] = None,
    held_after: Optional[List[str]] = None,
    fallback_used: Optional[List[str]] = None,
    held_released: Optional[List[str]] = None,
) -> str:
    total = cash + pos_value
    lines = [
        header,
        "",
        "STATE",
        f"- Cash: {fmt_eur(cash)}",
        f"- Positions: {fmt_eur(pos_value)}",
        f"- Total: {fmt_eur(total)}",
        f"- Rebalance day: {'yes' if do_rebalance else 'no'}",
        f"- Corr gate: {'yes' if corr_gate_hit else 'no'}",
        "",
        "PORTFOLIO",
        f"- Held now: {fmt_name_list(held_before)}",
        f"- Target basket: {fmt_name_list(desired)}",
    ]
    if rebalance_days_since is None:
        lines.append("- Rebalance counter: first run")
    else:
        lines.append(f"- Trading days since last rebalance: {rebalance_days_since}/{rebalance_interval}")
        if last_rebalance_label:
            lines.append(f"- Last rebalance seen: {last_rebalance_label}")
        if rebalance_days_since >= rebalance_interval:
            lines.append("- Rebalance status: due now")
        else:
            lines.append(f"- Rebalance status: in {rebalance_interval - rebalance_days_since} trading day(s)")
    if held_after is not None and do_rebalance:
        lines.append(f"- Held after orders: {fmt_name_list(held_after)}")
    if held_released:
        lines.append(f"- Held release: {fmt_name_list(held_released)}")
    if exec_date is not None and do_rebalance:
        lines.append(f"- Exec date: {exec_date.date()}")
    if fallback_used:
        lines.append(f"- Open fallback used: {fmt_name_list(sorted(fallback_used))}")

    lines.extend([
        "",
        "TARGETS",
        *build_target_lines(desired, weights, targets),
        f"- Fees reserved: {fmt_eur(fees_reserved)}",
        f"- Investable equity: {fmt_eur(investable_equity)}",
        "",
        "TOP MOMENTUM",
        *build_rank_lines(ranked, srow, limit=5 if do_rebalance else 10),
        "",
        "ORDERS",
        *build_order_lines(orders),
    ])
    return "\n".join(lines)


# =============================================================================
# Portfolio IO
# =============================================================================

def load_portfolio() -> dict:
    p = load_json(PORTFOLIO_FILE)
    if not p:
        p = {
            "cash": INITIAL_CASH,
            "positions": {},
            "entry_date": {},
            "pp_state": {},
            "exit_defer": {},
            "loss_last_exit_idx": {},
            "last_rebalance_idx": None,
            "last_rebalance_date": None,
            "last_dca_month": None,
        }
    p["cash"] = safe_float(p.get("cash", INITIAL_CASH), INITIAL_CASH)
    p["positions"] = p.get("positions", {}) or {}
    p["entry_date"] = p.get("entry_date", {}) or {}
    p["pp_state"] = p.get("pp_state", {}) or {}
    p["exit_defer"] = p.get("exit_defer", {}) or {}
    p["loss_last_exit_idx"] = p.get("loss_last_exit_idx", {}) or {}
    for ticker, pos_val in list(p["positions"].items()):
        nested_entry = None
        if isinstance(pos_val, dict):
            nested_entry = pos_val.get("entry_date")
        if nested_entry and ticker not in p["entry_date"]:
            p["entry_date"][ticker] = nested_entry
    if "last_rebalance_idx" not in p:
        p["last_rebalance_idx"] = None
    if "last_rebalance_date" not in p:
        p["last_rebalance_date"] = None
    if "last_dca_month" not in p:
        p["last_dca_month"] = None
    return p


def save_portfolio(p: dict) -> None:
    save_json(PORTFOLIO_FILE, p)


def append_trades(rows: List[dict]) -> None:
    hist = load_json(TRADES_FILE)
    if isinstance(hist, dict) and isinstance(hist.get("trades"), list):
        existing = hist.get("trades", [])
        next_id = max(
            [int(r.get("id", 0)) for r in existing if isinstance(r, dict)] + [0]
        ) + 1
        ts_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        for row in rows:
            shares = safe_float(row.get("Shares", 0.0), 0.0)
            price = safe_float(row.get("Price", np.nan))
            fee = safe_float(row.get("Fee", 0.0), 0.0)
            amount = shares * price if np.isfinite(price) else np.nan
            item = {
                "id": next_id,
                "date": row.get("Date"),
                "action": row.get("Side"),
                "ticker": row.get("Ticker"),
                "shares": shares,
                "price_eur": price if np.isfinite(price) else None,
                "amount_eur": amount if np.isfinite(amount) else None,
                "fee_eur": fee,
                "reason": row.get("Reason"),
                "ts": ts_now,
            }
            if "RealizedPnL" in row:
                pnl_val = safe_float(row.get("RealizedPnL", np.nan))
                if np.isfinite(pnl_val):
                    item["pnl_eur"] = pnl_val
            if "RealizedPnLPct" in row:
                pnl_pct = safe_float(row.get("RealizedPnLPct", np.nan))
                if np.isfinite(pnl_pct):
                    item["pnl_pct"] = pnl_pct
            if "OpenDate" in row:
                item["open_date"] = row.get("OpenDate")
            if "LastBuyDate" in row:
                item["last_buy_date"] = row.get("LastBuyDate")
            existing.append(item)
            next_id += 1

        sell_with_pnl = [
            r for r in existing
            if isinstance(r, dict) and str(r.get("action", "")).upper() == "SELL" and r.get("pnl_eur") is not None
        ]
        buy_dates = [
            str(r.get("date"))
            for r in existing
            if isinstance(r, dict) and str(r.get("action", "")).upper() == "BUY" and r.get("date")
        ]
        hist["summary"] = {
            "total_records": len(existing),
            "buys": sum(1 for r in existing if isinstance(r, dict) and str(r.get("action", "")).upper() == "BUY"),
            "sells": sum(1 for r in existing if isinstance(r, dict) and str(r.get("action", "")).upper() == "SELL"),
            "winning_sells": sum(1 for r in sell_with_pnl if safe_float(r.get("pnl_eur", np.nan)) > 0),
            "losing_sells": sum(1 for r in sell_with_pnl if safe_float(r.get("pnl_eur", np.nan)) < 0),
            "net_pnl_eur": round(sum(safe_float(r.get("pnl_eur", 0.0), 0.0) for r in sell_with_pnl), 2),
            "total_fees_eur": round(sum(safe_float(r.get("fee_eur", 0.0), 0.0) for r in existing if isinstance(r, dict)), 2),
            "win_rate_sells_pct": round(
                100.0 * sum(1 for r in sell_with_pnl if safe_float(r.get("pnl_eur", np.nan)) > 0) / len(sell_with_pnl),
                2,
            ) if sell_with_pnl else 0.0,
            "sell_records_with_pnl": len(sell_with_pnl),
            "portfolio_resets": sum(
                1 for r in existing if isinstance(r, dict) and str(r.get("action", "")).upper() == "RESET_PORTFOLIO"
            ),
            "notes": "Summary counts refer to BUY/SELL records; RESET_PORTFOLIO is excluded.",
            "last_buy_date": max(buy_dates) if buy_dates else None,
        }
        hist["trades"] = existing
        save_json(TRADES_FILE, hist)
        return

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
    active_from_csv = load_ticker_csv(ACTIVE_UNIVERSE_CSV)
    dynamic_selected_adds = load_ticker_csv(DYNAMIC_SELECTED_ADDS_CSV)
    dynamic_selected_dems = load_ticker_csv(DYNAMIC_SELECTED_DEMS_CSV)
    dynamic_approved = load_ticker_csv(DYNAMIC_APPROVED_CSV)
    print(
        "UNIVERSE_LAYER:"
        f" active_csv={'yes' if active_from_csv else 'no'}({len(active_from_csv)})"
        f" selected_adds={dynamic_selected_adds if dynamic_selected_adds else '[]'}"
        f" selected_dems={dynamic_selected_dems if dynamic_selected_dems else '[]'}"
        f" approved_adds={dynamic_approved if dynamic_approved else '[]'}"
    )
    print(
        "UNIVERSE_CHECK:"
        f" FIEE={('FIEE' in UNIVERSE)} UI={('UI' in UNIVERSE)}"
        f" PAAS={('PAAS' in UNIVERSE)} MELI={('MELI' in UNIVERSE)}"
    )
    held_tickers = [str(t).strip() for t in port.get("positions", {}).keys() if str(t).strip()]
    data_tickers = dedupe_keep_order(UNIVERSE + held_tickers)

    ohlcv = load_data_yfinance(data_tickers)

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
    ranks = sig["ranks"]
    rank126 = sig["rank126"]
    rank252 = sig["rank252"]
    sma220 = sig["sma220"]
    sma20 = sig["sma20"]
    vol20 = sig["vol20"]
    vol60 = sig["vol60"]
    vol_spike = sig["vol_spike"]
    atrp20 = sig["atrp20"]
    dd60 = sig["dd60"]
    dist_sma220 = sig["dist_sma220"]
    r63 = sig["r63"]
    rs63 = sig["rs63"]
    r5 = sig["r5"]
    r21 = sig["r21"]
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
    if last_reb is None:
        rebalance_days_since = None
        last_rebalance_label = port.get("last_rebalance_date") or None
    else:
        last_reb_idx = int(last_reb)
        rebalance_days_since = max(0, last_idx - last_reb_idx)
        if 0 <= last_reb_idx < len(cal):
            last_rebalance_label = f"{cal[last_reb_idx].date()} (idx {last_reb_idx})"
        elif port.get("last_rebalance_date"):
            last_rebalance_label = f"{port.get('last_rebalance_date')} (idx {last_reb_idx})"
        else:
            last_rebalance_label = f"idx {last_reb_idx} (outside current calendar)"
    print(
        "REBALANCE_STATUS:",
        "due" if do_rebalance else "not due",
        "|",
        f"trading_days_since={rebalance_days_since if rebalance_days_since is not None else 'first run'}",
        "|",
        f"last_rebalance={last_rebalance_label or 'n/a'}",
    )

    positions = port.get("positions", {}) or {}
    entry_date = port.get("entry_date", {}) or {}
    pp_state = port.get("pp_state", {}) or {}
    exit_defer = port.get("exit_defer", {}) or {}
    loss_last_exit_idx = port.get("loss_last_exit_idx", {}) or {}
    cash = float(port.get("cash", 0.0))
    loss_names = flatten_clusters(LOSS_CLUSTERS)

    def ensure_pp_state(ticker: str, entry_px: float, entry_idx: int) -> dict:
        st = pp_state.get(ticker)
        if st is None:
            st = {
                "entry_px": float(entry_px),
                "peak": float(entry_px),
                "armed": (PP_MFE_TRIGGER <= 0.0),
                "arm_idx": (entry_idx if PP_MFE_TRIGGER <= 0.0 else None),
                "entry_idx": int(entry_idx),
            }
            pp_state[ticker] = st
        return st

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

    if TAILVETO_ENABLE:
        tail = (
            (atrp20.loc[last_date] >= TAIL_ATR_TH) &
            ((vol20.loc[last_date] / (vol60.loc[last_date] + 1e-12)) >= TAIL_SPIKE_TH)
        )
        mask = pd.Index(ohlcv.close.columns).isin(list(RISK_SET_15))
        tail = tail & pd.Series(mask, index=ohlcv.close.columns)
        elig = elig & (~tail.fillna(False))

    if OEG2_ENABLE:
        dist = (ohlcv.close_ffill.loc[last_date] / (sma220.loc[last_date] + 1e-12)) - 1.0
        cond = dist >= OEG2_DIST_TH
        mask2 = pd.Index(ohlcv.close.columns).isin(list(RISK_SET_15))
        cond = cond & pd.Series(mask2, index=ohlcv.close.columns)
        cond = cond & (ohlcv.close_ffill.loc[last_date] < sma20.loc[last_date])
        elig = elig & (~cond.fillna(False))

    if loss_names:
        held_names = set(positions.keys())
        loss_candidate_mask = pd.Series(
            pd.Index(ohlcv.close.columns).isin(list(loss_names)),
            index=ohlcv.close.columns,
        ) & (~pd.Series(pd.Index(ohlcv.close.columns).isin(list(held_names)), index=ohlcv.close.columns))

        if LOSS_REENTRY_COOLDOWN_DAYS > 0 and loss_last_exit_idx:
            cooldown_mask = pd.Series(False, index=ohlcv.close.columns)
            for ticker, exit_idx in loss_last_exit_idx.items():
                if ticker in cooldown_mask.index and (last_idx - int(exit_idx)) < LOSS_REENTRY_COOLDOWN_DAYS:
                    cooldown_mask.loc[ticker] = True
            elig = elig & (~(loss_candidate_mask & cooldown_mask).fillna(False))

        if LOSS_ENTRY_GUARD_ENABLE:
            loss_guard = (
                loss_candidate_mask &
                (dist_sma220.loc[last_date] >= LOSS_ENTRY_DIST_TH) &
                (dd60.loc[last_date] >= -LOSS_ENTRY_DD60_TH)
            )
            elig = elig & (~loss_guard.fillna(False))

    breadth_now = np.nan
    breadth_valid = enough_history.reindex(ohlcv.close.columns, fill_value=False) & ohlcv.close_ffill.loc[last_date].notna() & sma220.loc[last_date].notna()
    if bool(breadth_valid.any()):
        breadth_now = float((ohlcv.close_ffill.loc[last_date, breadth_valid] > sma220.loc[last_date, breadth_valid]).mean())

    score_rank = score.loc[last_date].copy()
    if REENTRY_ENABLE:
        leader_lt = (rank126.loc[last_date] <= REENTRY_LT_TOPN) & (rank252.loc[last_date] <= REENTRY_LT_TOPN)
        pullback_lt = (
            (dd60.loc[last_date] <= -REENTRY_DD60_MIN) &
            (dd60.loc[last_date] >= -REENTRY_DD60_MAX) &
            (r21.loc[last_date] <= REENTRY_R21_MAX)
        )
        reentry_mask = leader_lt & pullback_lt
        if REENTRY_BREADTH_MIN > 0.0:
            reentry_mask = reentry_mask & bool(np.isfinite(breadth_now) and (breadth_now >= REENTRY_BREADTH_MIN))
        if REENTRY_RS63_MIN is not None:
            reentry_mask = reentry_mask & (rs63.loc[last_date] >= float(REENTRY_RS63_MIN))
        score_rank = score_rank + REENTRY_BONUS * reentry_mask.astype(float)

    srow = score_rank.where(elig, np.nan).dropna()
    ranked = list(srow.sort_values(ascending=False).head(RANK_POOL).index)

    print("TOP 15 MOMENTUM:")
    if len(ranked) == 0:
        print("(none)")
    else:
        for i, t in enumerate(ranked[:15], 1):
            print(f"{i}. {t} score {float(srow.loc[t]):.4f}")
    print("")

    active_topk = TOPK
    regime_flags: List[bool] = []
    if REGIME_ENABLE:
        if REGIME_SPY_SMA_FILTER and ("SPY" in ohlcv.close.columns):
            spy_close = float(ohlcv.close_ffill.loc[last_date, "SPY"]) if pd.notna(ohlcv.close_ffill.loc[last_date, "SPY"]) else np.nan
            spy_sma = float(sma220.loc[last_date, "SPY"]) if pd.notna(sma220.loc[last_date, "SPY"]) else np.nan
            if np.isfinite(spy_close) and np.isfinite(spy_sma):
                regime_flags.append(spy_close <= spy_sma)
        if REGIME_MKTVOL_ENABLE and ("SPY" in ret1.columns):
            spy_vol = ret1["SPY"].rolling(REGIME_MKTVOL_WIN, min_periods=REGIME_MKTVOL_WIN).std() * np.sqrt(252.0)
            spy_vol_now = float(spy_vol.loc[last_date]) if pd.notna(spy_vol.loc[last_date]) else np.nan
            if np.isfinite(spy_vol_now):
                regime_flags.append(spy_vol_now >= REGIME_MKTVOL_TH)
        if REGIME_BREADTH_TH > 0.0 and np.isfinite(breadth_now):
            regime_flags.append(breadth_now < REGIME_BREADTH_TH)
        if regime_flags:
            regime_weak = all(regime_flags) if REGIME_CONFIRM_ALL else any(regime_flags)
            if regime_weak:
                active_topk = max(1, min(REGIME_WEAK_TOPK, TOPK))

    desired_ranked = ranked[:active_topk]
    cluster_candidates = list(desired_ranked)

    current = list(positions.keys())
    loss_names = flatten_clusters(LOSS_CLUSTERS)
    released_held: List[str] = []

    corr_gate_hit = 0
    if len(ranked) >= 2:
        loc_full = ohlcv.close.index.get_loc(last_date)
        w0 = max(0, loc_full - CORR_WIN + 1)
        held = [t for t in positions.keys() if t in ret1.columns]
        held_for_corr = list(held)
        if HELD_RELEASE_ENABLE:
            held_for_corr, released_held = apply_held_release_rule(
                held=held,
                ranked=ranked,
                loss_names=loss_names,
                weak_rank_th=HELD_RELEASE_WEAK_RANK_TH,
                newcomer_topm=HELD_RELEASE_NEW_TOPM,
                min_new=HELD_RELEASE_MIN_NEW,
                loss_only=bool(HELD_RELEASE_LOSS_CLUSTER_ONLY),
            )
        if CORR_HELD_ENABLE:
            corr_names = list(dict.fromkeys(held_for_corr + ranked))
        else:
            corr_names = list(ranked)
        win_slice = ret1.iloc[w0:loc_full + 1][corr_names].to_numpy()
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
                    held=held_for_corr if CORR_HELD_ENABLE else None,
                    tickers=corr_names,
                )

    current = list(positions.keys())
    desired = apply_keep_rank(current=current, ranked=desired_ranked, topk=active_topk, keep_rank=KEEP_RANK)

    if QUALITY_FILTER_ENABLE:
        kept = [t for t in list(positions.keys()) if t in desired]
        desired_q = list(kept)
        quality_candidates = list(kept)
        for cand in desired_ranked:
            if cand in quality_candidates:
                continue
            slot_num = min(len(quality_candidates) + 1, active_topk)
            if slot_num in (2, 3) and (cand not in positions):
                if not _candidate_quality_ok(ranks, rank126, rank252, r63, last_date, cand, slot_num):
                    continue
            quality_candidates.append(cand)
            if len(desired_q) >= active_topk:
                continue
            desired_q.append(cand)
            if len(desired_q) >= active_topk:
                break
        desired = desired_q[:active_topk]
        cluster_candidates = quality_candidates

    if SLOT3_GATE_ENABLE and len(desired) >= 3:
        t3 = desired[2]
        try:
            rank3 = int(ranks.loc[last_date, t3]) if pd.notna(ranks.loc[last_date, t3]) else 10**9
        except Exception:
            rank3 = 10**9
        try:
            is_lt_leader = bool((rank126.loc[last_date, t3] <= SLOT3_LEADER_EXEMPT_TOPN) and (rank252.loc[last_date, t3] <= SLOT3_LEADER_EXEMPT_TOPN))
        except Exception:
            is_lt_leader = False
        if (rank3 > SLOT3_MAX_RANK) and (not is_lt_leader):
            desired = desired[:2]

    if LOSS_CLUSTER_GUARD_ENABLE:
        cluster_map = build_cluster_map(LOSS_CLUSTERS)
        if cluster_map:
            desired = apply_cluster_limit(
                primary=desired,
                fallback=cluster_candidates,
                topk=active_topk,
                cluster_map=cluster_map,
                max_per_cluster=LOSS_CLUSTER_MAX_PER_CLUSTER,
            )

    if STATE_CLUSTER_GUARD_ENABLE:
        dyn_cluster_map = active_state_group_map_for_date(
            tickers=cluster_candidates,
            clusters=STATE_CLUSTER_GROUPS,
            r_frame=r21,
            date=last_date,
            max_state=STATE_CLUSTER_R21_MAX,
        )
        cluster_need = (not STATE_CLUSTER_GUARD_ONLY_DUP) or has_group_duplicates(desired, dyn_cluster_map)
        if cluster_need and dyn_cluster_map:
            desired = apply_cluster_limit(
                primary=desired,
                fallback=cluster_candidates,
                topk=active_topk,
                cluster_map=dyn_cluster_map,
                max_per_cluster=STATE_CLUSTER_MAX_PER_CLUSTER,
            )

    held = sorted(list(positions.keys()))
    print(f"HELD: {held if held else '(none)'}")
    print(f"Desired: {desired if desired else '(none)'}")
    print(f"CorrGate: {corr_gate_hit}\n")
    if released_held:
        print(f"HeldRelease: {released_held}\n")

    # show weights and target € based on TOTAL equity (cash + positions),
    # and reserve fees only for missing BUYs (desired - held).
    total_equity_close = cash + total_pos
    w_dbg, targets_dbg, investable_eq_dbg, fees_reserved_dbg = pretty_weights_and_targets(
        total_equity=total_equity_close,
        cash=cash,
        desired=desired,
        held=held,
        vol_row=vol20.loc[last_date],
        fee_per_order=FEE_PER_ORDER,
        leader_ticker=(desired[0] if len(desired) > 0 else None),
    )

    if STATE_WEIGHT_CAP_ENABLE and w_dbg:
        dyn_weight_map = active_state_group_map_for_date(
            tickers=list(w_dbg.keys()),
            clusters=STATE_WEIGHT_GROUPS,
            r_frame=r21,
            date=last_date,
            max_state=STATE_WEIGHT_R21_MAX,
        )
        weight_need = (not STATE_WEIGHT_ONLY_DUP) or has_group_duplicates(desired, dyn_weight_map)
        if weight_need and dyn_weight_map:
            if STATE_WEIGHT_INDIV_CAP < 1.0:
                w_dbg = cap_ticker_weights(
                    weights=w_dbg,
                    capped_names=set(dyn_weight_map.keys()),
                    indiv_cap=STATE_WEIGHT_INDIV_CAP,
                )
            if STATE_WEIGHT_TOTAL_CAP < 1.0:
                w_dbg = cap_group_total_weight(
                    weights=w_dbg,
                    capped_names=set(dyn_weight_map.keys()),
                    total_cap=STATE_WEIGHT_TOTAL_CAP,
                )
            targets_dbg = {t: float(w_dbg.get(t, 0.0) * investable_eq_dbg) for t in desired}

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
        send_telegram(
            build_telegram_snapshot(
                header=header,
                cash=cash,
                pos_value=total_pos,
                held_before=held,
                desired=desired,
                ranked=ranked,
                srow=srow,
                corr_gate_hit=corr_gate_hit,
                weights=w_dbg,
                targets=targets_dbg,
                fees_reserved=fees_reserved_dbg,
                investable_equity=investable_eq_dbg,
                do_rebalance=False,
                orders=[],
                rebalance_days_since=rebalance_days_since,
                rebalance_interval=REB_EVERY_N_DAYS,
                last_rebalance_label=last_rebalance_label,
                held_released=released_held,
            )
        )
        return

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

    w = invvol_weights(
        vol20.loc[last_date],
        desired,
        leader_ticker=(desired[0] if len(desired) > 0 else None),
        leader_alpha=LEADER_ALPHA if LEADER_OVW_ENABLE else 0.0,
        leader_w_cap=LEADER_W_CAP if LEADER_OVW_ENABLE else 1.0,
    )
    targets_val = {t: w[t] * port_val_open for t in w if t in needed}

    orders: List[dict] = []

    pp_exit: List[str] = []
    if PP_ENABLE and positions:
        for t in list(positions.keys()):
            px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
            if not np.isfinite(px):
                continue
            st = ensure_pp_state(t, px, last_idx)
            st["peak"] = max(float(st.get("peak", px)), float(px))
            mfe = float(st["peak"]) / float(st["entry_px"]) - 1.0 if float(st["entry_px"]) > 0 else 0.0
            if (not bool(st.get("armed", False))) and mfe >= PP_MFE_TRIGGER:
                st["armed"] = True
                st["arm_idx"] = last_idx
            arm_idx = st.get("arm_idx", None)
            if bool(st.get("armed", False)) and arm_idx is not None and (last_idx - int(arm_idx)) >= PP_MIN_DAYS_AFTER_ARM:
                dd = 1.0 - float(px) / float(st["peak"]) if float(st["peak"]) > 0 else 0.0
                if dd >= PP_TRAIL_DD:
                    pp_exit.append(t)

    mie_exit: List[str] = []
    # MIE (Momentum Invalidation Exit): force exit when relative strength fails and the position is still weak.
    if MIE_ENABLE and positions:
        for t in list(positions.keys()):
            px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
            if not np.isfinite(px):
                continue
            st = pp_state.get(t)
            if st is None:
                continue
            entry_px = float(st.get("entry_px", px))
            entry_idx = int(st.get("entry_idx", last_idx))
            age = last_idx - entry_idx
            live_pnl = (px / entry_px) - 1.0 if entry_px > 0 else 0.0
            rs = safe_float(rs63.loc[last_date].get(t, np.nan))
            sma_val = safe_float(sma220.loc[last_date].get(t, np.nan))
            r5_val = safe_float(r5.loc[last_date].get(t, np.nan))
            rank_val = safe_float(ranks.loc[last_date].get(t, np.nan))
            dd60_val = safe_float(dd60.loc[last_date].get(t, np.nan))
            if bool(st.get("armed", False)):
                continue
            if age < MIE_MIN_HOLD_DAYS:
                continue
            if live_pnl > 0:
                continue
            if not np.isfinite(rs) or not np.isfinite(sma_val) or not np.isfinite(r5_val):
                continue
            if np.isfinite(rank_val) and rank_val <= MIE_EXEMPT_RANK_MAX:
                continue
            if (not np.isfinite(dd60_val)) or (dd60_val > -MIE_DD60_TH):
                continue
            if (rs <= MIE_RS63_TH) and (px < sma_val) and (r5_val < 0.0):
                mie_exit.append(t)

    # PP exits go first, then structural exits.
    for t in pp_exit:
        if t not in positions or t not in needed:
            continue
        pos_before = positions.get(t)
        sh = extract_shares(pos_before)
        if sh <= 0:
            continue
        sell_meta = build_sell_audit(pos_before, sh, float(px_open[t]), FEE_PER_ORDER)
        positions.pop(t, None)
        exit_defer.pop(t, None)
        pp_state.pop(t, None)
        entry_date.pop(t, None)
        if t in loss_names:
            loss_last_exit_idx[t] = int(last_idx)
        cash += sh * float(px_open[t]) - FEE_PER_ORDER
        orders.append({
            "Date": str(exec_date.date()),
            "Side": "SELL",
            "Ticker": t,
            "Shares": sh,
            "Price": float(px_open[t]),
            "Fee": FEE_PER_ORDER,
            "Reason": "PP_TRAIL",
            **sell_meta,
        })

    for t in mie_exit:
        if t not in positions or t not in needed:
            continue
        pos_before = positions.get(t)
        sh = extract_shares(pos_before)
        if sh <= 0:
            continue
        sell_meta = build_sell_audit(pos_before, sh, float(px_open[t]), FEE_PER_ORDER)
        positions.pop(t, None)
        exit_defer.pop(t, None)
        pp_state.pop(t, None)
        entry_date.pop(t, None)
        if t in loss_names:
            loss_last_exit_idx[t] = int(last_idx)
        cash += sh * float(px_open[t]) - FEE_PER_ORDER
        orders.append({
            "Date": str(exec_date.date()),
            "Side": "SELL",
            "Ticker": t,
            "Shares": sh,
            "Price": float(px_open[t]),
            "Fee": FEE_PER_ORDER,
            "Reason": "MOM_INV_RS63",
            **sell_meta,
        })

    if do_rebalance:
        for _t in list(exit_defer.keys()):
            if _t in targets_val:
                exit_defer.pop(_t, None)

        # Sell names not in target (deferred exit smoothing)
        for t in list(positions.keys()):
            if t in targets_val:
                continue

            pos_before = positions.get(t, 0.0)
            cur_sh = extract_shares(pos_before)
            if cur_sh <= 0:
                positions.pop(t, None)
                exit_defer.pop(t, None)
                pp_state.pop(t, None)
                entry_date.pop(t, None)
                continue

            if EXITSMOOTH_ENABLE:
                st = pp_state.get(t)
                if st is not None:
                    entry_px = float(st.get("entry_px", np.nan))
                    cl_px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
                    upnl = (cl_px / entry_px - 1.0) if (entry_px and entry_px > 0 and np.isfinite(cl_px)) else 0.0
                    ok = (upnl < 0.0)
                    if EXITSMOOTH_REQUIRE_TREND:
                        ok = ok and (cl_px > float(sma220.loc[last_date, t]))
                    if EXITSMOOTH_REQUIRE_POS_RS63:
                        ok = ok and (float(r63.loc[last_date, t]) > 0.0)
                    if ok and exit_defer.get(t, 0) < EXITSMOOTH_MAX_DEFERS:
                        exit_defer[t] = exit_defer.get(t, 0) + 1
                        continue

            if CONVEX_EXIT_GUARD_ENABLE and t in CONVEX_EXIT_GUARD_TICKERS:
                st = pp_state.get(t)
                if st is not None:
                    entry_px = float(st.get("entry_px", np.nan))
                    cl_px = safe_float(ohlcv.close_ffill.loc[last_date].get(t, np.nan))
                    upnl = (cl_px / entry_px - 1.0) if (entry_px and entry_px > 0 and np.isfinite(cl_px)) else 0.0
                    rs_now = safe_float(r63.loc[last_date].get(t, np.nan))
                    dd60_now = safe_float(dd60.loc[last_date].get(t, np.nan))
                    ok = (upnl < 0.0)
                    if CONVEX_EXIT_GUARD_DD60_TH > 0.0:
                        ok = ok and np.isfinite(dd60_now) and (dd60_now <= -CONVEX_EXIT_GUARD_DD60_TH)
                    if CONVEX_EXIT_GUARD_RS63_MIN > -1.0e8:
                        ok = ok and np.isfinite(rs_now) and (rs_now >= CONVEX_EXIT_GUARD_RS63_MIN)
                    if ok and exit_defer.get(t, 0) < CONVEX_EXIT_GUARD_MAX_DEFERS:
                        exit_defer[t] = exit_defer.get(t, 0) + 1
                        continue

            if t not in needed:
                continue  # cannot trade today (missing next open)

            cash += cur_sh * float(px_open[t]) - FEE_PER_ORDER
            sell_meta = build_sell_audit(pos_before, cur_sh, float(px_open[t]), FEE_PER_ORDER)
            positions.pop(t, None)
            exit_defer.pop(t, None)
            pp_state.pop(t, None)
            entry_date.pop(t, None)
            if t in loss_names:
                loss_last_exit_idx[t] = int(last_idx)
            orders.append({
                "Date": str(exec_date.date()),
                "Side": "SELL",
                "Ticker": t,
                "Shares": cur_sh,
                "Price": float(px_open[t]),
                "Fee": FEE_PER_ORDER,
                "Reason": "EXIT_NOT_TARGET",
                **sell_meta,
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
                pos_before = positions.get(t, 0.0)
                sell_sh = min((-diff) / price, cur_sh)
                cash += sell_sh * price - FEE_PER_ORDER
                sell_meta = build_sell_audit(pos_before, sell_sh, price, FEE_PER_ORDER)
                new_pos = reduce_position_after_sell(pos_before, sell_sh)
                if new_pos is None:
                    positions.pop(t, None)
                    pp_state.pop(t, None)
                    exit_defer.pop(t, None)
                    entry_date.pop(t, None)
                    if t in loss_names:
                        loss_last_exit_idx[t] = int(last_idx)
                else:
                    positions[t] = new_pos
                orders.append({
                    "Date": str(exec_date.date()),
                    "Side": "SELL",
                    "Ticker": t,
                    "Shares": sell_sh,
                    "Price": price,
                    "Fee": FEE_PER_ORDER,
                    "Reason": "DELTA_SELL",
                    **sell_meta,
                })

            elif diff > 0:
                max_buy_val = max(cash - FEE_PER_ORDER, 0.0)
                buy_val = min(diff, max_buy_val)
                if buy_val > 1e-8:
                    buy_sh = buy_val / price
                    cash -= buy_val + FEE_PER_ORDER
                    positions[t] = enrich_buy_position(positions.get(t, 0.0), buy_sh, price, exec_date)
                    new_total = extract_shares(positions[t])
                    if cur_sh <= 1e-12 and new_total > 1e-12:
                        entry_date[t] = str(exec_date.date())
                        if PP_ENABLE:
                            ensure_pp_state(t, price, last_idx)
                    port["last_buy_date"] = str(exec_date.date())
                    orders.append({
                        "Date": str(exec_date.date()),
                        "Side": "BUY",
                        "Ticker": t,
                        "Shares": buy_sh,
                        "Price": price,
                        "Fee": FEE_PER_ORDER,
                        "Reason": "DELTA_BUY",
                        "Amount": buy_val,
                    })



    port["cash"] = cash
    port["positions"] = positions
    port["entry_date"] = entry_date
    port["pp_state"] = pp_state
    port["exit_defer"] = exit_defer
    port["loss_last_exit_idx"] = loss_last_exit_idx
    if do_rebalance:
        port["last_rebalance_idx"] = int(last_idx)
        port["last_rebalance_date"] = str(last_date.date())
    save_portfolio(port)

    if orders:
        append_trades(orders)

    if not orders:
        print("ORDERS: none")
    else:
        print("ORDERS:")
        for o in orders:
            print(f" - {o['Side']} {o['Ticker']} sh={o['Shares']:.6f} @ {o['Price']:.2f} fee={o['Fee']:.2f} ({o['Reason']})")

    held_after = sorted(list(positions.keys()))
    send_telegram(
        build_telegram_snapshot(
            header=header,
            cash=cash,
            pos_value=pos_value_at_close(last_date),
            held_before=held,
            desired=desired,
            ranked=ranked,
            srow=srow,
            corr_gate_hit=corr_gate_hit,
            weights=w,
            targets=targets_val,
            fees_reserved=0.0,
            investable_equity=sum(targets_val.values()) if targets_val else 0.0,
            do_rebalance=True,
            orders=orders,
            rebalance_days_since=rebalance_days_since,
            rebalance_interval=REB_EVERY_N_DAYS,
            last_rebalance_label=last_rebalance_label,
            exec_date=exec_date,
            held_after=held_after,
            fallback_used=fallback_used,
            held_released=released_held,
        )
    )
    return

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
