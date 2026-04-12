from __future__ import annotations

"""APEX Research Engine — SMA220 canonical (CSV-driven)

Goal
----
Deterministic, reproducible backtest engine used as *single source of truth* for comparisons.

Data source
-----------
Uses the provided long-format CSV (date,ticker,open,close,...) from the project.
Default file name: apex_ohlcv_full_2015_2026.csv

Core logic (matches the SMA220 reference runs)
---------------------------------------------
- Signals computed on CLOSE of day t
- Execution on OPEN of day t+1 (T+1 open)
- Cross-sectional momentum score: 0.2*R63 + 0.4*R126 + 0.4*R252
- Trend filter: Close(t) > SMA220(t)
- Position sizing: inverse volatility (vol20) on Close(t)
- Rebalance: every 10 trading days, with delta-rebalance threshold
- Corr-guard: if max corr in rank pool > gate, pick diversified names
- DCA: +100€ on first trading day of each month (calendar anchored on SPY)
- Fees: 1€ per order (BUY/SELL)

Profit protection (PP)
----------------------
Optional trailing, armed after MFE >= trigger.
- When armed and min_days_after_arm elapsed, if drawdown from peak >= trail_dd -> exit next open.

This file is intended to be placed in the repo and executed locally.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Default parameters
# ----------------------------
DEFAULTS = dict(
    start="2015-01-02",
    end="2026-12-31",
    initial_cash=2000.0,
    monthly_dca=100.0,
    fee_per_order=1.0,
    sma_win=220,
    vol_win=20,
    w_r63=0.20,
    w_r126=0.40,
    w_r252=0.40,
    topk=3,
    rank_pool=15,
    keep_rank=5,
    rebalance_td=10,
    delta_rebal=0.10,
    corr_win=63,
    corr_gate=0.92,
    corr_pick=0.80,
    corr_scan=10,
    corr_held_enable=0,
    held_release_enable=0,
    held_release_loss_cluster_only=1,
    held_release_weak_rank_th=0,
    held_release_new_topm=0,
    held_release_min_new=0,
    loss_cluster_guard_enable=0,
    loss_cluster_max_per_cluster=1,
    loss_clusters=(),
    loss_entry_guard_enable=0,
    loss_entry_dist_th=0.0,
    loss_entry_dd60_th=0.0,
    late_entry_guard_enable=0,
    late_entry_dist_th=0.0,
    late_entry_dd60_th=0.0,
    late_entry_r21_th=0.0,
    late_entry_lt_topn=0,
    entry_heat_penalty_enable=0,
    entry_heat_penalty=0.0,
    entry_heat_dist_th=0.0,
    entry_heat_dd60_th=0.0,
    entry_heat_r21_th=0.0,
    entry_heat_lt_topn=0,
    entry_gap_penalty_enable=0,
    entry_gap_penalty=0.0,
    entry_gap_pct_th=0.0,
    entry_gap_dd60_th=0.0,
    entry_gap_r21_th=0.0,
    entry_gap_lt_topn=0,
    loss_reentry_cooldown_td=0,
    loss_weight_cap_enable=0,
    loss_weight_cap=1.0,
    loss_total_weight_cap_enable=0,
    loss_total_weight_cap=1.0,
    state_cluster_guard_enable=0,
    state_cluster_max_per_cluster=1,
    state_cluster_guard_only_dup=1,
    state_cluster_r21_max=0.0,
    state_cluster_lookback=21,
    state_cluster_groups=(),
    state_weight_cap_enable=0,
    state_weight_total_cap=1.0,
    state_weight_indiv_cap=1.0,
    state_weight_only_dup=1,
    state_weight_r21_max=0.0,
    state_weight_lookback=21,
    state_weight_groups=(),
    regime_enable=0,
    regime_spy_sma_filter=0,
    regime_breadth_th=0.0,
    regime_weak_topk=3,
    buy_slippage_bps=0.0,
    sell_slippage_bps=0.0,
    min_bars_required=260,
    leader_ovw_enable=0,
    leader_alpha=0.25,
    leader_persist_win=10,
    leader_topk=3,
    leader_persist_min=0.30,
    leader_w_cap=0.55,
    reentry_enable=0,
    reentry_lt_topn=12,
    reentry_dd60_min=0.08,
    reentry_dd60_max=0.35,
    reentry_r21_max=0.05,
    reentry_bonus=0.10,
    slot3_gate_enable=0,
    slot3_max_rank=4,
    slot3_weak_max_rank=0,
    slot3_leader_exempt_topn=12,
    slot3_weight_cap_enable=0,
    slot3_weight_cap=1.0,
    slot3_weight_cap_weak_only=0,
    quality_filter_enable=0,
    q_slot2_enable=0,
    q_slot3_enable=1,
    q_persist_win=5,
    q_slot2_rank_th=5,
    q_slot3_rank_th=6,
    q_min_count=3,
    q_leader_exempt_topn=12,
    q_require_pos_rs63=0,
    pp_leader_partial_enable=0,
    pp_leader_rank_th=2,
    pp_leader_rs63_th=0.50,
    pp_leader_sell_frac=0.50,
    pp_leader_partial_max=1,
    pp_leader_lt_topn=0,
    mie_exempt_rank_max=0,
    mie_exempt_lt_topn=0,
    mie_dd60_th=0.0,
    score_rs_blend=0.0,
    score_breakout_252_weight=0.0,
    score_breakout_252_maxdd=0.35,
    score_residual_blend=0.0,
    score_residual_clip=5.0,
    score_quality_blend=0.0,
    score_quality_win=63,
)



@dataclass
class Prices:
    open: pd.DataFrame
    close: pd.DataFrame


def _prices_cache(prices: Prices) -> Dict[str, object]:
    cache = getattr(prices, "_backtest_cache", None)
    if cache is None:
        cache = {}
        setattr(prices, "_backtest_cache", cache)
    return cache


def _cached_value(prices: Prices, key: str, factory):
    cache = _prices_cache(prices)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def load_prices_from_csv(path: str) -> Prices:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = ["date", "ticker", "open", "close"]
    feature_cols = [c for c in ("residual_momentum_score",) if c in cols]
    usecols = usecols + feature_cols
    df = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"])

    op = df.pivot(index="date", columns="ticker", values="open").sort_index().ffill()
    cl = df.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    prices = Prices(open=op, close=cl)
    if feature_cols:
        features = {}
        for col in feature_cols:
            features[col] = df.pivot(index="date", columns="ticker", values=col).sort_index().ffill()
        prices.features = features
    return prices


def corr_matrix(window_returns: np.ndarray) -> np.ndarray:
    m = window_returns.astype(float)
    m = m - np.nanmean(m, axis=0, keepdims=True)
    s = np.nanstd(m, axis=0, keepdims=True) + 1e-12
    m = m / s
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (m.T @ m) / max(m.shape[0] - 1, 1)
    return np.clip(corr, -1.0, 1.0)


def corr_cap_pick(
    ranked: List[str],
    win_slice: np.ndarray,
    topk: int,
    thr: float,
    max_scan: int,
    held: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
) -> List[str]:
    """Correlation-capped picker anchored on current holdings.

    ``tickers`` must match the column order of ``win_slice`` so the guard can
    evaluate fresh candidates against currently held names even if those held
    names are outside the new rank pool.
    """
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


def build_cluster_map(clusters) -> Dict[str, int]:
    cluster_map: Dict[str, int] = {}
    for idx, cluster in enumerate(clusters or []):
        for ticker in cluster:
            cluster_map[ticker] = idx
    return cluster_map


def flatten_clusters(clusters) -> set:
    names = set()
    for cluster in clusters or []:
        for ticker in cluster:
            names.add(ticker)
    return names


def build_group_state(close_px: pd.DataFrame, group_map: Dict[str, int], lookback: int) -> pd.DataFrame:
    if not group_map:
        return pd.DataFrame(index=close_px.index)
    rets = close_px / close_px.shift(lookback) - 1.0
    members: Dict[int, List[str]] = {}
    for ticker, group_id in group_map.items():
        if ticker not in rets.columns:
            continue
        members.setdefault(group_id, []).append(ticker)
    data: Dict[int, pd.Series] = {}
    for group_id, tickers in members.items():
        data[group_id] = rets[tickers].mean(axis=1)
    if not data:
        return pd.DataFrame(index=close_px.index)
    return pd.DataFrame(data, index=close_px.index)


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


def active_state_group_map(
    tickers: List[str],
    group_map: Dict[str, int],
    state_df: pd.DataFrame,
    date: pd.Timestamp,
    max_state: float,
) -> Dict[str, int]:
    if not group_map or state_df.empty or date not in state_df.index:
        return {}
    row = state_df.loc[date]
    active: Dict[str, int] = {}
    for ticker in tickers:
        group_id = group_map.get(ticker)
        if group_id is None or group_id not in row.index:
            continue
        state = row[group_id]
        if pd.notna(state) and float(state) <= max_state:
            active[ticker] = group_id
    return active


def apply_held_release_rule(
    held: List[str],
    ranked: List[str],
    loss_names: set,
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
    selected: List[str] = []
    cluster_counts: Dict[int, int] = {}
    order = list(primary) + [t for t in fallback if t not in primary]

    for ticker in order:
        if ticker in selected:
            continue
        cluster_id = cluster_map.get(ticker)
        if cluster_id is not None and cluster_counts.get(cluster_id, 0) >= max_per_cluster:
            continue
        selected.append(ticker)
        if cluster_id is not None:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        if len(selected) >= topk:
            break

    return selected[:topk]


def cap_ticker_weights(
    weights: Dict[str, float],
    capped_names: set,
    indiv_cap: float,
) -> Dict[str, float]:
    if not weights or not capped_names:
        return weights
    cap = float(indiv_cap)
    if cap <= 0.0:
        return weights
    capped_present = [k for k in weights.keys() if k in capped_names]
    if not capped_present:
        return weights
    # Feasibility only fails if every name is capped and the basket cannot sum to 1.
    if len(capped_present) == len(weights) and cap * len(weights) < 1.0 - 1e-12:
        return weights

    out = dict(weights)
    over = {k: v for k, v in out.items() if k in capped_names and v > cap}
    if not over:
        return out

    excess = sum(v - cap for v in over.values())
    for k in over:
        out[k] = cap

    receivers = [k for k in out.keys() if k not in over]
    recv_sum = sum(out[k] for k in receivers)
    if recv_sum > 0 and excess > 0:
        for k in receivers:
            out[k] += excess * (out[k] / recv_sum)

    ssum = float(sum(out.values()))
    if ssum > 0:
        for k in out:
            out[k] /= ssum
    return out


def cap_group_total_weight(
    weights: Dict[str, float],
    capped_names: set,
    total_cap: float,
) -> Dict[str, float]:
    if not weights or not capped_names:
        return weights
    cap = float(total_cap)
    if cap <= 0.0 or cap >= 1.0:
        return weights

    out = dict(weights)
    group = [k for k in out.keys() if k in capped_names]
    if not group:
        return out

    group_sum = sum(out[k] for k in group)
    if group_sum <= cap + 1e-12:
        return out

    scale = cap / group_sum
    removed = 0.0
    for k in group:
        old = out[k]
        out[k] = old * scale
        removed += old - out[k]

    receivers = [k for k in out.keys() if k not in group]
    recv_sum = sum(out[k] for k in receivers)
    if recv_sum > 0 and removed > 0:
        for k in receivers:
            out[k] += removed * (out[k] / recv_sum)

    ssum = float(sum(out.values()))
    if ssum > 0:
        for k in out:
            out[k] /= ssum
    return out


def invvol_weights(vol_row: pd.Series, tickers: List[str]) -> Dict[str, float]:
    v = vol_row.reindex(tickers).replace(0, np.nan)
    inv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
    if inv.empty:
        return {}
    inv = inv / inv.sum()
    return {t: float(inv.loc[t]) for t in inv.index}


def first_trading_day_each_month(calendar: pd.DatetimeIndex) -> Dict[Tuple[int, int], pd.Timestamp]:
    out: Dict[Tuple[int, int], pd.Timestamp] = {}
    for d in calendar:
        key = (d.year, d.month)
        if key not in out:
            out[key] = d
    return out


def _candidate_quality_ok(ranks, rank126, rank252, rs63, d, cand, slot_num, p):
    if int(p.get("quality_filter_enable", 0)) != 1:
        return True
    if slot_num == 2 and int(p.get("q_slot2_enable", 0)) != 1:
        return True
    if slot_num == 3 and int(p.get("q_slot3_enable", 0)) != 1:
        return True
    lt_n = int(p.get("q_leader_exempt_topn", 12))
    if (pd.notna(rank126.loc[d, cand]) and pd.notna(rank252.loc[d, cand]) and
        int(rank126.loc[d, cand]) <= lt_n and int(rank252.loc[d, cand]) <= lt_n):
        return True
    win = int(p.get("q_persist_win", 5))
    rank_th = int(p.get("q_slot2_rank_th", 5) if slot_num == 2 else p.get("q_slot3_rank_th", 6))
    try:
        end_loc = ranks.index.get_loc(d)
    except KeyError:
        return True
    start_loc = max(0, end_loc - win + 1)
    wdates = ranks.index[start_loc:end_loc+1]
    rr = ranks.loc[wdates, cand]
    count = int((rr <= rank_th).sum())
    if count < int(p.get("q_min_count", 3)):
        return False
    if int(p.get("q_require_pos_rs63", 0)) == 1:
        try:
            if float(rs63.loc[d, cand]) <= 0.0:
                return False
        except Exception:
            return False
    return True


def _convex_nonpp_guard_ok(
    *,
    ticker: str,
    st: dict | None,
    close_px: float,
    d,
    rs63,
    dd60,
    rank126,
    rank252,
    p,
    prefix: str,
) -> bool:
    if int(p.get(f"{prefix}_enable", 0)) != 1:
        return False
    names = set(p.get(f"{prefix}_tickers", ()))
    if ticker not in names or st is None:
        return False
    entry_px = float(st.get("entry_px", np.nan))
    if not np.isfinite(entry_px) or entry_px <= 0 or not np.isfinite(close_px):
        return False
    upnl = close_px / entry_px - 1.0
    if upnl >= 0.0:
        return False

    dd_th = float(p.get(f"{prefix}_dd60_th", 0.0))
    if dd_th > 0.0:
        dd_now = float(dd60.loc[d, ticker]) if pd.notna(dd60.loc[d, ticker]) else np.nan
        if (not np.isfinite(dd_now)) or dd_now > -dd_th:
            return False

    rs_min = float(p.get(f"{prefix}_rs63_min", -1.0e9))
    rs_now = float(rs63.loc[d, ticker]) if pd.notna(rs63.loc[d, ticker]) else np.nan

    lt_topn = int(p.get(f"{prefix}_lt_topn", 0))
    lt_ok = False
    if lt_topn > 0:
        r126_now = float(rank126.loc[d, ticker]) if pd.notna(rank126.loc[d, ticker]) else np.nan
        r252_now = float(rank252.loc[d, ticker]) if pd.notna(rank252.loc[d, ticker]) else np.nan
        lt_ok = (
            np.isfinite(r126_now)
            and np.isfinite(r252_now)
            and r126_now <= lt_topn
            and r252_now <= lt_topn
        )

    if rs_min <= -1.0e8:
        return True
    return (np.isfinite(rs_now) and rs_now >= rs_min) or lt_ok



def run_backtest(
    prices: Prices,
    start: str,
    end: str,
    *,
    pp_enabled: bool,
    mie_enabled: bool = False,
    mie_rs_th: float = 0.03,
    mie_min_hold: int = 9,
     doom_enabled: bool = False,
     doom_dd60_th: float = 0.15,
     doom_rs_th: float = 0.03,
     doom_min_hold: int = 9,
     doom_pick_mode: str = 'score',
     persist_win: int = 10,
     persist_topk: int = 5,
    pp_mfe_trigger: float,
    pp_trail_dd: float,
    pp_min_days_after_arm: int,
    exit_smooth_enable: int = 0,
    exit_smooth_max_defers: int = 1,
    exit_smooth_require_trend: int = 1,
    exit_smooth_require_pos_rs63: int = 0,
    **p,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:

    c = prices.close
    o = prices.open
    features = getattr(prices, "features", {})
    cache = _prices_cache(prices)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = c.index[(c.index >= start_ts) & (c.index <= end_ts)]
    if dates.empty:
        raise RuntimeError("No dates in range.")

    # Signals on CLOSE
    ret1 = _cached_value(prices, "ret1", lambda: c.pct_change(fill_method=None))
    r63 = _cached_value(prices, "r63", lambda: c / c.shift(63) - 1.0)
    r126 = _cached_value(prices, "r126", lambda: c / c.shift(126) - 1.0)
    r252 = _cached_value(prices, "r252", lambda: c / c.shift(252) - 1.0)
    r21 = _cached_value(prices, "r21", lambda: c / c.shift(21) - 1.0)
    r5 = _cached_value(prices, "r5", lambda: c / c.shift(5) - 1.0)
    high60 = _cached_value(prices, "high60", lambda: c.rolling(60, min_periods=60).max())
    dd60 = _cached_value(prices, "dd60", lambda: c / high60 - 1.0)
    high252 = _cached_value(prices, "high252", lambda: c.rolling(252, min_periods=252).max())
    dd252 = _cached_value(prices, "dd252", lambda: c / high252 - 1.0)

    # Short-term trend + short-term drawdown (for conditional over-extension guard)
    sma_short_win = int(p.get("oeg2_sma_win", 20))
    sma_short = _cached_value(
        prices,
        f"sma_short:{sma_short_win}",
        lambda: c.rolling(sma_short_win, min_periods=sma_short_win).mean(),
    )
    dd_win = int(p.get('oeg2_dd_win', 10))
    highN = _cached_value(prices, f"highN:{dd_win}", lambda: c.rolling(dd_win, min_periods=dd_win).max())
    ddN = _cached_value(prices, f"ddN:{dd_win}", lambda: c / highN - 1.0)
    # Relative strength vs SPY over 63d (opportunity proxy)
    if "SPY" in c.columns and c["SPY"].notna().any():
        spy_r63 = _cached_value(prices, "spy_r63", lambda: c["SPY"] / c["SPY"].shift(63) - 1.0)
        spy_r126 = _cached_value(prices, "spy_r126", lambda: c["SPY"] / c["SPY"].shift(126) - 1.0)
        spy_r252 = _cached_value(prices, "spy_r252", lambda: c["SPY"] / c["SPY"].shift(252) - 1.0)
        rs63 = _cached_value(prices, "rs63", lambda: (1.0 + r63).div(1.0 + spy_r63, axis=0) - 1.0)
        rs126 = _cached_value(prices, "rs126", lambda: (1.0 + r126).div(1.0 + spy_r126, axis=0) - 1.0)
        rs252 = _cached_value(prices, "rs252", lambda: (1.0 + r252).div(1.0 + spy_r252, axis=0) - 1.0)
        beta63 = _cached_value(prices, "beta63", lambda: ret1.rolling(63, min_periods=40).cov(ret1["SPY"]))
        residual_base_63 = _cached_value(prices, "residual_base_63", lambda: r63 - beta63.mul(spy_r63, axis=0))
        residual_base_126 = _cached_value(prices, "residual_base_126", lambda: r126 - beta63.mul(spy_r126, axis=0))
        residual_base_252 = _cached_value(prices, "residual_base_252", lambda: r252 - beta63.mul(spy_r252, axis=0))
        residual_score = (
            p["w_r63"] * residual_base_63
            + p["w_r126"] * residual_base_126
            + p["w_r252"] * residual_base_252
        )
    else:
        rs63 = r63 * 0.0  # fallback: neutral
        rs126 = r126 * 0.0
        rs252 = r252 * 0.0
        beta63 = r63 * 0.0
        residual_score = r63 * 0.0

    score = p["w_r63"] * r63 + p["w_r126"] * r126 + p["w_r252"] * r252
    rs_blend = float(p.get("score_rs_blend", 0.0))
    if rs_blend != 0.0:
        rel_score = p["w_r63"] * rs63 + p["w_r126"] * rs126 + p["w_r252"] * rs252
        score = score + rs_blend * rel_score
    residual_blend = float(p.get("score_residual_blend", 0.0))
    if residual_blend != 0.0:
        residual_clip = float(p.get("score_residual_clip", 5.0))
        residual_feature = features.get("residual_momentum_score")
        if isinstance(residual_feature, pd.DataFrame):
            residual_score = residual_feature.reindex(index=c.index, columns=c.columns)
        score = score + residual_blend * residual_score.clip(lower=-residual_clip, upper=residual_clip)
    breakout_w = float(p.get("score_breakout_252_weight", 0.0))
    if breakout_w != 0.0:
        breakout_maxdd = max(1e-9, float(p.get("score_breakout_252_maxdd", 0.35)))
        breakout_252 = (1.0 + (dd252 / breakout_maxdd)).clip(lower=0.0, upper=1.0)
        score = score + breakout_w * breakout_252
    quality_blend = float(p.get("score_quality_blend", 0.0))
    if quality_blend != 0.0:
        quality_win = max(5, int(p.get("score_quality_win", 63)))
        path = c.diff().abs().rolling(quality_win, min_periods=quality_win).sum()
        eff = c.diff(quality_win).abs() / (path + 1e-12)
        score = score + quality_blend * eff.clip(lower=0.0, upper=1.0)
    sma_win = int(p["sma_win"])
    vol_win = int(p["vol_win"])
    sma = _cached_value(prices, f"sma:{sma_win}", lambda: c.rolling(sma_win, min_periods=sma_win).mean())
    vol = _cached_value(prices, f"vol:{vol_win}", lambda: ret1.rolling(vol_win, min_periods=vol_win).std())
    atrp20 = _cached_value(prices, "atrp20", lambda: ret1.abs().rolling(20, min_periods=20).mean())
    vol60 = _cached_value(prices, "vol60", lambda: ret1.rolling(60, min_periods=60).std())
    enough = _cached_value(prices, f"enough:{int(p['min_bars_required'])}", lambda: (c.notna().sum() >= p["min_bars_required"]))
    prev_close = _cached_value(prices, "prev_close", lambda: c.shift(1))
    dist220 = _cached_value(prices, f"dist220:{sma_win}", lambda: (c / (sma + 1e-12)) - 1.0)
    gap_open = _cached_value(prices, "gap_open", lambda: (o / (prev_close + 1e-12)) - 1.0)
    breadth = cache.get(f"breadth:{sma_win}:{int(p['min_bars_required'])}")
    if breadth is None:
        valid_mask = enough.reindex(c.columns, fill_value=False)
        valid_frame = valid_mask.values.reshape(1, -1) & c.notna().values & sma.notna().values
        breadth_values = np.divide(
            ((c.values > sma.values) & valid_frame).sum(axis=1),
            valid_frame.sum(axis=1),
            out=np.full(c.shape[0], np.nan, dtype=float),
            where=valid_frame.sum(axis=1) > 0,
        )
        breadth = pd.Series(breadth_values, index=c.index)
        cache[f"breadth:{sma_win}:{int(p['min_bars_required'])}"] = breadth
    rank126 = _cached_value(prices, "rank126", lambda: r126.rank(axis=1, ascending=False, method='min'))
    rank252 = _cached_value(prices, "rank252", lambda: r252.rank(axis=1, ascending=False, method='min'))
    leader_lt = (rank126 <= int(p.get('reentry_lt_topn', 12))) & (rank252 <= int(p.get('reentry_lt_topn', 12)))
    pullback_lt = (dd60 <= -float(p.get('reentry_dd60_min', 0.08))) & (dd60 >= -float(p.get('reentry_dd60_max', 0.35))) & (r21 <= float(p.get('reentry_r21_max', 0.05)))
    score_eff = score.copy()
    if int(p.get('reentry_enable', 0)) == 1:
        reentry_mask = leader_lt & pullback_lt
        breadth_min = float(p.get("reentry_breadth_min", 0.0) or 0.0)
        if breadth_min > 0.0:
            reentry_mask = reentry_mask & (breadth.to_numpy()[:, None] >= breadth_min)
        rs63_gate = p.get("reentry_rs63_min", None)
        if rs63_gate is not None:
            reentry_mask = reentry_mask & (rs63 >= float(rs63_gate))
        score_eff = score_eff + float(p.get('reentry_bonus', 0.10)) * reentry_mask.astype(float)
    ranks = score.rank(axis=1, ascending=False, method='min')

    state_cluster_map: Dict[str, int] = {}
    state_cluster_state = pd.DataFrame(index=c.index)
    if int(p.get("state_cluster_guard_enable", 0)) == 1:
        state_cluster_groups = p.get("state_cluster_groups", ())
        state_cluster_map = build_cluster_map(state_cluster_groups)
        if state_cluster_map:
            lookback = max(1, int(p.get("state_cluster_lookback", 21)))
            group_sig = tuple(tuple(str(x) for x in cluster) for cluster in (state_cluster_groups or ()))
            cache_key = f"state_cluster_state:{lookback}:{group_sig}"
            state_cluster_state = _cached_value(
                prices,
                cache_key,
                lambda: build_group_state(c, state_cluster_map, lookback),
            )
    state_weight_map: Dict[str, int] = {}
    state_weight_state = pd.DataFrame(index=c.index)
    if int(p.get("state_weight_cap_enable", 0)) == 1:
        state_weight_groups = p.get("state_weight_groups", ())
        state_weight_map = build_cluster_map(state_weight_groups)
        if state_weight_map:
            lookback = max(1, int(p.get("state_weight_lookback", 21)))
            group_sig = tuple(tuple(str(x) for x in cluster) for cluster in (state_weight_groups or ()))
            cache_key = f"state_weight_state:{lookback}:{group_sig}"
            state_weight_state = _cached_value(
                prices,
                cache_key,
                lambda: build_group_state(c, state_weight_map, lookback),
            )

    idx_map = {d: i for i, d in enumerate(c.index)}
    buy_slippage = float(p.get("buy_slippage_bps", 0.0)) / 10000.0
    sell_slippage = float(p.get("sell_slippage_bps", 0.0)) / 10000.0

    def buy_exec_price(raw_price: float) -> float:
        return float(raw_price) * (1.0 + buy_slippage)

    def sell_exec_price(raw_price: float) -> float:
        return float(raw_price) * (1.0 - sell_slippage)

    # Calendar anchor (SPY if present)
    if "SPY" in c.columns and c["SPY"].notna().any():
        cal = c.index[c["SPY"].notna()]
    else:
        cal = c.index
    month_first = first_trading_day_each_month(cal)

    cash = float(p["initial_cash"])
    invested = float(p["initial_cash"])
    positions: Dict[str, float] = {}
    exit_defer: Dict[str, int] = {}
    loss_names = flatten_clusters(p.get("loss_clusters", ()))
    loss_last_exit_idx: Dict[str, int] = {}

    # PP state
    pp_state: Dict[str, dict] = {}

    # Leader persistence history (TopK names per day)
    leader_hist = deque(maxlen=int(p.get('leader_persist_win', 10)))

    trades_rows = []
    equity_rows = []

    for d in dates:
        # DCA
        key = (d.year, d.month)
        if key in month_first and month_first[key] == d:
            cash += float(p["monthly_dca"])
            invested += float(p["monthly_dca"])

        # Mark-to-market at close
        pos_val = 0.0
        for t, sh in positions.items():
            px = c.loc[d, t]
            if pd.notna(px):
                pos_val += float(sh) * float(px)
        equity_rows.append({"date": d, "equity": cash + pos_val})

        i = idx_map[d]
        do_rebalance = (i % int(p["rebalance_td"]) == 0)
        doom_exit: List[str] = []
        doom_buy: Optional[str] = None

        # Eligible set
        elig = (c.loc[d] > sma.loc[d]) & score.loc[d].notna() & vol.loc[d].notna()
        elig = elig & enough.reindex(c.columns, fill_value=False)

        # Tail-risk veto (selection-level): exclude names that look like "tail + turning"
        if int(p.get('tailveto_enable', 0)) == 1:
            atr_th = float(p.get('tailveto_atrp_th', 0.055))
            spike_th = float(p.get('tailveto_volspike_th', 1.3))
            dd_th = float(p.get('tailveto_dd60_th', 0.15))
            tail = (atrp20.loc[d] >= atr_th) & ((vol.loc[d] / (vol60.loc[d] + 1e-12)) >= spike_th)
            rs = p.get('tailveto_risk_set', None)
            if rs is not None and len(rs) > 0:
                mask = pd.Index(c.columns).isin(list(rs))
                tail = tail & pd.Series(mask, index=c.columns)
            elig = elig & (~tail.fillna(False))

        # OEG2 (conditional over-extension guard) — avoid late-parabola entries without killing convexity
        if int(p.get('oeg2_enable', 0)) == 1:
            dist_th = float(p.get('oeg2_dist_th', 0.90))
            mode = str(p.get('oeg2_mode', 'sma')).lower()  # 'sma' or 'dd'
            # dist vs SMA220
            dist = dist220.loc[d]
            cond = dist >= dist_th
            # scope: risk_set only by default
            rs2 = p.get('tailveto_risk_set', None)
            if rs2 is not None and len(rs2) > 0:
                mask2 = pd.Index(c.columns).isin(list(rs2))
                cond = cond & pd.Series(mask2, index=c.columns)
            if mode == 'sma':
                smaw = int(p.get('oeg2_sma_win', 20))
                # recompute short SMA only if window differs from cached sma_short
                if smaw != int(getattr(run_backtest, '_oeg2_last_smaw', -1)):
                    pass
                # use precomputed sma_short (based on provided oeg2_sma_win)
                cond = cond & (c.loc[d] < sma_short.loc[d])
            else:
                dd_th = float(p.get('oeg2_dd_th', 0.08))
                cond = cond & (ddN.loc[d] <= -dd_th)
            elig = elig & (~cond.fillna(False))

        if loss_names:
            held_names = set(positions.keys())
            loss_candidate_mask = pd.Series(
                pd.Index(c.columns).isin(list(loss_names)),
                index=c.columns,
            ) & (~pd.Series(pd.Index(c.columns).isin(list(held_names)), index=c.columns))

            cooldown_td = int(p.get("loss_reentry_cooldown_td", 0))
            if cooldown_td > 0 and loss_last_exit_idx:
                cooldown_mask = pd.Series(False, index=c.columns)
                for ticker, exit_idx in loss_last_exit_idx.items():
                    if ticker in cooldown_mask.index and (i - int(exit_idx)) < cooldown_td:
                        cooldown_mask.loc[ticker] = True
                elig = elig & (~(loss_candidate_mask & cooldown_mask).fillna(False))

            if int(p.get("loss_entry_guard_enable", 0)) == 1:
                dist_th = float(p.get("loss_entry_dist_th", 0.0))
                dd_th = float(p.get("loss_entry_dd60_th", 0.0))
                loss_guard = loss_candidate_mask.copy()
                if dist_th > 0.0:
                    loss_guard = loss_guard & (dist220.loc[d] >= dist_th)
                if dd_th > 0.0:
                    loss_guard = loss_guard & (dd60.loc[d] >= -dd_th)
                elig = elig & (~loss_guard.fillna(False))

        if int(p.get("late_entry_guard_enable", 0)) == 1:
            held_names = set(positions.keys())
            candidate_mask = ~pd.Series(pd.Index(c.columns).isin(list(held_names)), index=c.columns)
            dist_th = float(p.get("late_entry_dist_th", 0.0))
            dd_th = float(p.get("late_entry_dd60_th", 0.0))
            r21_th = float(p.get("late_entry_r21_th", 0.0))
            lt_topn = int(p.get("late_entry_lt_topn", 0))

            late_guard = candidate_mask.copy()
            if lt_topn > 0:
                late_guard = late_guard & (rank126.loc[d] <= lt_topn) & (rank252.loc[d] <= lt_topn)
            if dist_th > 0.0:
                late_guard = late_guard & (dist220.loc[d] >= dist_th)
            if dd_th > 0.0:
                late_guard = late_guard & (dd60.loc[d] >= -dd_th)
            if r21_th > 0.0:
                late_guard = late_guard & (r21.loc[d] >= r21_th)
            elig = elig & (~late_guard.fillna(False))

        score_rank = score_eff.loc[d].copy()
        if int(p.get("entry_heat_penalty_enable", 0)) == 1:
            held_names = set(positions.keys())
            candidate_mask = ~pd.Series(pd.Index(c.columns).isin(list(held_names)), index=c.columns)
            penalty = float(p.get("entry_heat_penalty", 0.0))
            dist_th = float(p.get("entry_heat_dist_th", 0.0))
            dd_th = float(p.get("entry_heat_dd60_th", 0.0))
            r21_th = float(p.get("entry_heat_r21_th", 0.0))
            lt_topn = int(p.get("entry_heat_lt_topn", 0))

            hot_mask = candidate_mask.copy()
            if lt_topn > 0:
                hot_mask = hot_mask & (rank126.loc[d] <= lt_topn) & (rank252.loc[d] <= lt_topn)
            if dist_th > 0.0:
                hot_mask = hot_mask & (dist220.loc[d] >= dist_th)
            if dd_th > 0.0:
                hot_mask = hot_mask & (dd60.loc[d] >= -dd_th)
            if r21_th > 0.0:
                hot_mask = hot_mask & (r21.loc[d] >= r21_th)
            score_rank = score_rank - penalty * hot_mask.astype(float)

        if int(p.get("entry_gap_penalty_enable", 0)) == 1:
            held_names = set(positions.keys())
            candidate_mask = ~pd.Series(pd.Index(c.columns).isin(list(held_names)), index=c.columns)
            penalty = float(p.get("entry_gap_penalty", 0.0))
            gap_th = float(p.get("entry_gap_pct_th", 0.0))
            dd_th = float(p.get("entry_gap_dd60_th", 0.0))
            r21_th = float(p.get("entry_gap_r21_th", 0.0))
            lt_topn = int(p.get("entry_gap_lt_topn", 0))
            gap_now = gap_open.loc[d]

            gap_mask = candidate_mask.copy()
            if lt_topn > 0:
                gap_mask = gap_mask & (rank126.loc[d] <= lt_topn) & (rank252.loc[d] <= lt_topn)
            if gap_th > 0.0:
                gap_mask = gap_mask & (gap_now >= gap_th)
            if dd_th > 0.0:
                gap_mask = gap_mask & (dd60.loc[d] >= -dd_th)
            if r21_th > 0.0:
                gap_mask = gap_mask & (r21.loc[d] >= r21_th)
            score_rank = score_rank - penalty * gap_mask.astype(float)

        srow_full = score_rank.where(elig, np.nan).dropna()
        ranked_full = list(srow_full.sort_values(ascending=False).index)
        ranked = ranked_full[: int(p["rank_pool"]) ]
        # Update leader persistence history using full TopK ranks (before corr-guard)
        leader_hist.append(set(ranked_full[: int(p.get('leader_topk', p.get('topk', 3))) ]))

        active_topk = int(p["topk"])
        regime_weak = False
        if int(p.get("regime_enable", 0)) == 1:
            regime_flags = []
            if int(p.get("regime_spy_sma_filter", 0)) == 1 and "SPY" in c.columns:
                spy_close = float(c.loc[d, "SPY"]) if pd.notna(c.loc[d, "SPY"]) else np.nan
                spy_sma = float(sma.loc[d, "SPY"]) if pd.notna(sma.loc[d, "SPY"]) else np.nan
                if np.isfinite(spy_close) and np.isfinite(spy_sma):
                    regime_flags.append(spy_close <= spy_sma)
            if int(p.get("regime_mktvol_enable", 0)) == 1 and "SPY" in ret1.columns:
                regime_vol_win = int(p.get("regime_mktvol_win", 20))
                if regime_vol_win > 1:
                    spy_vol = _cached_value(
                        prices,
                        f"spy_vol:{regime_vol_win}",
                        lambda: ret1["SPY"].rolling(regime_vol_win, min_periods=regime_vol_win).std() * np.sqrt(252.0),
                    )
                    spy_vol_now = float(spy_vol.loc[d]) if pd.notna(spy_vol.loc[d]) else np.nan
                    if np.isfinite(spy_vol_now):
                        regime_flags.append(spy_vol_now >= float(p.get("regime_mktvol_th", 0.35)))
            breadth_th = float(p.get("regime_breadth_th", 0.0))
            if breadth_th > 0.0:
                breadth_now = float(breadth.loc[d]) if pd.notna(breadth.loc[d]) else np.nan
                if np.isfinite(breadth_now):
                    regime_flags.append(breadth_now < breadth_th)
            if regime_flags:
                if int(p.get("regime_confirm_all", 0)) == 1:
                    regime_weak = all(regime_flags)
                else:
                    regime_weak = any(regime_flags)
            if regime_weak:
                active_topk = max(1, min(int(p.get("regime_weak_topk", p["topk"])), int(p["topk"])))

        desired_ranked = ranked[:active_topk]

        # Corr guard
        released_held: List[str] = []
        if len(ranked) >= 2:
            w0 = max(0, i - int(p["corr_win"]) + 1)
            held = [t for t in positions.keys() if t in ret1.columns]
            held_for_corr = list(held)
            if int(p.get("held_release_enable", 0)) == 1:
                held_for_corr, released_held = apply_held_release_rule(
                    held=held,
                    ranked=ranked,
                    loss_names=loss_names,
                    weak_rank_th=int(p.get("held_release_weak_rank_th", 0)),
                    newcomer_topm=int(p.get("held_release_new_topm", 0)),
                    min_new=int(p.get("held_release_min_new", 0)),
                    loss_only=bool(int(p.get("held_release_loss_cluster_only", 1))),
                )
            if int(p.get("corr_held_enable", 0)) == 1:
                corr_names = list(dict.fromkeys(held_for_corr + ranked))
            else:
                corr_names = list(ranked)
            win_slice = ret1.iloc[w0 : i + 1][corr_names].to_numpy()
            if win_slice.shape[0] >= int(0.8 * int(p["corr_win"])):
                corr = corr_matrix(win_slice)
                max_corr = float(np.nanmax(np.where(np.eye(corr.shape[0]), np.nan, corr)))
                if np.isfinite(max_corr) and max_corr > float(p["corr_gate"]):
                    desired_ranked = corr_cap_pick(
                        ranked=ranked,
                        win_slice=win_slice,
                        topk=int(p["topk"]),
                        thr=float(p["corr_pick"]),
                        max_scan=int(p["corr_scan"]),
                        held=held_for_corr if int(p.get("corr_held_enable", 0)) == 1 else None,
                        tickers=corr_names,
                    )

        desired = apply_keep_rank(
            current=list(positions.keys()),
            ranked=desired_ranked,
            topk=active_topk,
            keep_rank=int(p["keep_rank"]),
        )
        cluster_candidates = list(desired_ranked)

        if int(p.get('quality_filter_enable', 0)) == 1:
            kept = [t for t in list(positions.keys()) if t in desired]
            desired_q = list(kept)
            quality_candidates = list(kept)
            for cand in desired_ranked:
                if cand in quality_candidates:
                    continue
                slot_num = min(len(quality_candidates) + 1, active_topk)
                if slot_num in (2, 3) and (cand not in positions):
                    if not _candidate_quality_ok(ranks, rank126, rank252, rs63, d, cand, slot_num, p):
                        continue
                quality_candidates.append(cand)
                if len(desired_q) >= active_topk:
                    continue
                desired_q.append(cand)
            desired = desired_q[:active_topk]
            cluster_candidates = quality_candidates

        if int(p.get('slot3_gate_enable', 0)) == 1 and active_topk >= 3 and len(desired) >= 3:
            t3 = desired[2]
            rank3 = int(ranks.loc[d, t3]) if pd.notna(ranks.loc[d, t3]) else 10**9
            is_lt_leader = bool((rank126.loc[d, t3] <= int(p.get('slot3_leader_exempt_topn', 12))) and (rank252.loc[d, t3] <= int(p.get('slot3_leader_exempt_topn', 12))))
            slot3_rank_max = int(p.get('slot3_max_rank', 4))
            weak_slot3_rank_max = int(p.get('slot3_weak_max_rank', 0))
            if regime_weak and weak_slot3_rank_max > 0:
                slot3_rank_max = min(slot3_rank_max, weak_slot3_rank_max)
            if (rank3 > slot3_rank_max) and (not is_lt_leader):
                desired = desired[:2]

        if int(p.get("loss_cluster_guard_enable", 0)) == 1:
            cluster_map = build_cluster_map(p.get("loss_clusters", ()))
            if cluster_map:
                desired = apply_cluster_limit(
                    primary=desired,
                    fallback=cluster_candidates,
                    topk=active_topk,
                    cluster_map=cluster_map,
                    max_per_cluster=int(p.get("loss_cluster_max_per_cluster", 1)),
                )

        if int(p.get("state_cluster_guard_enable", 0)) == 1 and state_cluster_map:
            dyn_cluster_map = active_state_group_map(
                tickers=cluster_candidates,
                group_map=state_cluster_map,
                state_df=state_cluster_state,
                date=d,
                max_state=float(p.get("state_cluster_r21_max", 0.0)),
            )
            cluster_need = (int(p.get("state_cluster_guard_only_dup", 1)) != 1) or has_group_duplicates(desired, dyn_cluster_map)
            if cluster_need and dyn_cluster_map:
                desired = apply_cluster_limit(
                    primary=desired,
                    fallback=cluster_candidates,
                    topk=active_topk,
                    cluster_map=dyn_cluster_map,
                    max_per_cluster=int(p.get("state_cluster_max_per_cluster", 1)),
                )

        # PP update & exits (checked on close, executed next open)
        pp_exit: List[str] = []
        pp_partial: Dict[str, float] = {}
        if pp_enabled:
            for t, sh in positions.items():
                px = float(c.loc[d, t])
                st = pp_state.get(t)
                if st is None:
                    st = {
                        "entry_px": px,
                        "peak": px,
                        "armed": (pp_mfe_trigger <= 0.0),
                        "arm_idx": (i if pp_mfe_trigger <= 0.0 else None),
                        "entry_idx": i,
                        "partial_count": 0,
                    }
                    pp_state[t] = st
                st["peak"] = max(float(st["peak"]), px)
                mfe = float(st["peak"]) / float(st["entry_px"]) - 1.0
                if (not st["armed"]) and mfe >= pp_mfe_trigger:
                    st["armed"] = True
                    st["arm_idx"] = i
                if st["armed"] and (i - int(st["arm_idx"])) >= pp_min_days_after_arm:
                    dd = 1.0 - px / float(st["peak"]) if float(st["peak"]) > 0 else 0.0
                    if dd >= pp_trail_dd:
                        if int(p.get("pp_leader_partial_enable", 0)) == 1:
                            rank_now = float(ranks.loc[d, t]) if pd.notna(ranks.loc[d, t]) else np.nan
                            rs_now = float(rs63.loc[d, t]) if pd.notna(rs63.loc[d, t]) else np.nan
                            lt_topn = int(p.get("pp_leader_lt_topn", 0))
                            lt_ok = True
                            if lt_topn > 0:
                                r126_now = float(rank126.loc[d, t]) if pd.notna(rank126.loc[d, t]) else np.nan
                                r252_now = float(rank252.loc[d, t]) if pd.notna(rank252.loc[d, t]) else np.nan
                                lt_ok = (
                                    np.isfinite(r126_now)
                                    and np.isfinite(r252_now)
                                    and r126_now <= lt_topn
                                    and r252_now <= lt_topn
                                )
                            partial_count = int(st.get("partial_count", 0))
                            if (
                                partial_count < int(p.get("pp_leader_partial_max", 1))
                                and np.isfinite(rank_now)
                                and np.isfinite(rs_now)
                                and rank_now <= float(p.get("pp_leader_rank_th", 2))
                                and rs_now >= float(p.get("pp_leader_rs63_th", 0.50))
                                and lt_ok
                            ):
                                pp_partial[t] = float(p.get("pp_leader_sell_frac", 0.50))
                        pp_exit.append(t)


        # MIE (Momentum Invalidation Exit) — checked on close, executed next open
        mie_exit: List[str] = []
        if mie_enabled and ("SPY" in c.columns):
            for t in positions.keys():
                px = float(c.loc[d, t])
                st = pp_state.get(t)
                if st is None:
                    continue
                entry_px = float(st.get("entry_px", px))
                entry_idx = int(st.get("entry_idx", i))
                age = i - entry_idx
                live_pnl = (px / entry_px) - 1.0 if entry_px > 0 else 0.0
                # Guard: do not interfere with PP once armed
                if bool(st.get("armed", False)):
                    continue
                if age < int(mie_min_hold):
                    continue
                if live_pnl > 0:
                    continue
                rank_now = float(ranks.loc[d, t]) if pd.notna(ranks.loc[d, t]) else np.nan
                if int(p.get("mie_exempt_rank_max", 0)) > 0 and np.isfinite(rank_now):
                    if rank_now <= float(p.get("mie_exempt_rank_max", 0)):
                        continue
                lt_topn = int(p.get("mie_exempt_lt_topn", 0))
                if lt_topn > 0:
                    r126_now = float(rank126.loc[d, t]) if pd.notna(rank126.loc[d, t]) else np.nan
                    r252_now = float(rank252.loc[d, t]) if pd.notna(rank252.loc[d, t]) else np.nan
                    if (
                        np.isfinite(r126_now)
                        and np.isfinite(r252_now)
                        and r126_now <= lt_topn
                        and r252_now <= lt_topn
                    ):
                        continue
                mie_dd60_th = float(p.get("mie_dd60_th", 0.0))
                if mie_dd60_th > 0.0:
                    dd_now = float(dd60.loc[d, t]) if pd.notna(dd60.loc[d, t]) else np.nan
                    if (not np.isfinite(dd_now)) or (dd_now > -mie_dd60_th):
                        continue
                # Invalidation: underperform SPY and lose SMA220 + short-term weakness
                if (float(rs63.loc[d, t]) <= -float(mie_rs_th)) and (px < float(sma.loc[d, t])) and (float(r5.loc[d, t]) < 0.0):
                    mie_exit.append(t)

        # DOOM_SWITCH evaluation (intra-cycle, PP-safe)
        if doom_enabled:
            if (not do_rebalance) and len(positions) > 0:
                doom_candidates = []
                for t in positions.keys():
                    st = pp_state.get(t)
                    if st is None:
                        continue
                    if bool(st.get('armed', False)):
                        continue
                    entry_px = float(st.get('entry_px', float(c.loc[d, t])))
                    entry_idx = int(st.get('entry_idx', i))
                    age = i - entry_idx
                    if age < int(doom_min_hold):
                        continue
                    px = float(c.loc[d, t])
                    live_pnl = (px / entry_px) - 1.0 if entry_px > 0 else 0.0
                    if live_pnl > 0:
                        continue
                    if (float(rs63.loc[d, t]) <= -float(doom_rs_th)) and (float(r5.loc[d, t]) < 0.0) and (float(dd60.loc[d, t]) <= -float(doom_dd60_th)):
                        doom_candidates.append((live_pnl, t))
                if doom_candidates:
                    doom_candidates.sort(key=lambda x: x[0])
                    worst_t = doom_candidates[0][1]
                    # Select replacement candidate
                    best_cand = None
                    if doom_pick_mode == 'score':
                        for cand in ranked:
                            if cand == worst_t or cand in positions:
                                continue
                            if float(score.loc[d, cand]) <= 0.0:
                                continue
                            best_cand = cand
                            break
                    else:
                        # Persistence-based: maximize fraction of days in Top(persist_topk) over last persist_win days
                        cand_list = []
                        for cand in ranked:
                            if cand == worst_t or cand in positions:
                                continue
                            if float(score.loc[d, cand]) <= 0.0:
                                continue
                            cand_list.append(cand)
                            if len(cand_list) >= int(p.get('rank_pool', 15)):
                                break
                        if cand_list:
                            # if insufficient history, fallback to score
                            try:
                                end_loc = ranks.index.get_loc(d)
                            except KeyError:
                                end_loc = None
                            if end_loc is None or end_loc < (persist_win - 1):
                                best_cand = max(cand_list, key=lambda x: float(score.loc[d, x]))
                            else:
                                start_loc = end_loc - (persist_win - 1)
                                window_dates = ranks.index[start_loc:end_loc+1]
                                def persist(ca):
                                    rr = ranks.loc[window_dates, ca].values
                                    return float(np.mean(rr <= float(persist_topk)))
                                # tie-break by score then -vol
                                best_cand = max(
                                    cand_list,
                                    key=lambda x: (persist(x), float(score.loc[d, x]), -float(vol.loc[d, x]))
                                )
                    if best_cand is not None:
                        doom_exit = [worst_t]
                        doom_buy = best_cand


        # Need next open to execute
        if i + 1 >= len(o.index):
            break

        exec_date = o.index[i + 1]
        px_open = o.loc[exec_date]
        px_close = c.loc[d]

        if not (do_rebalance or pp_exit or mie_exit or doom_exit):
            continue

        # 1) PP exits
        for t in pp_exit:
            if t not in positions:
                continue
            sh = float(positions[t])
            raw_price = float(px_open[t])
            price = sell_exec_price(raw_price)
            partial_frac = float(pp_partial.get(t, 0.0))
            if partial_frac > 0.0:
                sell_sh = min(sh, sh * partial_frac)
            else:
                sell_sh = sh

            cash += sell_sh * price - float(p["fee_per_order"])
            remaining = sh - sell_sh
            if remaining > 1e-10:
                positions[t] = remaining
                prev_st = pp_state.get(t, {})
                pp_state[t] = {
                    "entry_px": price,
                    "peak": price,
                    "armed": False,
                    "arm_idx": None,
                    "entry_idx": i + 1,
                    "partial_count": int(prev_st.get("partial_count", 0)) + 1,
                }
                reason = "PP_TRAIL_PARTIAL"
            else:
                positions.pop(t, None)
                pp_state.pop(t, None)
                if t in loss_names:
                    loss_last_exit_idx[t] = i
                reason = "PP_TRAIL"
            trades_rows.append(
                {
                    "Date": exec_date,
                    "Side": "SELL",
                    "Ticker": t,
                    "Shares": sell_sh,
                    "Price": price,
                    "Fee": float(p["fee_per_order"]),
                    "Reason": reason,
                }
            )


        # 1b) MIE exits
        for t in mie_exit:
            if t not in positions:
                continue
            sh = float(positions.pop(t))
            if t in loss_names:
                loss_last_exit_idx[t] = i
            raw_price = float(px_open[t])
            price = sell_exec_price(raw_price)
            cash += sh * price - float(p["fee_per_order"])
            trades_rows.append(
                {
                    "Date": exec_date,
                    "Side": "SELL",
                    "Ticker": t,
                    "Shares": sh,
                    "Price": price,
                    "Fee": float(p["fee_per_order"]),
                    "Reason": "MOM_INV_RS63",
                }
            )
            pp_state.pop(t, None)

        # 1c) DOOM switches (sell worst doom holding; buy best alternative) 
        if doom_exit:
            freed_cash = 0.0
            for t in doom_exit:
                if t not in positions:
                    continue
                sh = float(positions.pop(t))
                if t in loss_names:
                    loss_last_exit_idx[t] = i
                raw_price = float(px_open[t])
                price = sell_exec_price(raw_price)
                freed_cash += sh * raw_price
                cash += sh * price - float(p['fee_per_order'])
                trades_rows.append({
                    'Date': exec_date,
                    'Side': 'SELL',
                    'Ticker': t,
                    'Shares': sh,
                    'Price': price,
                    'Fee': float(p['fee_per_order']),
                    'Reason': 'DOOM_SWITCH',
                })
                pp_state.pop(t, None)
            # Buy replacement
            if doom_buy is not None:
                raw_price = float(px_open[doom_buy])
                price = buy_exec_price(raw_price)
                max_buy_sh = max((cash - float(p['fee_per_order'])) / price, 0.0) if price > 0 else 0.0
                desired_buy_sh = (freed_cash / raw_price) if raw_price > 0 else 0.0
                buy_sh = min(desired_buy_sh, max_buy_sh)
                if buy_sh > 1e-8:
                    cash -= buy_sh * price + float(p['fee_per_order'])
                    positions[doom_buy] = float(positions.get(doom_buy, 0.0)) + buy_sh
                    trades_rows.append({
                        'Date': exec_date,
                        'Side': 'BUY',
                        'Ticker': doom_buy,
                        'Shares': buy_sh,
                        'Price': price,
                        'Fee': float(p['fee_per_order']),
                        'Reason': 'DOOM_SWITCH',
                    })
                    if pp_enabled and doom_buy not in pp_state:
                        pp_state[doom_buy] = {
                            'entry_px': price,
                            'peak': price,
                            'armed': (pp_mfe_trigger <= 0.0),
                            'arm_idx': (i if pp_mfe_trigger <= 0.0 else None),
                            'entry_idx': i,
                            'partial_count': 0,
                        }

        # 2) Structural rebalance
        if do_rebalance:
            port_val_open = cash
            for t, sh in positions.items():
                port_val_open += float(sh) * float(px_open[t])

            w = invvol_weights(vol.loc[d], desired)
            # Leader overweight (no churn; sizing only)
            if int(p.get('leader_ovw_enable', 0)) == 1 and len(w) > 0:
                win = int(p.get('leader_persist_win', 10))
                topk_lead = int(p.get('leader_topk', 3))
                pmin = float(p.get('leader_persist_min', 0.30))
                alpha = float(p.get('leader_alpha', 0.25))
                cap = float(p.get('leader_w_cap', 0.50))
                if len(leader_hist) >= win:
                    for tkr in list(w.keys()):
                        st = pp_state.get(tkr, {})
                        if bool(st.get('armed', False)):
                            continue
                        hits = sum(1 for s in leader_hist if tkr in s)
                        persist = hits / float(win)
                        if persist >= pmin:
                            w[tkr] *= (1.0 + alpha * persist)
                    # renormalize
                    ssum = float(sum(w.values()))
                    if ssum > 0:
                        for tkr in w:
                            w[tkr] /= ssum
                    # cap + redistribute
                    over = {k:v for k,v in w.items() if v > cap}
                    if over:
                        excess = sum(v - cap for v in over.values())
                        for k in over:
                            w[k] = cap
                        under = [k for k,v in w.items() if v < cap]
                        under_sum = sum(w[k] for k in under)
                        if under_sum > 0 and excess > 0:
                            for k in under:
                                w[k] += excess * (w[k] / under_sum)
                    # final renorm
                    ssum = float(sum(w.values()))
                    if ssum > 0:
                        for tkr in w:
                            w[tkr] /= ssum

            # Tail-risk conditional cap (targets-only, sizing only; no forced sells)
            # Condition: high ATR% + trend break + vol-spike (turning tails)
            if int(p.get('tailcap_enable', 0)) == 1 and len(w) > 0:
                atr_th = float(p.get('tailcap_atrp_th', 0.12))
                spike_th = float(p.get('tailcap_volspike_th', 1.5))
                indiv_cap = float(p.get('tailcap_indiv_cap', 0.15))
                total_cap = float(p.get('tailcap_total_cap', 0.30))

                # compute risky set among desired names at date d (signals on close d)
                risky = []
                for tkr in list(w.keys()):
                    cl_d = float(c.loc[d, tkr]) if pd.notna(c.loc[d, tkr]) else np.nan
                    sma_d = float(sma.loc[d, tkr]) if pd.notna(sma.loc[d, tkr]) else np.nan
                    a_d = float(atrp20.loc[d, tkr]) if pd.notna(atrp20.loc[d, tkr]) else np.nan
                    v20 = float(vol.loc[d, tkr]) if pd.notna(vol.loc[d, tkr]) else np.nan
                    v60 = float(vol60.loc[d, tkr]) if pd.notna(vol60.loc[d, tkr]) else np.nan
                    if np.isnan(cl_d) or np.isnan(sma_d) or np.isnan(a_d) or np.isnan(v20) or np.isnan(v60):
                        continue
                    dd60_d = float(dd60.loc[d, tkr]) if pd.notna(dd60.loc[d, tkr]) else np.nan
                    # dd60 is a distance-from-high metric (<= 0), so a drawdown
                    # threshold must be checked on the negative side.
                    turning = (cl_d < sma_d) or (not np.isnan(dd60_d) and dd60_d <= -float(p.get('tailcap_dd60_th', 0.15)))
                    vol_spike = (v20 / (v60 + 1e-12)) >= spike_th
                    if (a_d >= atr_th) and turning and vol_spike:
                        risky.append(tkr)

                if risky:
                    removed = 0.0
                    # indiv cap first
                    for tkr in risky:
                        if w.get(tkr, 0.0) > indiv_cap:
                            removed += (w[tkr] - indiv_cap)
                            w[tkr] = indiv_cap
                    # total cap
                    s_r = sum(w.get(tkr, 0.0) for tkr in risky)
                    if s_r > total_cap:
                        scale = total_cap / s_r
                        for tkr in risky:
                            old = w[tkr]
                            w[tkr] = old * scale
                        removed += (s_r - total_cap)
                    # redistribute removed weight to non-risk names proportionally
                    if removed > 0:
                        nonr = [k for k in w.keys() if k not in risky]
                        non_sum = sum(w[k] for k in nonr)
                        if non_sum > 0:
                            for k in nonr:
                                w[k] += removed * (w[k] / non_sum)
                    # final renorm
                    ssum = float(sum(w.values()))
                    if ssum > 0:
                        for tkr in w:
                            w[tkr] /= ssum
            if int(p.get("loss_weight_cap_enable", 0)) == 1 and loss_names:
                w = cap_ticker_weights(
                    weights=w,
                    capped_names=loss_names,
                    indiv_cap=float(p.get("loss_weight_cap", 1.0)),
                )
            if int(p.get("loss_total_weight_cap_enable", 0)) == 1 and loss_names:
                w = cap_group_total_weight(
                    weights=w,
                    capped_names=loss_names,
                    total_cap=float(p.get("loss_total_weight_cap", 1.0)),
                )
            if int(p.get("state_weight_cap_enable", 0)) == 1 and state_weight_map:
                dyn_weight_map = active_state_group_map(
                    tickers=list(w.keys()),
                    group_map=state_weight_map,
                    state_df=state_weight_state,
                    date=d,
                    max_state=float(p.get("state_weight_r21_max", 0.0)),
                )
                weight_need = (int(p.get("state_weight_only_dup", 1)) != 1) or has_group_duplicates(list(w.keys()), dyn_weight_map)
                if weight_need and dyn_weight_map:
                    indiv_cap = float(p.get("state_weight_indiv_cap", 1.0))
                    total_cap = float(p.get("state_weight_total_cap", 1.0))
                    if indiv_cap < 1.0:
                        w = cap_ticker_weights(
                            weights=w,
                            capped_names=set(dyn_weight_map.keys()),
                            indiv_cap=indiv_cap,
                        )
                    if total_cap < 1.0:
                        w = cap_group_total_weight(
                            weights=w,
                            capped_names=set(dyn_weight_map.keys()),
                            total_cap=total_cap,
                        )
            if int(p.get("slot3_weight_cap_enable", 0)) == 1 and len(desired) >= 3:
                if (int(p.get("slot3_weight_cap_weak_only", 0)) != 1) or regime_weak:
                    slot3 = desired[2]
                    if slot3 in w:
                        w = cap_ticker_weights(
                            weights=w,
                            capped_names=[slot3],
                            indiv_cap=float(p.get("slot3_weight_cap", 1.0)),
                        )
            targets_val = {t: w[t] * port_val_open for t in w}
            # Reset deferred exits for names that are back in target
            for _t in list(exit_defer.keys()):
                if _t in targets_val:
                    exit_defer.pop(_t, None)

            # Sell names not in target
            for t in list(positions.keys()):
                if t not in targets_val:
                    # Optional EXIT_NOT_TARGET smoothing: defer selling losers that still pass trend (and optionally pos RS63)
                    if exit_smooth_enable:
                        st = pp_state.get(t)
                        if st is not None:
                            entry_px = float(st.get("entry_px", np.nan))
                            cl_px = float(px_close[t])
                            upnl = (cl_px / entry_px - 1.0) if (entry_px and entry_px > 0) else 0.0
                            ok = (upnl < 0.0)
                            if exit_smooth_require_trend:
                                ok = ok and (cl_px > float(sma.loc[d, t]))
                            if exit_smooth_require_pos_rs63:
                                ok = ok and (float(r63.loc[d, t]) > 0.0)
                            if ok and exit_defer.get(t, 0) < exit_smooth_max_defers:
                                exit_defer[t] = exit_defer.get(t, 0) + 1
                                continue  # keep position one more rebalance
                    st = pp_state.get(t)
                    cl_px = float(px_close[t]) if pd.notna(px_close[t]) else np.nan
                    if _convex_nonpp_guard_ok(
                        ticker=t,
                        st=st,
                        close_px=cl_px,
                        d=d,
                        rs63=rs63,
                        dd60=dd60,
                        rank126=rank126,
                        rank252=rank252,
                        p=p,
                        prefix="convex_exit_guard",
                    ):
                        max_defers = int(p.get("convex_exit_guard_max_defers", 0))
                        if exit_defer.get(t, 0) < max_defers:
                            exit_defer[t] = exit_defer.get(t, 0) + 1
                            continue
                    sh = float(positions.pop(t))
                    if t in loss_names:
                        loss_last_exit_idx[t] = i
                    exit_defer.pop(t, None)
                    raw_price = float(px_open[t])
                    price = sell_exec_price(raw_price)
                    cash += sh * price - float(p["fee_per_order"])
                    trades_rows.append(
                        {
                            "Date": exec_date,
                            "Side": "SELL",
                            "Ticker": t,
                            "Shares": sh,
                            "Price": price,
                            "Fee": float(p["fee_per_order"]),
                            "Reason": "EXIT_NOT_TARGET",
                        }
                    )
                    pp_state.pop(t, None)

            # Recompute portfolio value after sells
            port_val_open = cash
            for t, sh in positions.items():
                port_val_open += float(sh) * float(px_open[t])

            # Delta rebalance
            for t, tgt_val in targets_val.items():
                cur_sh = float(positions.get(t, 0.0))
                raw_price = float(px_open[t])
                cur_val = cur_sh * raw_price
                diff = float(tgt_val) - cur_val

                if abs(diff) < float(p["delta_rebal"]) * port_val_open:
                    continue

                if diff < 0 and cur_sh > 0:
                    st = pp_state.get(t)
                    cl_px = float(px_close[t]) if pd.notna(px_close[t]) else np.nan
                    if _convex_nonpp_guard_ok(
                        ticker=t,
                        st=st,
                        close_px=cl_px,
                        d=d,
                        rs63=rs63,
                        dd60=dd60,
                        rank126=rank126,
                        rank252=rank252,
                        p=p,
                        prefix="convex_delta_guard",
                    ):
                        continue
                    price = sell_exec_price(raw_price)
                    sell_sh = min((-diff) / raw_price, cur_sh) if raw_price > 0 else 0.0
                    cash += sell_sh * price - float(p["fee_per_order"])
                    new_sh = cur_sh - sell_sh
                    if new_sh <= 1e-10:
                        positions.pop(t, None)
                        pp_state.pop(t, None)
                        if t in loss_names:
                            loss_last_exit_idx[t] = i
                    else:
                        positions[t] = new_sh
                    trades_rows.append(
                        {
                            "Date": exec_date,
                            "Side": "SELL",
                            "Ticker": t,
                            "Shares": sell_sh,
                            "Price": price,
                            "Fee": float(p["fee_per_order"]),
                            "Reason": "DELTA_SELL",
                        }
                    )

                elif diff > 0:
                    price = buy_exec_price(raw_price)
                    max_buy_sh = max((cash - float(p["fee_per_order"])) / price, 0.0) if price > 0 else 0.0
                    desired_buy_sh = (diff / raw_price) if raw_price > 0 else 0.0
                    buy_sh = min(desired_buy_sh, max_buy_sh)
                    if buy_sh > 1e-8:
                        cash -= buy_sh * price + float(p["fee_per_order"])
                        positions[t] = cur_sh + buy_sh
                        trades_rows.append(
                            {
                                "Date": exec_date,
                                "Side": "BUY",
                                "Ticker": t,
                                "Shares": buy_sh,
                                "Price": price,
                                "Fee": float(p["fee_per_order"]),
                                "Reason": "DELTA_BUY",
                            }
                        )
                        # init PP state on entry
                        if pp_enabled and t not in pp_state:
                            pp_state[t] = {
                                "entry_px": price,
                                "peak": price,
                                "armed": (pp_mfe_trigger <= 0.0),
                                "arm_idx": (i if pp_mfe_trigger <= 0.0 else None),
                                "entry_idx": i,
                                "partial_count": 0,
                            }

    eq = pd.DataFrame(equity_rows).set_index("date")
    tr = pd.DataFrame(trades_rows)

    final = float(eq["equity"].iloc[-1])

    # Metrics
    peak = eq["equity"].cummax()
    dd = eq["equity"] / peak - 1.0
    maxdd = float(dd.min()) * 100.0

    r = eq["equity"].pct_change().dropna()
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(252))

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (final / invested) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    out = {
        "FinalEUR": final,
        "InvestedEUR": float(invested),
        "NetGainEUR": float(final - invested),
        "ROI_%": float((final - invested) / invested * 100.0),
        "CAGR_%": float(cagr * 100.0),
        "MaxDD_%": float(maxdd),
        "Sharpe": float(sharpe),
        "Orders": int(len(tr)),
        "PP": {
            "enabled": bool(pp_enabled),
            "mfe_trigger": float(pp_mfe_trigger),
            "trail_dd": float(pp_trail_dd),
            "min_days_after_arm": int(pp_min_days_after_arm),
        },
    }

    return eq, tr, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="apex_ohlcv_full_2015_2026.csv")
    ap.add_argument("--start", default=DEFAULTS["start"])
    ap.add_argument("--end", default=DEFAULTS["end"])

    ap.add_argument("--pp", action="store_true")
    ap.add_argument("--pp_mfe", type=float, default=0.37)
    ap.add_argument("--pp_trail", type=float, default=0.09)
    ap.add_argument("--pp_min", type=int, default=4)

    ap.add_argument("--out_prefix", default="run")

    args = ap.parse_args()

    prices = load_prices_from_csv(args.csv)

    eq, tr, metrics = run_backtest(
        prices,
        args.start,
        args.end,
        pp_enabled=args.pp,
        pp_mfe_trigger=args.pp_mfe,
        pp_trail_dd=args.pp_trail,
        pp_min_days_after_arm=args.pp_min,
        **DEFAULTS,
    )

    eq.to_csv(f"{args.out_prefix}__equity.csv")
    tr.to_csv(f"{args.out_prefix}__trades.csv", index=False)
    pd.Series(metrics).to_json(f"{args.out_prefix}__metrics.json", indent=2)

    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
