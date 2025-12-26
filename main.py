import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize

# ============================================================
# APEX v24.6 — EXPOSITION DISCRÈTE + PDBC & XLU (DÉFENSIFS)
# ============================================================
# Instructions : Configurez vos secrets (TOKEN, CHAT_ID) dans les variables d'environnement GitHub.

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOTAL_CAPITAL = 1000  # À mettre à jour selon votre capital réel

OFFENSIVE_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "ASML", "AVGO", "SMH", "VRT",
    "PLTR", "PANW", "RKLB", "CRWD", "SMCI", "ARM", "APP", "BTC-USD", "ETH-USD", "SOL-USD"
]

DEFENSIVE_TICKERS = [
    "LLY", "UNH", "ISRG", "ETN", "URNM", "XLE", "COPX", "SIL", "REMX", "GLD", "ITA", "RACE", "MC.PA",
    "PDBC",   # Commodities (Protection contre l'inflation)
    "XLU"     # Utilities (Secteur ultra-défensif)
]

ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def run():
    print(f"🚀 Lancement APEX v24.6 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # --- 1. Téléchargement des données ---
    try:
        raw = yf.download(
            ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"],
            period="2y",
            auto_adjust=True,
            progress=False
        )
        close = raw["Close"].ffill()
        high = raw["High"].ffill()
        low = raw["Low"].ffill()
    except Exception as e:
        print(f"❌ Erreur Data: {e}")
        return

    prices = close[ALL_TICKERS].ffill().bfill()
    spy = close[MARKET_INDEX].reindex(prices.index).ffill().bfill()
    vix = close["^VIX"].reindex(prices.index).ffill().bfill()
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0

    # --- 2. ANALYSE DU RÉGIME DE MARCHÉ (SCORE 0-4) ---
    ma200 = spy.rolling(200).mean()
    vix_threshold = vix.rolling(252).quantile(0.4).iloc[-1]

    score = sum([
        float(spy.iloc[-1]) > float(ma200.iloc[-1]),
        float(ma200.iloc[-1]) > float(ma200.iloc[-20]),
        float(vix.iloc[-1]) < float(vix_threshold)
    ])

    # Yield Curve Spread (10Y - 3M)
    try:
        spread = close["^TNX"] - close["^IRX"]
        score += int(spread.iloc[-1] > 0)
    except:
        pass

    # EXPOSITION DISCRÈTE (Paliers de 25%)
    exposure_map = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
    exposure = exposure_map.get(score, 0.0)

    regime_icons = {0: "🔴", 1: "🟡", 2: "🟢🟢", 3: "🟢🟢🟢", 4: "🟢🟢🟢🟢"}
    regime_names = {0: "BEAR (Cash)", 1: "CAUTIOUS", 2: "BULL", 3: "STRONG BULL", 4: "MAX BULL"}

    regime_icon = regime_icons.get(score, "🔴")
    regime_name = regime_names.get(score, "BEAR")
    ground = DEFENSIVE_TICKERS if score <= 1 else ALL_TICKERS

    # GESTION DU CASH TOTAL
    if exposure == 0.0:
        msg = f"🤖 APEX v24.6 | {datetime.now().strftime('%d %B %Y')}\n"
        msg += f"**Régime:** {regime_icon} **{regime_name}** | Expo: **0%**\n"
        msg += f"💰 Capital: **{TOTAL_CAPITAL:,}€**\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += "⚠️ **TOTAL CASH — Protection maximale activée.**\n"
        msg += "Le marché présente trop de risques structurels.\n\n"
        msg += "Process > Conviction | Never Average Down"
        if TOKEN and CHAT_ID:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        print("Signal: 100% Cash")
        return

    # --- 3. SCORING ALPHA ---
    active_prices = prices[ground].copy()
    returns = active_prices.pct_change()

    # Momentum multi-temporel
    m = (0.2 * (active_prices / active_prices.shift(63) - 1) +
         0.3 * (active_prices / active_prices.shift(126) - 1) +
         0.5 * (active_prices / active_prices.shift(252) - 1))

    # Force Relative vs SPY
    spy_ratio = spy / spy.shift(126)
    asset_ratio = active_prices / active_prices.shift(126)
    rs = asset_ratio.div(spy_ratio, axis=0)

    # Volatility Adjusted Momentum
    vol_252 = returns.rolling(252).std() * np.sqrt(252)
    risk_adj_mom = m.div(vol_252.clip(lower=0.2), axis=0)

    # Normalisation Z-Score
    z_mom = (m.sub(m.mean(axis=1), axis=0)).div(m.std(axis=1), axis=0)
    z_rs = (rs.sub(rs.mean(axis=1), axis=0)).div(rs.std(axis=1), axis=0)

    final_scores = (
        0.55 * z_mom.iloc[-1] +
        0.30 * z_rs.iloc[-1] +
        0.15 * risk_adj_mom.iloc[-1].rank(pct=True)
    )

    # Filtres de sécurité
    rsi_vals = active_prices.apply(calculate_rsi).iloc[-1]
    ma150 = active_prices.rolling(150).mean().iloc[-1]
    valid = (final_scores > 0) & (rsi_vals < 80) & (active_prices.iloc[-1] > ma150)
    candidates = final_scores[valid].nlargest(10)

    # --- 4. SÉLECTION ET PORTFOLIO ---
    msg = f"🤖 APEX v24.6 | {datetime.now().strftime('%d %B %Y')}\n"
    msg += f"**Régime:** {regime_icon} **{regime_name}** | Expo: **{int(exposure*100)}%**\n"
    msg += f"💰 Capital: **{TOTAL_CAPITAL:,}€**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

    if candidates.empty:
        msg += "⚠️ **Aucun signal robuste.** Augmenter la part de cash.\n"
    else:
        for n in [1, 2, 3]:
            # HRP Simplifié (Clustering pour diversifier les choix)
            if len(candidates) <= n:
                selected = candidates.index.tolist()
            else:
                corr_mat = returns[candidates.index].iloc[-126:].corr().fillna(0)
                dist = np.sqrt(2 * (1 - corr_mat.replace(1, 0.999)))
                try:
                    Z = linkage(dist, method='ward')
                    clusters = fcluster(Z, t=1.8, criterion='distance')
                    selected = []
                    for cid in np.unique(clusters):
                        cluster_ticks = candidates.index[clusters == cid]
                        selected.append(candidates[cluster_ticks].idxmax())
                        if len(selected) >= n: break
                except:
                    selected = candidates.index[:n].tolist()

            recommended = (
                (n == 1 and TOTAL_CAPITAL < 3000) or
                (n == 2 and 3000 <= TOTAL_CAPITAL < 7000) or
                (n == 3 and TOTAL_CAPITAL >= 7000)
            )
            status = "⭐ RECOMMANDÉ" if recommended else "🔹 Option"
            msg += f"🏆 **TOP {n}** | {status}\n"

            # Optimisation de l'allocation (Mean-Variance simple avec plafond)
            if len(selected) == 1:
                weights = pd.Series([exposure], index=selected)
            else:
                cov = returns[selected].iloc[-252:].cov() * 252
                def port_vol(w): return np.sqrt(w @ cov @ w + 1e-8)
                res = minimize(port_vol, np.full(len(selected), exposure/len(selected)), 
                               constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - exposure}),
                               bounds=[(0.05, 0.45)] * len(selected))
                weights = pd.Series(res.x, index=selected)
                weights = (weights / weights.sum()) * exposure

            for t in selected:
                price_eur = float(prices[t].iloc[-1]) * (1 if t.endswith(".PA") else fx)
                # Calcul ATR pour Stop Loss
                tr = pd.concat([high[t]-low[t], abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                
                # Stop Hybride
                mult = 3.3 if exposure >= 0.75 else 4.2
                chandelier = float(prices[t].iloc[-1]) - mult * atr
                ma_protect = float(prices[t].rolling(100).mean().iloc[-1] * 0.93)
                stop_eur = max(chandelier, ma_protect) * (1 if t.endswith(".PA") else fx)

                msg += f"• **{t}** — {weights[t]*100:.1f}% | {price_eur:.2f}€ → **SL {stop_eur:.2f}€**\n"
            msg += "\n"

    msg += "Process > Conviction | Never Average Down"

    if TOKEN and CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e: print(f"Erreur Telegram: {e}")
    print(msg)

if __name__ == "__main__":
    run()
