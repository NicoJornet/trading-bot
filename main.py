import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# 🛡️ APEX v25.7 — PRODUCTION (SAFE MODE) - CORRIGÉ
# Suppression de matplotlib pour éviter l'erreur ModuleNotFoundError
# ============================================================

# --- 1. CONFIGURATION ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ⚠️ METTRE A JOUR CE MONTANT REGULIEREMENT AVEC TON SOLDE REEL
TOTAL_CAPITAL = 2000  

# Paramètres de Risque
RISK_PER_TRADE = 0.02
ATR_MULT = 3.0
MIN_QUALITY = 2
TARGET_RISK_DAILY = 0.0125  # Tolérance de risque : 1.25% du capital par jour max

# Univers d'Investissement
OFFENSIVE_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "AAPL", "AVGO", "AMD", "QCOM", "MU",
    "CRWD", "PANW", "NET", "DDOG", "ZS", "ASML", "TSM", "LRCX", "AMAT", "KLAC",
    "TSLA", "PLTR", "RKLB", "ABNB", "SHOP", "VRT", "APP", "QQQ", "SMH", "SOXX", "IGV",
    "BTC-USD", "ETH-USD"
]

DEFENSIVE_TICKERS = [
    "LLY", "UNH", "JNJ", "ABBV", "TMO", "DHR", "ISRG", "PG", "KO", "PEP", "WMT", 
    "XLU", "NEE", "XLE", "GLD", "SLV", "DBA", "PDBC", "LMT", "RTX", "BA", "ITA",
    "MC.PA", "RACE", "RMS.PA"
]

ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"

# --- 2. INDICATEURS TECHNIQUES ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_adx_vectorized(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()

def quality_score_fast(prices, spy):
    scores = pd.Series(0, index=prices.columns)
    ma50 = prices.rolling(50).mean().iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    current = prices.iloc[-1]
    scores += ((current > ma50) & (ma50 > ma200)).astype(int)
    
    ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1)
    ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1)
    ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1)
    scores += ((ret_1m > 0) & (ret_3m > 0) & (ret_6m > 0)).astype(int)
    
    spy_ret_6m = spy.iloc[-1] / spy.iloc[-126] - 1
    rel_strength = ret_6m / spy_ret_6m
    scores += (rel_strength > 1.0).astype(int)
    
    max_gap = prices.pct_change().tail(20).abs().max()
    scores += (max_gap < 0.10).astype(int)
    return scores

def detect_regime(spy, vix, tnx, irx):
    spy_ma200 = spy.rolling(200).mean()
    vix_ma50 = vix.rolling(50).mean()
    score = 0
    if float(spy.iloc[-1]) > float(spy_ma200.iloc[-1]): score += 0.4
    if float(vix.iloc[-1]) < float(vix_ma50.iloc[-1]): score += 0.3
    if float(spy.iloc[-1]) > float(spy.iloc[-63]): score += 0.2
    try:
        if (float(tnx.iloc[-1]) - float(irx.iloc[-1])) > 0: score += 0.1
    except: pass
    
    if score >= 0.70: return 1.00, "🟢🟢🟢 MAX BULL"
    elif score >= 0.55: return 0.80, "🟢🟢 STRONG"
    elif score >= 0.40: return 0.60, "🟢 BULL"
    elif score >= 0.25: return 0.35, "🟡 NEUTRAL"
    elif score >= 0.15: return 0.15, "🟠 CAUTIOUS"
    else: return 0.00, "🔴 BEAR"

def get_safe_weight(price, atr, n_positions):
    if price <= 0: return 0
    volatility_pct = atr / price
    if volatility_pct <= 0: volatility_pct = 0.01
    max_weight_vol = TARGET_RISK_DAILY / volatility_pct
    ceiling = 1.0 / n_positions
    return min(max_weight_vol, ceiling)

def select_positions(active_prices, high, low, spy, exposure, capital, n_positions):
    mom_6m = active_prices.pct_change(126).iloc[-1]
    spy_ret = spy.pct_change(126).iloc[-1]
    rel_strength = mom_6m / spy_ret
    z_mom = (mom_6m - mom_6m.mean()) / mom_6m.std()
    z_rs = (rel_strength - rel_strength.mean()) / rel_strength.std()
    q_scores = quality_score_fast(active_prices, spy)
    final_scores = 0.50 * z_mom + 0.30 * z_rs + 0.20 * (q_scores / 4.0)
    rsi = active_prices.apply(calculate_rsi).iloc[-1]
    ma150 = active_prices.rolling(150).mean().iloc[-1]
    
    adx = pd.Series(index=active_prices.columns, dtype=float)
    for ticker in active_prices.columns:
        try:
            adx[ticker] = float(calculate_adx_vectorized(high[ticker], low[ticker], active_prices[ticker]).iloc[-1])
        except: adx[ticker] = 0
    
    valid = (final_scores > 0) & (rsi < 78) & (active_prices.iloc[-1] > ma150) & (q_scores >= MIN_QUALITY) & (adx > 20) & (mom_6m > 0)
    candidates = final_scores[valid].nlargest(n_positions * 2)
    if len(candidates) == 0: return []
    selected = list(candidates.nlargest(n_positions).index)
    
    positions = []
    for ticker in selected:
        price = float(active_prices[ticker].iloc[-1])
        tr = pd.concat([high[ticker]-low[ticker], abs(high[ticker]-active_prices[ticker].shift(1)), abs(low[ticker]-active_prices[ticker].shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        sl_price = price - (ATR_MULT * atr)
        safe_weight = get_safe_weight(price, atr, n_positions)
        final_weight = safe_weight * exposure
        positions.append({'ticker': ticker, 'weight': final_weight, 'price': price, 'sl': sl_price})
    return positions

# --- 3. EXECUTION ---

def run():
    print(f"🚀 APEX v25.7 (SAFE) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    try:
        data = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"], period="2y", auto_adjust=True, progress=False)
        close = data['Close'].ffill().bfill(); high = data['High'].ffill().bfill(); low = data['Low'].ffill().bfill()
    except Exception as e:
        print(f"Erreur Data: {e}"); return

    spy, vix, tnx, irx = close[MARKET_INDEX], close["^VIX"], close["^TNX"], close["^IRX"]
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0
    exposure, regime = detect_regime(spy, vix, tnx, irx)
    
    if exposure == 0:
        msg = f"🛡️ **APEX PROTECTION**\n{regime} | Expo: 0%\n⚠️ **RESTER 100% CASH**"
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        return

    universe = OFFENSIVE_TICKERS if exposure >= 0.50 else DEFENSIVE_TICKERS
    active_p = close[universe].dropna(axis=1)
    msg = f"🤖 **APEX v25.7** | {regime} (Expo {int(exposure*100)}%)\n🛡️ **MODE SAFE** | Cap: {TOTAL_CAPITAL}€\n━━━━━━━━━━━━━━\n\n"
    
    pos = select_positions(active_p, high[universe], low[universe], spy, exposure, TOTAL_CAPITAL, 2)
    if pos:
        total_invested = 0
        for p in pos:
            p_eur = p['price'] * (1 if p['ticker'].endswith(".PA") else fx)
            sl_eur = p['sl'] * (1 if p['ticker'].endswith(".PA") else fx)
            alloc = TOTAL_CAPITAL * p['weight']
            total_invested += alloc
            icon = "🛡️" if p['weight'] < 0.45 else "⚡"
            msg += f"• **{p['ticker']}** {icon}\n  Alloc: **{p['weight']*100:.1f}%** ({alloc:.0f}€)\n  Prix: {p_eur:.2f}€ | SL: {sl_eur:.2f}€\n\n"
        cash_restant = TOTAL_CAPITAL - total_invested
        msg += f"💰 **CASH :** {cash_restant:.0f}€ ({(cash_restant/TOTAL_CAPITAL)*100:.0f}%)\n"
    else:
        msg += "⚠️ Aucune opportunité trouvée. Rester Cash."

    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__": run()
