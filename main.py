import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v25.2.1 — MULTI-TOP AVEC PRIX & % STOP LOSS
# ============================================================

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOTAL_CAPITAL = 1000
RISK_PER_TRADE = 0.02  
ATR_MULT = 3.3         

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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()

def run():
    print(f"🚀 Lancement APEX v25.2.1 — {datetime.now().strftime('%Y-%m-%d')}")
    try:
        data = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"], period="2y", auto_adjust=True, progress=False)
        close = data['Close'].ffill().bfill(); high = data['High'].ffill().bfill(); low = data['Low'].ffill().bfill()
    except: return

    spy, vix, tnx, irx = close[MARKET_INDEX], close["^VIX"], close["^TNX"], close["^IRX"]
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0
    
    score = (0.4*(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]) + 
             0.3*(vix.iloc[-1] < vix.rolling(50).mean().iloc[-1]) + 
             0.2*(spy.iloc[-1] > spy.iloc[-63]) + 
             0.1*((tnx.iloc[-1]-irx.iloc[-1]) > 0))
    
    exposure = 1.0 if score >= 0.65 else 0.75 if score >= 0.45 else 0.5 if score >= 0.3 else 0.0
    regime = "🟢🟢🟢 MAX" if exposure == 1.0 else "🟢 STRONG" if exposure >= 0.75 else "🟡 NEUTRAL" if exposure > 0 else "🔴 BEAR"

    if exposure == 0:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": f"🤖 APEX v25.2.1\n{regime} | Expo: 0%\n⚠️ **100% CASH**", "parse_mode": "Markdown"})
        return

    universe = ALL_TICKERS if exposure >= 0.5 else DEFENSIVE_TICKERS
    active_p = close[universe].dropna(axis=1)
    mom = active_p.pct_change(126).iloc[-1]
    ma150 = active_p.rolling(150).mean().iloc[-1]
    rsi = active_p.apply(calculate_rsi).iloc[-1]
    adx = pd.Series({t: calculate_adx(high[t], low[t], close[t]).iloc[-1] for t in active_p.columns})
    
    valid = (rsi < 78) & (active_p.iloc[-1] > ma150) & (adx > 20) & (mom > 0)
    all_candidates = mom[valid].nlargest(8).index.tolist()

    msg = f"🤖 APEX v25.2.1 | {regime} ({int(exposure*100)}%)\n💰 Cap: {TOTAL_CAPITAL}€ | 🛡️ SL: {ATR_MULT} ATR\n━━━━━━━━━━━━━━\n\n"

    for n in [2, 3, 6, 8]:
        selected = all_candidates[:n]
        if not selected: continue
        
        msg += f"🏆 **TOP {len(selected)}**\n"
        weights_sum = 0
        pos_details = []

        for t in selected:
            p_eur = float(active_p[t].iloc[-1]) * (1 if t.endswith(".PA") else fx)
            tr = pd.concat([high[t]-low[t], abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            sl_eur = p_eur - (ATR_MULT * atr * (1 if t.endswith(".PA") else fx))
            
            # Distance du stop en %
            sl_pct = ((p_eur - sl_eur) / p_eur) * 100
            
            w = min(((TOTAL_CAPITAL * RISK_PER_TRADE) / (p_eur - sl_eur)) * p_eur / TOTAL_CAPITAL, 0.40 if n <= 3 else 0.25)
            pos_details.append((t, w, p_eur, sl_eur, sl_pct))
            weights_sum += w

        scale = exposure / weights_sum if weights_sum > 0 else 0
        for t, w, p_eur, sl_eur, sl_pct in pos_details:
            final_w = w * scale
            msg += f"• **{t}**: {final_w*100:.1f}% ({TOTAL_CAPITAL*final_w:.0f}€)\n"
            msg += f"  Prix: {p_eur:.2f}€ | **SL: {sl_eur:.2f}€ (-{sl_pct:.1f}%)**\n"
        msg += "\n"

    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg + "⚡ Process > Emotion", "parse_mode": "Markdown"})

if __name__ == "__main__": run()
