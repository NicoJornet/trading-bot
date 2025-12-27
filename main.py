import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v25.2.3 — FULL PRODUCTION (FIXED REGIME SCOPE)
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
    print(f"🚀 Initialisation APEX v25.2.3")
    
    # 1. Téléchargement des données
    try:
        data = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"], period="2y", auto_adjust=True, progress=False)
        if data.empty:
            print("❌ Erreur: DataFrame vide")
            return
        close = data['Close'].ffill().bfill()
        high = data['High'].ffill().bfill()
        low = data['Low'].ffill().bfill()
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return

    # 2. Calcul du régime de marché (Indispensable pour définir exposure et regime)
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    tnx = close["^TNX"]
    irx = close["^IRX"]
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0
    
    spy_ma200 = spy.rolling(200).mean()
    vix_ma50 = vix.rolling(50).mean()
    
    score = 0
    if float(spy.iloc[-1]) > float(spy_ma200.iloc[-1]): score += 0.4
    if float(vix.iloc[-1]) < float(vix_ma50.iloc[-1]): score += 0.3
    if float(spy.iloc[-1]) > float(spy.shift(63).iloc[-1]): score += 0.2
    if (float(tnx.iloc[-1]) - float(irx.iloc[-1])) > 0: score += 0.1
    
    # Définition sécurisée des variables globales de la fonction
    if score >= 0.65:
        exposure = 1.0
        regime = "🟢🟢🟢 MAX"
    elif score >= 0.45:
        exposure = 0.75
        regime = "🟢 STRONG"
    elif score >= 0.30:
        exposure = 0.50
        regime = "🟡 NEUTRAL"
    else:
        exposure = 0.0
        regime = "🔴 BEAR"

    # 3. Traitement du message
    if exposure == 0:
        msg = f"🤖 APEX v25.2.3\n{regime} | Expo: 0%\n💰 Capital: {TOTAL_CAPITAL}€\n━━━━━━━━━━━━━━\n⚠️ **100% CASH**"
        if TOKEN and CHAT_ID:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        return

    # 4. Sélection des candidats
    universe = ALL_TICKERS if exposure >= 0.5 else DEFENSIVE_TICKERS
    active_p = close[universe].dropna(axis=1)
    
    mom = active_p.pct_change(126).iloc[-1]
    ma150 = active_p.rolling(150).mean().iloc[-1]
    rsi = active_p.apply(calculate_rsi).iloc[-1]
    adx = pd.Series({t: calculate_adx(high[t], low[t], close[t]).iloc[-1] for t in active_p.columns})
    
    valid = (rsi < 78) & (active_p.iloc[-1] > ma150) & (adx > 20) & (mom > 0)
    all_candidates = mom[valid].nlargest(8).index.tolist()

    # 5. Construction du rapport multi-top
    msg = f"🤖 APEX v25.2.3 | {regime} ({int(exposure*100)}%)\n💰 Cap: {TOTAL_CAPITAL}€ | 🛡️ SL: {ATR_MULT} ATR\n━━━━━━━━━━━━━━\n\n"

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
            sl_pct = ((p_eur - sl_eur) / p_eur) * 100
            
            w = min(((TOTAL_CAPITAL * RISK_PER_TRADE) / (p_eur - sl_eur)) * p_eur / TOTAL_CAPITAL, 0.40 if n <= 3 else 0.25)
            pos_details.append((t, w, p_eur, sl_eur, sl_pct))
            weights_sum += w

        scale = exposure / weights_sum if weights_sum > 0 else 0
        for t, w, p_eur, sl_eur, sl_pct in pos_details:
            final_w = w * scale
            msg += f"• **{t}**: {final_w*100:.1f}% ({TOTAL_CAPITAL*final_w:.0f}€)\n"
            msg += f"  Prix: {p_eur:.2f}€ | **SL: {sl_eur:.2f}€ (-{sl_pct:.1f}%)**\n"
        msg += "──────────────\n"

    msg += "💡 Ne changez de position que si un titre entre ou sort du TOP.\n"
    msg += "⚡ Process > Emotion"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    print("✅ Notification APEX v25.2.3 envoyée.")

if __name__ == "__main__":
    run()
