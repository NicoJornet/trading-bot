import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v25.3 — PRODUCTION OPTIMIZED
# ============================================================

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOTAL_CAPITAL = 1000
RISK_PER_TRADE = 0.02  
ATR_MULT = 3.0  # Réduit à 3.0 (vs 3.3) pour couper pertes plus vite
MIN_QUALITY = 2  # Score minimum pour filtrer les actifs

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

# ============================================================
# INDICATEURS OPTIMISÉS
# ============================================================

def calculate_rsi(series, period=14):
    """RSI standard"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_adx_vectorized(high, low, close, period=14):
    """ADX vectorisé pour performance"""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    return dx.rolling(period).mean()

def quality_score_fast(prices, spy):
    """Score de qualité rapide (0-4)"""
    scores = pd.Series(0, index=prices.columns)
    
    # 1. Tendance (Prix > MA50 > MA200)
    ma50 = prices.rolling(50).mean().iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    current = prices.iloc[-1]
    scores += ((current > ma50) & (ma50 > ma200)).astype(int)
    
    # 2. Momentum consistant (1M, 3M, 6M tous positifs)
    ret_1m = (prices.iloc[-1] / prices.iloc[-21] - 1)
    ret_3m = (prices.iloc[-1] / prices.iloc[-63] - 1)
    ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1)
    scores += ((ret_1m > 0) & (ret_3m > 0) & (ret_6m > 0)).astype(int)
    
    # 3. Force relative vs SPY (6M)
    spy_ret_6m = spy.iloc[-1] / spy.iloc[-126] - 1
    rel_strength = ret_6m / spy_ret_6m
    scores += (rel_strength > 1.0).astype(int)
    
    # 4. Stabilité (pas de gap brutal récent)
    max_gap = prices.pct_change().tail(20).abs().max()
    scores += (max_gap < 0.10).astype(int)
    
    return scores

# ============================================================
# DÉTECTION DE RÉGIME
# ============================================================

def detect_regime(spy, vix, tnx, irx):
    """Régime de marché pondéré avec 4 facteurs"""
    
    # Facteur 1: Position vs MA200 (40%)
    spy_ma200 = spy.rolling(200).mean()
    f1 = 0.4 if float(spy.iloc[-1]) > float(spy_ma200.iloc[-1]) else 0.0
    
    # Facteur 2: VIX vs MA50 (30%)
    vix_ma50 = vix.rolling(50).mean()
    f2 = 0.3 if float(vix.iloc[-1]) < float(vix_ma50.iloc[-1]) else 0.0
    
    # Facteur 3: Momentum 3M (20%)
    ret_3m = float(spy.iloc[-1] / spy.iloc[-63] - 1)
    f3 = 0.2 if ret_3m > 0 else 0.0
    
    # Facteur 4: Courbe des taux (10%)
    try:
        curve = float(tnx.iloc[-1]) - float(irx.iloc[-1])
        f4 = 0.1 if curve > 0 else 0.0
    except:
        f4 = 0.05  # Neutre si données manquantes
    
    score = f1 + f2 + f3 + f4
    
    # Exposition et régime
    if score >= 0.70:
        return 1.00, "🟢🟢🟢 MAX BULL", score
    elif score >= 0.55:
        return 0.80, "🟢🟢 STRONG", score
    elif score >= 0.40:
        return 0.60, "🟢 BULL", score
    elif score >= 0.25:
        return 0.35, "🟡 NEUTRAL", score
    elif score >= 0.15:
        return 0.15, "🟠 CAUTIOUS", score
    else:
        return 0.00, "🔴 BEAR", score

# ============================================================
# SÉLECTION ET POSITION SIZING
# ============================================================

def select_and_size_positions(active_prices, high, low, spy, exposure, capital, n_positions):
    """Sélection + sizing optimisé"""
    
    # 1. Calcul des scores
    mom_6m = active_prices.pct_change(126).iloc[-1]
    spy_ret = spy.pct_change(126).iloc[-1]
    rel_strength = mom_6m / spy_ret
    
    # Z-scores
    z_mom = (mom_6m - mom_6m.mean()) / mom_6m.std()
    z_rs = (rel_strength - rel_strength.mean()) / rel_strength.std()
    
    # Score qualité
    q_scores = quality_score_fast(active_prices, spy)
    
    # Score composite
    final_scores = 0.50 * z_mom + 0.30 * z_rs + 0.20 * (q_scores / 4.0)
    
    # 2. Filtres stricts
    rsi = active_prices.apply(calculate_rsi).iloc[-1]
    ma150 = active_prices.rolling(150).mean().iloc[-1]
    
    # ADX vectorisé
    adx = pd.Series(index=active_prices.columns, dtype=float)
    for ticker in active_prices.columns:
        try:
            adx[ticker] = float(calculate_adx_vectorized(
                high[ticker], low[ticker], active_prices[ticker]
            ).iloc[-1])
        except:
            adx[ticker] = 0
    
    # Validation
    valid = (
        (final_scores > 0) &
        (rsi < 75) &
        (rsi > 30) &
        (active_prices.iloc[-1] > ma150) &
        (q_scores >= MIN_QUALITY) &
        (adx > 20) &
        (mom_6m > 0)
    )
    
    candidates = final_scores[valid].nlargest(n_positions * 2)
    
    if len(candidates) == 0:
        return []
    
    selected = list(candidates.nlargest(n_positions).index)
    
    # 3. Position sizing avec ATR
    positions = []
    
    for ticker in selected:
        price = float(active_prices[ticker].iloc[-1])
        
        # ATR
        tr = pd.concat([
            high[ticker] - low[ticker],
            abs(high[ticker] - active_prices[ticker].shift(1)),
            abs(low[ticker] - active_prices[ticker].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Stop loss
        sl_price = price - (ATR_MULT * atr)
        sl_distance = price - sl_price
        
        if sl_distance <= 0:
            continue
        
        # Sizing basé risque
        risk_amount = capital * RISK_PER_TRADE
        shares = risk_amount / sl_distance
        position_value = shares * price
        weight = min(position_value / capital, 0.30 if n_positions <= 3 else 0.20)
        
        positions.append({
            'ticker': ticker,
            'weight': weight,
            'price': price,
            'sl': sl_price,
            'atr': atr,
            'rsi': float(rsi[ticker]),
            'quality': int(q_scores[ticker])
        })
    
    # Normalisation pour respecter exposition
    total_weight = sum(p['weight'] for p in positions)
    if total_weight > 0:
        for p in positions:
            p['weight'] = (p['weight'] / total_weight) * exposure
    
    return positions

# ============================================================
# MAIN
# ============================================================

def run():
    print(f"🚀 APEX v25.3 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 1. Téléchargement
    try:
        tickers = ALL_TICKERS + [MARKET_INDEX, "EURUSD=X", "^VIX", "^TNX", "^IRX"]
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        
        if data.empty:
            print("❌ Erreur: Aucune donnée")
            return
        
        close = data['Close'].ffill().bfill() if 'Close' in data else data.ffill().bfill()
        high = data['High'].ffill().bfill() if 'High' in data else close
        low = data['Low'].ffill().bfill() if 'Low' in data else close
        
    except Exception as e:
        print(f"❌ Erreur data: {e}")
        return
    
    # 2. Préparation
    spy = close[MARKET_INDEX]
    vix = close["^VIX"]
    tnx = close["^TNX"]
    irx = close["^IRX"]
    fx = 1 / float(close["EURUSD=X"].iloc[-1]) if "EURUSD=X" in close.columns else 1.0
    
    # 3. Détection régime
    exposure, regime, score = detect_regime(spy, vix, tnx, irx)
    
    print(f"📊 {regime} | Score: {score:.2f} | Exposition: {exposure*100:.0f}%")
    
    # 4. Si bear market
    if exposure == 0.0:
        msg = (f"🤖 APEX v25.3\n"
               f"{regime} | Expo: **0%**\n"
               f"💰 Capital: **{TOTAL_CAPITAL}€**\n"
               f"━━━━━━━━━━━━━━\n"
               f"⚠️ **100% CASH** - Attente marché favorable")
        
        if TOKEN and CHAT_ID:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
                timeout=10
            )
        return
    
    # 5. Sélection univers
    universe = ALL_TICKERS if exposure >= 0.50 else DEFENSIVE_TICKERS
    active_prices = close[universe].dropna(axis=1, how='any')
    active_high = high[universe]
    active_low = low[universe]
    
    # 6. Message multi-top
    msg = (f"🤖 APEX v25.3 | {datetime.now().strftime('%d/%m/%Y')}\n"
           f"**{regime}** | Expo: **{int(exposure*100)}%** | Score: {score:.2f}\n"
           f"💰 Capital: **{TOTAL_CAPITAL}€** | 🛡️ SL: {ATR_MULT:.1f} ATR\n"
           f"━━━━━━━━━━━━━━━━━━\n\n")
    
    # Générer plusieurs tops
    for n in [2, 3, 6, 8]:
        # Affiche tous les TOPs peu importe le capital (pour analyse)
        # if TOTAL_CAPITAL < 1500 and n > 3:
        #     continue
        
        positions = select_and_size_positions(
            active_prices, active_high, active_low, spy, 
            exposure, TOTAL_CAPITAL, n
        )
        
        if not positions:
            continue
        
        # Label recommandé
        is_recommended = (
            (n == 2 and TOTAL_CAPITAL < 1500) or
            (n == 3 and 1500 <= TOTAL_CAPITAL < 3000) or
            (n == 6 and 3000 <= TOTAL_CAPITAL < 6000) or
            (n == 8 and TOTAL_CAPITAL >= 6000)
        )
        
        label = "⭐ RECOMMANDÉ" if is_recommended else "🔹 Alternative"
        msg += f"🏆 **TOP {n}** | {label}\n"
        
        for p in positions:
            price_eur = p['price'] * (1 if p['ticker'].endswith(".PA") else fx)
            sl_eur = p['sl'] * (1 if p['ticker'].endswith(".PA") else fx)
            alloc = TOTAL_CAPITAL * p['weight']
            sl_pct = ((price_eur - sl_eur) / price_eur) * 100
            
            msg += f"• **{p['ticker']}**: {p['weight']*100:.1f}% ({alloc:.0f}€)\n"
            msg += f"  Prix: {price_eur:.2f}€ | SL: {sl_eur:.2f}€ (-{sl_pct:.1f}%)\n"
            msg += f"  RSI: {p['rsi']:.0f} | Qualité: {p['quality']}/4\n"
        
        msg += "\n"
    
    msg += ("💡 Changez seulement si un titre entre/sort du TOP\n"
            "⚡ Discipline > Émotion | Coupez vos pertes rapidement")
    
    # 7. Envoi
    if TOKEN and CHAT_ID:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Notification envoyée")
            else:
                print(f"❌ Erreur Telegram: {response.text}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    print("✅ Analyse terminée")

if __name__ == "__main__":
    run()
