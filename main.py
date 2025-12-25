import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ============================================================
# APEX v23.2 — STRATEGIC MATERIALS (AI + Energy + Metals)
# ============================================================

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- 1. LES ATTAQUANTS (Offensive - Tech & Alpha) ---
OFFENSIVE_TICKERS = [
    # Magnificent 7 & Cloud
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    # Infrastructure AI (Hardware + Cooling + Space)
    "AVGO", "SMH", "VRT", "RKLB", "PLTR",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD"
]

# --- 2. LES DÉFENSEURS (Real Assets & Strategic) ---
DEFENSIVE_TICKERS = [
    # Santé (Stabilité)
    "LLY", "UNH", "ISRG",
    
    # Énergie IA (Le carburant des Data Centers)
    "URNM",                # Uranium (Nucléaire)
    "XLE",                 # Pétrole/Gaz
    
    # Métaux Stratégiques (Le squelette de l'IA)
    "COPX",                # Cuivre (Câblage/Réseau)
    "SIL",                 # Argent (Conductivité/Solaire)  <-- AJOUT
    "REMX",                # Terres Rares (Aimants/Robots)  <-- AJOUT
    
    # Valeurs Refuges
    "GLD",                 # Or
    "ITA",                 # Défense
    "RACE", "MC.PA"        # Luxe
]

ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"
TOP_MAX = 3

# Paramètres
MAX_CRYPTO_ALLOC = 0.20
MAX_SINGLE_POS = 0.40

def run():
    print("\n" + "="*50)
    print("💎 APEX v23.2 — STRATEGIC MATERIALS")
    print("="*50)

    # --- 1. DATA LOADING ---
    try:
        raw = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X"], period="2y", auto_adjust=True, progress=False)
        if raw.empty: return
        
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].ffill()
            high = raw["High"].ffill()
            low = raw["Low"].ffill()
        else:
            close = raw["Close"].ffill()
            high = raw["High"].ffill()
            low = raw["Low"].ffill()
    except: return

    prices = close[ALL_TICKERS]
    spy = close[MARKET_INDEX]
    fx = 1 / close["EURUSD=X"].iloc[-1] if "EURUSD=X" in close.columns else 1.0

    # --- 2. DÉTECTION DU RÉGIME ---
    ma200 = spy.rolling(200).mean()
    spy_bullish = (spy.iloc[-1] > ma200.iloc[-1]) and (ma200.iloc[-1] > ma200.iloc[-20])
    
    if spy_bullish:
        hunting_ground = ALL_TICKERS
        regime_msg = "🟢 BULL (Offensive + Strategic)"
    else:
        hunting_ground = DEFENSIVE_TICKERS
        regime_msg = "🔴 BEAR (Strategic/Defensive Only)"

    # --- 3. SÉLECTION ---
    active_prices = prices[hunting_ground]
    
    # Momentum (Z-Score)
    m = (0.2 * (active_prices/active_prices.shift(63)-1) + 
         0.3 * (active_prices/active_prices.shift(126)-1) + 
         0.5 * (active_prices/active_prices.shift(252)-1))
    z_mom = (m - m.mean(axis=1).values.reshape(-1,1)) / m.std(axis=1).values.reshape(-1,1).clip(0.001)
    
    # RS vs SPY
    rs = (active_prices/active_prices.shift(126)) / (spy/spy.shift(126)).values.reshape(-1,1)
    rs_z = (rs - rs.mean(axis=1).values.reshape(-1,1)) / rs.std(axis=1).values.reshape(-1,1).clip(0.001)
    
    # Score
    score = z_mom.iloc[-1] + (rs_z.iloc[-1] * 0.5)
    
    # Filtres
    valid = (z_mom.iloc[-1] > 0) & (rs_z.iloc[-1] > 0)
    candidates = score[valid].nlargest(TOP_MAX)
    
    selected = []
    for t in candidates.index:
        if not selected:
            selected.append(t)
        else:
            corr = active_prices[selected + [t]].pct_change().iloc[-63:].corr().iloc[-1, :-1].max()
            if corr < 0.80:
                selected.append(t)
        if len(selected) == TOP_MAX: break

    # --- 4. ALLOCATION ---
    msg = f"🤖 **APEX v23.2 — STRATEGIC**\n"
    msg += f"🌍 Régime: {regime_msg}\n"
    
    if not selected:
        msg += "\n🛑 **MODE CASH (100%)**\n"
        msg += "Aucun actif stratégique ne performe.\n"
    else:
        # Risk Parity
        vols = active_prices[selected].pct_change().iloc[-126:].std() * np.sqrt(252)
        vols = vols.clip(lower=0.15)
        weights = (1/vols) / (1/vols).sum()
        
        # Caps
        crypto_sel = [t for t in selected if "USD" in t]
        if crypto_sel and weights[crypto_sel].sum() > MAX_CRYPTO_ALLOC:
            weights[crypto_sel] *= MAX_CRYPTO_ALLOC / weights[crypto_sel].sum()
            
        weights = weights.clip(upper=MAX_SINGLE_POS)
        weights /= weights.sum()

        msg += "\n✅ **SÉLECTION ACTIVE :**\n"
        for t in selected:
            p = prices[t].iloc[-1] * (1 if t.endswith(".PA") else fx)
            tr = np.maximum(high[t]-low[t], np.maximum(abs(high[t]-close[t].shift(1)), abs(low[t]-close[t].shift(1))))
            stop = p - (4.0 * tr.rolling(14).mean().iloc[-1]) * (1 if t.endswith(".PA") else fx)
            
            # Iconographie
            if t in OFFENSIVE_TICKERS:
                icon = "🪙" if "USD" in t else "🚀"
            else:
                # Icones spécifiques Matières Premières
                if t in ["SIL", "GLD", "COPX", "REMX"]: icon = "💎"
                elif t in ["URNM", "XLE"]: icon = "⚡"
                else: icon = "🛡️"
            
            msg += f"{icon} **{t}**\n"
            msg += f"   📊 Alloc: {weights[t]*100:.1f}%\n"
            msg += f"   💰 Prix: {p:.2f}€ | Stop: {stop:.2f}€\n\n"
            
        msg += f"🔥 Investi: {weights.sum()*100:.0f}%"

    if TOKEN and CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 
                      data={"chat_id":CHAT_ID, "text":msg, "parse_mode":"Markdown"})
    print(msg)

if __name__ == "__main__":
    run()
