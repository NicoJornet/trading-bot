import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ============================================================
# APEX v23.3 — FULL SPECTRUM (Reactive Mode) — Version mise à jour
# ============================================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- 1. L'UNIVERS D'INVESTISSEMENT (Mis à jour) ---
# Les Attaquants (Tech, IA, Crypto, Space, Cyber) — renforcés
OFFENSIVE_TICKERS = [
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    "ASML", "AVGO", "SMH", "VRT", "PLTR", "PANW", "RKLB",
    "CRWD",      # Nouveau : Cyber-sécurité IA leader
    "SMCI",      # Nouveau : Serveurs AI / data centers
    "ARM",       # Nouveau : Architecture chips AI/mobile
    "APP",       # Nouveau : AI advertising & gaming
    "BTC-USD", "ETH-USD", "SOL-USD"
]

# Les Défenseurs (Santé, Énergie, Matières, Luxe) — inchangés
DEFENSIVE_TICKERS = [
    "LLY", "UNH", "ISRG",
    "ETN", "URNM", "XLE",
    "COPX", "SIL", "REMX",
    "GLD", "ITA", "RACE", "MC.PA"
]

ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"
TOP_MAX = 3

# Paramètres de Risque
MAX_CRYPTO_ALLOC = 0.20
MAX_SINGLE_POS = 0.40


def run():
    print("\n" + "="*50)
    print("🌍 APEX v23.3 — PRODUCTION RUN (Univers mis à jour)")
    print("="*50)

    # --- 1. DATA LOADING ---
    try:
        # Téléchargement optimisé (2 ans de données)
        raw = yf.download(ALL_TICKERS + [MARKET_INDEX, "EURUSD=X"], period="2y", auto_adjust=True, progress=False)
        if raw.empty:
            return

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].ffill()
            high = raw["High"].ffill()
            low = raw["Low"].ffill()
        else:
            close = raw["Close"].ffill()
            high = raw["High"].ffill()
            low = raw["Low"].ffill()
    except Exception as e:
        print(f"Data Error: {e}")
        return

    prices = close[ALL_TICKERS]
    spy = close[MARKET_INDEX]
    fx = 1 / close["EURUSD=X"].iloc[-1] if "EURUSD=X" in close.columns else 1.0

    # --- 2. DÉTECTION DU RÉGIME (Le Sniper) ---
    ma200 = spy.rolling(200).mean()

    # Condition 1 : Prix > MA200
    # Condition 2 : Pente MA200 positive
    spy_bullish = (spy.iloc[-1] > ma200.iloc[-1]) and (ma200.iloc[-1] > ma200.iloc[-20])

    if spy_bullish:
        hunting_ground = ALL_TICKERS
        regime_icon = "🟢"
        regime_msg = "BULL (Chasse Totale)"
    else:
        hunting_ground = DEFENSIVE_TICKERS
        regime_icon = "🔴"
        regime_msg = "BEAR (Repli Défensif)"

    # --- 3. CALCUL DES SCORES (Sur Zone de Chasse) ---
    active_prices = prices[hunting_ground]

    # Momentum pondéré (Court, Moyen, Long terme)
    m = (0.2 * (active_prices / active_prices.shift(63) - 1) +
         0.3 * (active_prices / active_prices.shift(126) - 1) +
         0.5 * (active_prices / active_prices.shift(252) - 1))

    # Z-Score momentum
    z_mom = (m - m.mean(axis=1).values.reshape(-1, 1)) / m.std(axis=1).values.reshape(-1, 1).clip(0.001)

    # Force Relative vs SPY
    rs = (active_prices / active_prices.shift(126)) / (spy / spy.shift(126)).values.reshape(-1, 1)
    rs_z = (rs - rs.mean(axis=1).values.reshape(-1, 1)) / rs.std(axis=1).values.reshape(-1, 1).clip(0.001)

    # Score Composite Final
    score = z_mom.iloc[-1] + (rs_z.iloc[-1] * 0.5)

    # Filtres de Qualité
    valid = (z_mom.iloc[-1] > 0) & (rs_z.iloc[-1] > 0)

    # Sélection préliminaire (Top 8)
    candidates = score[valid].nlargest(8)

    # --- 4. SÉLECTION FINALE (Diversification) ---
    selected = []

    for t in candidates.index:
        if not selected:
            selected.append(t)
        else:
            # Corrélation max sur 3 mois
            corr = active_prices[selected + [t]].pct_change().iloc[-63:].corr().iloc[-1, :-1].max()
            if corr < 0.85:
                selected.append(t)

        if len(selected) == TOP_MAX:
            break

    # --- 5. ALLOCATION (Risk Parity) ---
    msg = f"🤖 **APEX v23.3** {regime_icon}\n"
    msg += f"🌍 Régime: {regime_msg}\n"

    if not selected:
        msg += "\n🛑 **MODE CASH (100%)**\n"
        if spy_bullish:
            msg += "Marché haussier mais aucun leader détecté (Rotation).\n"
        else:
            msg += "Marché baissier et aucun refuge ne tient.\n"
    else:
        # Volatilité annualisée sur 6 mois
        vols = active_prices[selected].pct_change().iloc[-126:].std() * np.sqrt(252)
        vols = vols.clip(lower=0.15)

        weights = (1 / vols) / (1 / vols).sum()

        # Cap Crypto
        crypto_sel = [t for t in selected if "USD" in t]
        if crypto_sel and weights[crypto_sel].sum() > MAX_CRYPTO_ALLOC:
            weights[crypto_sel] *= MAX_CRYPTO_ALLOC / weights[crypto_sel].sum()
            remaining = 1.0 - weights[crypto_sel].sum()
            others = [t for t in selected if t not in crypto_sel]
            if others:
                weights[others] = weights[others] / weights[others].sum() * remaining

        # Cap Single Position
        weights = weights.clip(upper=MAX_SINGLE_POS)
        weights /= weights.sum()  # Renormalisation

        msg += "\n🚀 **TOP 3 ACTIFS :**\n"
        for t in selected:
            # Prix en EUR
            p = prices[t].iloc[-1] * (1 if t.endswith(".PA") else fx)

            # Stop Loss : 4 ATR
            tr = np.maximum(high[t] - low[t],
                            np.maximum(abs(high[t] - close[t].shift(1)),
                                       abs(low[t] - close[t].shift(1))))
            atr = tr.rolling(14).mean().iloc[-1]
            stop_price = prices[t].iloc[-1] - (4.0 * atr)
            stop_eur = stop_price * (1 if t.endswith(".PA") else fx)
            dist_stop = (p - stop_eur) / p * 100

            # Icônes thématiques
            if t in ["ASML", "NVDA", "MSFT", "SMH", "GOOGL", "META", "AMZN", "AVGO", "ARM", "SMCI", "APP"]: 
                icon = "🧠"  # IA / Tech core
            elif t in ["VRT", "ETN"]: 
                icon = "🔌"  # Infrastructure / Énergie
            elif t in ["PANW", "PLTR", "CRWD"]: 
                icon = "🔒"  # Cyber
            elif t in ["RKLB"]: 
                icon = "🌌"  # Space
            elif "USD" in t: 
                icon = "🪙"  # Crypto
            elif t in ["SIL", "COPX", "REMX", "GLD"]: 
                icon = "💎"  # Métaux précieux/industriels
            elif t in ["URNM", "XLE"]: 
                icon = "⚡"  # Énergie/Uranium
            else: 
                icon = "🛡️"

            msg += f"{icon} **{t}**\n"
            msg += f" 📊 Alloc: **{weights[t]*100:.1f}%**\n"
            msg += f" 💰 Prix: {p:.2f}€\n"
            msg += f" 🛡️ Stop: {stop_eur:.2f}€ (-{dist_stop:.1f}%)\n\n"

        msg += f"🔥 **Investi : {weights.sum()*100:.0f}%**"

    # Envoi Telegram
    if TOKEN and CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
            )
        except Exception as e:
            print(f"Telegram Error: {e}")

    print(msg)


if __name__ == "__main__":
    run()
