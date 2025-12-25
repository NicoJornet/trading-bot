import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import logging

# ============================================================
# CONFIGURATION - APEX v17.3 ABSOLUTE MOMENTUM
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Selection Parameters
TOP_MAX = 3
MAX_CANDIDATES = 6

# Lookback Periods (in trading days)
MOMENTUM_PERIOD_1M = 21
MOMENTUM_PERIOD_3M = 63
MOMENTUM_PERIOD_6M = 126
MOMENTUM_PERIOD_12M = 252
MA_TREND_PERIOD = 150
RSI_PERIOD = 14
CORRELATION_PERIOD = 126

# Risk Parameters
MAX_CRYPTO_WEIGHT = 0.35
MAX_SINGLE_POSITION = 0.40
CORRELATION_THRESHOLD = 0.75
ATR_PERIOD = 14
ATR_MULTIPLIER = 4

# VIX Thresholds
VIX_CALM = 25
VIX_ALERT = 35
RSI_OVERBOUGHT = 80

# Tickers
TICKERS = [
    "NVDA","MSFT","GOOGL","AAPL","TSLA","SMH","PLTR","ASML",
    "BTC-USD","ETH-USD","SOL-USD",
    "MC.PA","RMS.PA","RACE",
    "LLY","UNH","ISRG","PANW",
    "URNM","COPX","XLE","ALB","GLD","SIL","ITA"
]
GLOBAL_PROTECT = "^VIX"

CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "SOL-USD"}


def send_telegram_message(message):
    """Send message via Telegram Bot API"""
    if not TOKEN or not CHAT_ID:
        logging.warning("Telegram credentials not configured. Skipping message send.")
        return False

    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Telegram message sent successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return False


def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate RSI with robust handling of edge cases"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr_stop(high, low, close, ticker, fx, period=ATR_PERIOD, multiplier=ATR_MULTIPLIER):
    """Calculate ATR-based stop loss"""
    try:
        tr = np.maximum(
            high[ticker] - low[ticker],
            np.maximum(
                abs(high[ticker] - close[ticker].shift(1)),
                abs(low[ticker] - close[ticker].shift(1))
            )
        )
        atr = tr.rolling(period).mean().iloc[-1]
        current_price = close[ticker].iloc[-1]
        stop_price = current_price - (multiplier * atr)

        # Convert to EUR if needed
        conversion = 1 if ticker.endswith(".PA") else fx
        return stop_price * conversion
    except Exception as e:
        logging.warning(f"Failed to calculate ATR stop for {ticker}: {e}")
        return None


def calculate_momentum_score(prices):
    """Calculate weighted absolute momentum score"""
    m1 = prices / prices.shift(MOMENTUM_PERIOD_3M) - 1
    m2 = prices / prices.shift(MOMENTUM_PERIOD_6M) - 1
    m3 = prices / prices.shift(MOMENTUM_PERIOD_12M) - 1

    momentum = 0.2 * m1 + 0.3 * m2 + 0.5 * m3
    return momentum


def apply_position_limits(weights):
    """Apply crypto and single position weight limits"""
    adjusted = weights.copy()

    # Cap crypto exposure
    crypto_weight = adjusted[adjusted.index.isin(CRYPTO_TICKERS)].sum()
    if crypto_weight > MAX_CRYPTO_WEIGHT:
        scale_factor = MAX_CRYPTO_WEIGHT / crypto_weight
        for ticker in adjusted.index:
            if ticker in CRYPTO_TICKERS:
                adjusted[ticker] *= scale_factor
        logging.info(f"Crypto exposure capped: {crypto_weight:.1%} -> {MAX_CRYPTO_WEIGHT:.1%}")

    # Cap individual positions
    for ticker in adjusted.index:
        if adjusted[ticker] > MAX_SINGLE_POSITION:
            logging.info(f"Position {ticker} capped: {adjusted[ticker]:.1%} -> {MAX_SINGLE_POSITION:.1%}")
            adjusted[ticker] = MAX_SINGLE_POSITION

    # Renormalize to sum to original total
    total_target = weights.sum()
    current_total = adjusted.sum()
    if current_total > 0:
        adjusted = adjusted * (total_target / current_total)

    return adjusted


def run():
    try:
        logging.info("Starting APEX v17.3 Absolute Momentum strategy...")
        raw_data = yf.download(TICKERS + [GLOBAL_PROTECT, "EURUSD=X"],
                               period="2y", auto_adjust=True, progress=False)
        close = raw_data['Close'].ffill()
        high = raw_data['High'].ffill()
        low = raw_data['Low'].ffill()
    except Exception as e:
        logging.error(f"Failed to download data: {e}")
        return

    # Get FX conversion rate and VIX level
    fx = 1 / close["EURUSD=X"].iloc[-1] if "EURUSD=X" in close.columns else 1.0
    vix = close[GLOBAL_PROTECT].iloc[-1]
    logging.info(f"VIX Level: {vix:.2f}, FX Rate (USD->EUR): {fx:.4f}")

    # --- 1. SÉLECTION PAR MOMENTUM ABSOLU ---
    prices = close[TICKERS]

    # Calculate momentum score
    m = calculate_momentum_score(prices)

    # Normalize momentum scores (z-score)
    z_mom = (m - m.mean(axis=1).values.reshape(-1, 1)) / m.std(axis=1).values.reshape(-1, 1).clip(min=0.001)

    # FILTRE INDIVIDUEL : L'actif doit être en tendance haussière PROPRE
    # Il doit être au-dessus de sa MA et avoir un RSI < 80 (pas de surchauffe)
    ma_trend = prices.rolling(MA_TREND_PERIOD).mean()
    rsi = calculate_rsi(prices, RSI_PERIOD)

    # Critères de sélection
    valid_mask = (
        (prices.iloc[-1] > ma_trend.iloc[-1]) &
        (z_mom.iloc[-1] > 0) &
        (rsi.iloc[-1] < RSI_OVERBOUGHT) &
        (prices.count() >= MOMENTUM_PERIOD_12M)  # Ensure sufficient data
    )
    candidates = z_mom.iloc[-1][valid_mask].nlargest(MAX_CANDIDATES)
    logging.info(f"Found {len(candidates)} valid candidates after filtering")

    # Correlation-based selection
    selected = []
    if len(candidates) > 0:
        returns = prices.pct_change()

        # Pre-compute correlation matrix for efficiency
        corr_data = returns[candidates.index].iloc[-CORRELATION_PERIOD:]
        corr_matrix = corr_data.corr()

        for t in candidates.index:
            if not selected:
                selected.append(t)
            else:
                # Check correlation with already selected assets
                max_corr = corr_matrix.loc[t, selected].max()
                if max_corr < CORRELATION_THRESHOLD:
                    selected.append(t)
                else:
                    logging.info(f"Skipping {t} due to high correlation ({max_corr:.2f}) with selected assets")

            if len(selected) == TOP_MAX:
                break

    logging.info(f"Final selected assets: {selected}")

    # --- 2. GESTION DE L'EXPOSITION GLOBALE ---
    # L'exposition n'est plus dictée par le SPY mais par le VIX (Panique)
    # Si VIX < VIX_CALM : 100% de ce que l'algo trouve
    # Si VIX > VIX_CALM : On réduit à 50% (Prudence)
    # Si VIX > VIX_ALERT : On réduit à 0% (Krach éclair)
    if vix < VIX_CALM:
        exposure = 1.0
    elif vix < VIX_ALERT:
        exposure = 0.5
    else:
        exposure = 0.0

    logging.info(f"VIX-based exposure: {exposure:.0%}")

    # --- 3. MESSAGE ---
    msg = f"🚀 **APEX v17.3 ABSOLUTE**\n"
    msg += f"📊 VIX: {vix:.2f} | Exposition: {exposure:.0%}\n"
    msg += f"⚠️ Risque Marché: {'🟢 CALME' if vix < VIX_CALM else '🟡 ALERTE' if vix < VIX_ALERT else '🔴 PANIQUE'}\n\n"

    if selected and exposure > 0:
        # Calculate base weights
        weights = pd.Series(1.0/len(selected), index=selected) * exposure

        # Apply position limits (crypto cap + single position cap)
        weights = apply_position_limits(weights)

        msg += "✅ **SIGNAUX ACTIFS :**\n"
        for t in selected:
            p_eur = prices[t].iloc[-1] * (1 if t.endswith(".PA") else fx)

            # Calculate ATR-based stop loss
            stop = calculate_atr_stop(high, low, close, t, fx)
            stop_str = f"{stop:.2f}€" if stop else "N/A"

            crypto_tag = " 🪙" if t in CRYPTO_TICKERS else ""
            msg += f"🔹 **{t}**{crypto_tag}\n"
            msg += f"   Alloc: **{weights[t]*100:.1f}%** | Prix: {p_eur:.2f}€ | Stop: {stop_str}\n\n"
    else:
        msg += "🛑 **ATTENTE** : Aucun actif en tendance saine ou panique VIX."

    # Send message via Telegram
    logging.info(f"Generated message:\n{msg}")
    send_telegram_message(msg)


if __name__ == "__main__":
    run()
