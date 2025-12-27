import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CONFIG
INITIAL_CAPITAL = 2000
MONTHLY_DCA = 150
START_DATE = "2024-01-01"
BACKTEST_START = "2025-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

COST_PER_BUY = 1.0
COST_PER_SELL = 1.0
TARGET_RISK_DAILY = 0.0125
MIN_QUALITY = 2
ROTATION_THRESHOLD = 0.20
MIN_DAYS_BETWEEN_REBALANCE = 1

# Univers
OFFENSIVE_TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "AAPL", "AVGO", "AMD", "QCOM", "MU", "CRWD", "PANW", "NET", "DDOG", "ZS", "ASML", "TSM", "LRCX", "AMAT", "KLAC", "TSLA", "PLTR", "RKLB", "ABNB", "SHOP", "QQQ", "SMH", "SOXX", "IGV", "VRT", "APP"]
DEFENSIVE_TICKERS = ["LLY", "UNH", "JNJ", "ABBV", "TMO", "DHR", "ISRG", "PG", "KO", "PEP", "WMT", "XLU", "NEE", "XLE", "GLD", "SLV", "DBA", "PDBC", "LMT", "RTX", "ITA"]
ALL_TICKERS = list(set(OFFENSIVE_TICKERS + DEFENSIVE_TICKERS))
MARKET_INDEX = "SPY"
NASDAQ_INDEX = "QQQ"

# --- FONCTIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr_series(high, low, close, period=14):
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_regime(spy, vix, idx):
    spy_ma200 = spy.rolling(200).mean()
    vix_ma50 = vix.rolling(50).mean()
    score = 0
    if idx >= 200 and float(spy.iloc[idx]) > float(spy_ma200.iloc[idx]): score += 0.4
    if idx >= 50 and float(vix.iloc[idx]) < float(vix_ma50.iloc[idx]): score += 0.3
    if idx >= 63 and float(spy.iloc[idx]) > float(spy.iloc[max(0, idx-63)]): score += 0.2
    score += 0.1
    return (1.00, "MAX") if score >= 0.70 else (0.80, "STRONG") if score >= 0.55 else (0.60, "BULL") if score >= 0.40 else (0.35, "NEUTRAL") if score >= 0.25 else (0.15, "CAUTIOUS") if score >= 0.15 else (0.00, "BEAR")

def quality_score_fast(prices, spy):
    scores = pd.Series(0, index=prices.columns)
    ma50 = prices.rolling(50).mean().iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    current = prices.iloc[-1]
    scores += ((current > ma50) & (ma50 > ma200)).astype(int)
    if len(prices) >= 126:
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

def select_positions_with_scores(prices, high, low, spy, n_positions, idx):
    if idx < 252: return [], {}
    prices_slice = prices.iloc[:idx+1]
    spy_slice = spy.iloc[:idx+1]
    mom_6m = prices_slice.pct_change(126).iloc[-1]
    spy_ret = spy_slice.pct_change(126).iloc[-1]
    if spy_ret == 0: spy_ret = 0.0001
    rel_strength = mom_6m / spy_ret
    z_mom = (mom_6m - mom_6m.mean()) / (mom_6m.std() + 1e-8)
    z_rs = (rel_strength - rel_strength.mean()) / (rel_strength.std() + 1e-8)
    q_scores = quality_score_fast(prices_slice, spy_slice)
    final_scores = 0.50 * z_mom + 0.30 * z_rs + 0.20 * (q_scores / 4.0)
    rsi = prices_slice.apply(calculate_rsi).iloc[-1]
    ma150 = prices_slice.rolling(150).mean().iloc[-1]
    valid = (final_scores > 0) & (rsi < 75) & (rsi > 30) & (prices_slice.iloc[-1] > ma150) & (q_scores >= MIN_QUALITY) & (mom_6m > 0)
    candidates = final_scores[valid].nlargest(n_positions * 2)
    if len(candidates) == 0: return [], {}
    selected = list(candidates.nlargest(min(n_positions, len(candidates))).index)
    return selected, {t: final_scores[t] for t in selected}

def should_rebalance(current_portfolio, new_selection, new_scores, old_scores, threshold):
    if not current_portfolio: return True
    if set(current_portfolio.keys()) == set(new_selection): return False
    tickers_to_add = set(new_selection) - set(current_portfolio.keys())
    if len(tickers_to_add) > 0 and len(old_scores) > 0:
        avg_old_score = np.mean(list(old_scores.values()))
        for t in tickers_to_add:
            if new_scores.get(t, 0) > avg_old_score * (1 + threshold): return True
        return False
    return True

def get_safe_weight(price, atr, n_positions, capital_total):
    if price <= 0: return 0
    volatility_pct = atr / price
    if volatility_pct <= 0: volatility_pct = 0.01
    max_weight_vol = TARGET_RISK_DAILY / volatility_pct
    ceiling = 1.0 / n_positions
    return min(max_weight_vol, ceiling)

# --- BACKTEST ---
def run_backtest(n_positions, data_pack):
    print(f"🛡️ Simulation TOP {n_positions} SAFE en cours...")
    
    prices = data_pack['prices']; high = data_pack['high']; low = data_pack['low']; close = data_pack['close']
    spy = data_pack['spy']; vix = data_pack['vix']; atr_df = data_pack['atr']; dates = data_pack['dates']
    
    capital = INITIAL_CAPITAL
    portfolio = {} 
    last_scores = {}
    last_rebalance_idx = -999
    last_exposure = 0
    last_dca_month = None
    
    trade_log = []
    total_sl = 0
    cumulative_pnl = 0.0 
    
    for date in dates:
        idx = close.index.get_loc(date)
        current_prices = prices.loc[date]
        current_lows = low.loc[date]
        
        current_port_val = sum(portfolio[t]['shares'] * current_prices[t] for t in portfolio if t in current_prices)

        # DCA
        if last_dca_month != (date.year, date.month):
            capital += MONTHLY_DCA
            trade_log.append({
                'Date': date.strftime('%Y-%m-%d'), 'Action': '💰 DEPOT', 'Ticker': 'CASH', 
                'Détail': '-', 'Mvt': f"+{MONTHLY_DCA}€", 'Cumul P&L': f"{cumulative_pnl:+.0f}€",
                'Portf. Total': f"{(capital + current_port_val + MONTHLY_DCA):.0f}€"
            })
            last_dca_month = (date.year, date.month)
            
        # 1. Check SL
        if len(portfolio) > 0:
            for t in list(portfolio.keys()):
                if t in current_lows:
                    info = portfolio[t]
                    if current_lows[t] <= info['sl']:
                        exit_price = info['sl']
                        val = info['shares'] * exit_price
                        capital += val - COST_PER_SELL
                        pnl = val - (info['shares'] * info['entry']) - COST_PER_SELL - COST_PER_BUY
                        cumulative_pnl += pnl
                        current_port_val = sum(portfolio[k]['shares'] * current_prices[k] for k in portfolio if k != t and k in current_prices)
                        trade_log.append({
                            'Date': date.strftime('%Y-%m-%d'), 'Action': '🔴 STOP LOSS', 'Ticker': t, 
                            'Détail': f"{exit_price:.2f}", 'Mvt': f"{pnl:+.0f}€", 'Cumul P&L': f"{cumulative_pnl:+.0f}€",
                            'Portf. Total': f"{(capital + current_port_val):.0f}€"
                        })
                        total_sl += 1
                        del portfolio[t]

        # 2. Trading
        if idx >= 252:
            exposure, regime = detect_regime(spy, vix, idx)
            
            # Cash Out
            if exposure == 0 and len(portfolio) > 0:
                for t, info in list(portfolio.items()):
                    if t in current_prices:
                        val = info['shares'] * current_prices[t]
                        capital += val - COST_PER_SELL
                        pnl = val - (info['shares'] * info['entry']) - COST_PER_SELL - COST_PER_BUY
                        cumulative_pnl += pnl
                        trade_log.append({
                            'Date': date.strftime('%Y-%m-%d'), 'Action': '⚠️ CASH OUT', 'Ticker': t, 
                            'Détail': f"{current_prices[t]:.2f}", 'Mvt': f"{pnl:+.0f}€", 'Cumul P&L': f"{cumulative_pnl:+.0f}€",
                            'Portf. Total': f"{capital:.0f}€"
                        })
                portfolio = {}
                last_exposure = 0
            
            elif exposure > 0:
                days = idx - last_rebalance_idx
                if days >= MIN_DAYS_BETWEEN_REBALANCE:
                    universe = ALL_TICKERS if exposure >= 0.5 else DEFENSIVE_TICKERS
                    active_p = prices[universe]; active_h = high[universe]; active_l = low[universe]
                    
                    new_sel, new_scores = select_positions_with_scores(active_p, active_h, active_l, spy, n_positions, idx)
                    
                    if should_rebalance(portfolio, new_sel, new_scores, last_scores, ROTATION_THRESHOLD) or abs(exposure-last_exposure)>0.1:
                        last_rebalance_idx = idx
                        last_exposure = exposure
                        last_scores = new_scores.copy()
                        
                        # Vente
                        for t in list(portfolio.keys()):
                            if t not in new_sel:
                                if t in current_prices:
                                    val = portfolio[t]['shares'] * current_prices[t]
                                    capital += val - COST_PER_SELL
                                    pnl = val - (portfolio[t]['shares'] * portfolio[t]['entry']) - COST_PER_SELL - COST_PER_BUY
                                    cumulative_pnl += pnl
                                    current_port_val = sum(portfolio[k]['shares'] * current_prices[k] for k in portfolio if k != t and k in current_prices)
                                    trade_log.append({
                                        'Date': date.strftime('%Y-%m-%d'), 'Action': '🔄 ROTATION', 'Ticker': t, 
                                        'Détail': f"{current_prices[t]:.2f}", 'Mvt': f"{pnl:+.0f}€", 'Cumul P&L': f"{cumulative_pnl:+.0f}€",
                                        'Portf. Total': f"{(capital + current_port_val):.0f}€"
                                    })
                                del portfolio[t]
                        
                        # Achat
                        if len(new_sel) > 0:
                            investable_capital = capital 
                            for t in new_sel:
                                if t not in portfolio:
                                    if t in current_prices:
                                        p = current_prices[t]
                                        cur_atr = atr_df.loc[date, t]
                                        
                                        # MODE SAFE : CALCUL DU POIDS
                                        safe_w = get_safe_weight(p, cur_atr, n_positions, investable_capital)
                                        final_w = safe_w * exposure 
                                        alloc = investable_capital * final_w
                                        
                                        if alloc > COST_PER_BUY and capital > alloc:
                                            s = (alloc - COST_PER_BUY) / p
                                            sl = p - (3.0 * cur_atr) 
                                            portfolio[t] = {'shares': s, 'entry': p, 'sl': sl}
                                            capital -= alloc
                                            current_port_val = sum(portfolio[k]['shares'] * current_prices[k] for k in portfolio if k in current_prices)
                                            trade_log.append({
                                                'Date': date.strftime('%Y-%m-%d'), 'Action': '🟢 ACHAT SAFE', 'Ticker': t, 
                                                'Détail': f"Alloc {final_w*100:.0f}%", 'Mvt': '-', 'Cumul P&L': f"{cumulative_pnl:+.0f}€",
                                                'Portf. Total': f"{(capital + current_port_val):.0f}€"
                                            })

    final = capital + sum(portfolio[t]['shares'] * prices.iloc[-1][t] for t in portfolio if t in prices.iloc[-1])
    
    last_prices = prices.iloc[-1]
    for t, info in portfolio.items():
        if t in last_prices:
            curr_val = info['shares'] * last_prices[t]
            unrealized = curr_val - (info['shares'] * info['entry'])
            trade_log.append({'Date': 'FIN', 'Action': '💼 POS. OUVERTE', 'Ticker': t, 'Détail': f"Val {curr_val:.0f}€", 'Mvt': f"Lat: {unrealized:+.0f}€", 'Cumul P&L': '-', 'Portf. Total': '-'})

    return {'n': n_positions, 'final': final, 'log': trade_log, 'sl_count': total_sl}

if __name__ == "__main__":
    print(f"📥 Data...")
    tickers = ALL_TICKERS + [MARKET_INDEX, NASDAQ_INDEX, "^VIX"]
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        close = data['Close'].ffill().bfill(); high = data['High'].ffill().bfill(); low = data['Low'].ffill().bfill()
        atr_df = pd.DataFrame(index=close.index, columns=ALL_TICKERS)
        for t in ALL_TICKERS:
            if t in close.columns: atr_df[t] = calculate_atr_series(high[t], low[t], close[t])
            
        data_pack = {'prices': close[ALL_TICKERS].dropna(axis=1, how='all'), 'high': high, 'low': low, 'close': close, 'spy': close[MARKET_INDEX], 'vix': close["^VIX"], 'atr': atr_df, 'dates': close.index[close.index >= BACKTEST_START]}
        
        # Lancement TOP 3
        r = run_backtest(3, data_pack)
        
        invested = INITIAL_CAPITAL + (12*150)
        roi = ((r['final'] - invested)/invested)*100
        
        print("\n" + "="*115)
        print(f"📊 JOURNAL DE BORD 2025 (TOP 3 SAFE)")
        print(f"📈 Performance: {roi:+.2f}% | Capital Final: {r['final']:.2f}€ | Stop Loss: {r['sl_count']}")
        print("="*115)
        
        df = pd.DataFrame(r['log'])
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        print(df[['Date', 'Action', 'Ticker', 'Détail', 'Mvt', 'Cumul P&L', 'Portf. Total']].to_string(index=False))

    except Exception as e: print(e)
