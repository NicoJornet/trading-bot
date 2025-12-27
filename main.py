import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ============================================================
# APEX v25.2.2 — OPTIMISÉ FRAIS (BALISES NOUVEAU/MAINTENU)
# ============================================================

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOTAL_CAPITAL = 1000
RISK_PER_TRADE = 0.02  
ATR_MULT = 3.3         

# ... (garder les listes TICKERS identiques) ...

def run():
    # ... (garder les calculs de régime et sélection identiques) ...
    
    msg = f"🤖 APEX v25.2.2 | {regime} ({int(exposure*100)}%)\n💰 Cap: {TOTAL_CAPITAL}€ | 🛡️ SL: {ATR_MULT} ATR\n━━━━━━━━━━━━━━\n\n"

    # Liste pour suivre les titres du TOP 8 global (pour comparer les entrées/sorties)
    all_candidates = mom[valid].nlargest(8).index.tolist()

    for n in [2, 3, 6, 8]:
        selected = all_candidates[:n]
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
            # Note: Ici on pourrait comparer avec le fichier de la veille pour mettre "✨ NOUVEAU"
            msg += f"• **{t}**: {final_w*100:.1f}% ({TOTAL_CAPITAL*final_w:.0f}€)\n"
            msg += f"  Prix: {p_eur:.2f}€ | **SL: {sl_eur:.2f}€ (-{sl_pct:.1f}%)**\n"
        msg += "──────────────\n"

    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg + "💡 Ne changez de position que si un titre entre ou sort du TOP.", "parse_mode": "Markdown"})

if __name__ == "__main__": run()
