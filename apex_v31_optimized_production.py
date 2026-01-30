# File: APEX_CHAMPION_PROD_GITHUB_MIN.py
from __future__ import annotations

"""
APEX v33.x — PROD (GitHub) — Champion-params (MIN DIFF)
======================================================

⚠️ Objectif
- Repartir du script PROD V33 qui fonctionne, et ne changer que le minimum.
- Garder la plomberie: portfolio.json, trades_history.json, Telegram optionnel, parquet>yfinance.
- Passer sur les paramètres "algo champion" demandés.

Changements (diff checklist)
- FULLY_INVESTED: plus d'allocation 50/30/20 => allocation equal-split du cash dispo sur les slots restants.
- Rotation portfolio (prod):
    MAX_POSITIONS=3
    EDGE_MULT=1.00 (non utilisé ici: pas de swap edge dans le script prod; laissé en constante pour traçabilité)
    CONFIRM_DAYS=3 (achat seulement après N confirmations consécutives en top ranks)
    COOLDOWN_DAYS=1 (interdit de racheter un ticker pendant N jours après une vente)_
