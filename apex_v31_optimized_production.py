Run python apex_v31_optimized_production.py
Traceback (most recent call last):
==========================================================================================
APEX CHAMPION — PROD (YFINANCE ONLY) — INDENTATION-PROOF
==========================================================================================
Date: 2026-01-29 | EURUSD 1.1978 | Universe 53
BREADTH: 64.15% (34/53) | Gate: PASS
  File "/home/runner/work/trading-bot/trading-bot/apex_v31_optimized_production.py", line 916, in <module>
    main()
  File "/home/runner/work/trading-bot/trading-bot/apex_v31_optimized_production.py", line 589, in main
    cm = corr_matrix(df, UNIVERSE_U54, CORR_WINDOW)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/trading-bot/trading-bot/apex_v31_optimized_production.py", line 421, in corr_matrix
    if r_windowed.notna().sum() < window // 2:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/generic.py", line 1513, in __bool__
    raise ValueError(
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
Error: Process completed with exit code 1.
