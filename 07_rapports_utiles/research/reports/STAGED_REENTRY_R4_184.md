# Staged Reentry R4 184

Promoted system upgrade after `r3`: keep long-term leader pullback reentry, but allow it only when both conditions hold:

- market breadth `>= 0.50`
- candidate `rs63 >= 0.01`

Retained configuration:

- `reentry_enable = 1`
- `reentry_bonus = 0.15`
- `reentry_dd60_min = 0.05`
- `reentry_dd60_max = 0.25`
- `reentry_r21_max = 0.03`
- `reentry_lt_topn = 12`
- `reentry_breadth_min = 0.50`
- `reentry_rs63_min = 0.01`

Why it is retained:

- it keeps the `r3` pullback reentry edge
- it repairs the weak-year tax paid by unconditional reentry in `2018` and `2019`
- it beats both `r2` and `r3` on the official staged replay

Official staged replay metrics on `2026-04-02`:

- Full `2015-01-02 -> 2026-04-02`
  - `ROI 556046.31%`
  - `CAGR 115.27%`
  - `MaxDD -45.04%`
  - `Sharpe 2.02965`
  - `Orders 550`

- OOS `2022-01-03 -> 2026-04-02`
  - `ROI 11315.42%`
  - `CAGR 205.38%`
  - `MaxDD -35.15%`
  - `Sharpe 2.78489`
  - `Orders 234`

Delta vs `r2`:

- full:
  - `ROI +188091.9964`
  - `Sharpe +0.0704`
  - `MaxDD +0.0469`
- OOS:
  - `ROI +722.2486`
  - `Sharpe +0.0268`
  - `MaxDD -0.0121`

Year behaviour vs `r2`:

- `2018`: `ROI +2.0277`, `Sharpe +0.0371`, `MaxDD +0.3815`
- `2019`: `ROI +0.0205`, `Sharpe +0.0008`, `MaxDD +0.0000`
- `2024`: `ROI +34.8855`, `Sharpe +0.1606`, `MaxDD +4.7710`

Interpretation:

- breadth alone was not enough
- candidate-level relative strength confirmation was the missing permission layer
- the new edge is not “more reentry”, but “reentry only when both the market and the candidate still deserve it”
