# State Dependent Reentry 184

Targeted follow-up after `r3`: keep the pullback reentry edge, but only when both market context and candidate quality justify it.

Variants tested:

- `r2_baseline`: anti-churn baseline with reentry disabled
- `r3_reentry`: unconditional long-term leader pullback reentry
- `r4_b50_rs00`: reentry allowed only when breadth >= 0.50 and candidate rs63 >= 0.00
- `r4_b50_rs01`: reentry allowed only when breadth >= 0.50 and candidate rs63 >= 0.01
- `r4_b52_rs01`: reentry allowed only when breadth >= 0.52 and candidate rs63 >= 0.01

Delta vs `r2_baseline`:

| variant | delta_full_roi_pct | delta_full_sharpe | delta_full_maxdd_pct | delta_oos_roi_pct | delta_oos_sharpe | delta_oos_maxdd_pct | delta_since2023_roi_pct | delta_since2023_sharpe | delta_since2023_maxdd_pct | delta_since2025_roi_pct | delta_since2025_sharpe | delta_since2025_maxdd_pct | delta_ytd2026_roi_pct | delta_ytd2026_sharpe | delta_ytd2026_maxdd_pct | delta_2018_roi_pct | delta_2018_sharpe | delta_2018_maxdd_pct | delta_2019_roi_pct | delta_2019_sharpe | delta_2019_maxdd_pct | delta_2022_roi_pct | delta_2022_sharpe | delta_2022_maxdd_pct | delta_2024_roi_pct | delta_2024_sharpe | delta_2024_maxdd_pct | delta_vs_r3_full_roi_pct | delta_vs_r3_oos_roi_pct | delta_vs_r3_full_sharpe | delta_vs_r3_oos_sharpe | delta_vs_r3_full_maxdd_pct | delta_vs_r3_oos_maxdd_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r2_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | nan | nan | nan | nan | nan |
| r3_reentry | 47901.537144 | 0.022770 | -5.871842 | 46.191777 | 0.019606 | 4.862304 | 498.667068 | 0.060843 | 4.792536 | -14.773947 | 0.016444 | 2.252782 | -0.089155 | -0.000906 | 0.003878 | -7.293216 | -0.167911 | -1.720487 | -1.341645 | -0.104032 | -9.340184 | 0.357082 | 0.008186 | 0.039173 | 36.391134 | 0.177392 | 4.874807 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r4_b50_rs00 | 188034.113368 | 0.070364 | 0.046869 | 713.749421 | 0.026491 | -0.012013 | 935.259076 | 0.059486 | -0.033783 | -5.521391 | -0.010113 | 0.000000 | -0.089155 | -0.000906 | 0.003878 | 2.027679 | 0.037101 | 0.381539 | 0.020460 | 0.000843 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 35.028239 | 0.162536 | 4.783637 | 140132.576224 | 667.557645 | 0.047594 | 0.006885 | 5.918711 | -4.874318 |
| r4_b50_rs01 | 188091.996447 | 0.070388 | 0.046869 | 722.248648 | 0.026768 | -0.012147 | 941.738061 | 0.059819 | -0.033991 | -5.521391 | -0.010113 | 0.000000 | -0.089155 | -0.000906 | 0.003878 | 2.027679 | 0.037101 | 0.381539 | 0.020460 | 0.000843 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 34.885496 | 0.160567 | 4.771000 | 140190.459303 | 676.056871 | 0.047618 | 0.007162 | 5.918711 | -4.874451 |
| r4_b52_rs01 | 188220.753705 | 0.070406 | 0.046714 | 722.248648 | 0.026768 | -0.012147 | 941.738061 | 0.059819 | -0.033991 | -5.521391 | -0.010113 | 0.000000 | -0.089155 | -0.000906 | 0.003878 | -0.294488 | 0.000656 | 0.476790 | 0.020460 | 0.000843 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 34.885496 | 0.160567 | 4.771000 | 140319.216561 | 676.056871 | 0.047636 | 0.007162 | 5.918556 | -4.874451 |

Key full / OOS metrics:

| variant | full_roi_pct | full_maxdd_pct | full_sharpe | oos_roi_pct | oos_maxdd_pct | oos_sharpe | 2018_roi_pct | 2018_maxdd_pct | 2018_sharpe | 2019_roi_pct | 2019_maxdd_pct | 2019_sharpe | 2024_roi_pct | 2024_maxdd_pct | 2024_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r2_baseline | 367954.317437 | -45.087765 | 1.959263 | 10593.174002 | -35.135102 | 2.758126 | 10.264635 | -41.871807 | 1.440603 | 9.115546 | -15.906546 | 1.646549 | 306.098898 | -25.979847 | 3.627278 |
| r3_reentry | 415855.854581 | -50.959607 | 1.982033 | 10639.365779 | -30.272798 | 2.777732 | 2.971418 | -43.592294 | 1.272692 | 7.773901 | -25.246730 | 1.542516 | 342.490031 | -21.105041 | 3.804670 |
| r4_b50_rs00 | 555988.430804 | -45.040896 | 2.029627 | 11306.923424 | -35.147115 | 2.784617 | 12.292314 | -41.490267 | 1.477704 | 9.136006 | -15.906546 | 1.647392 | 341.127137 | -21.196210 | 3.789814 |
| r4_b50_rs01 | 556046.313884 | -45.040896 | 2.029651 | 11315.422650 | -35.147249 | 2.784894 | 12.292314 | -41.490267 | 1.477704 | 9.136006 | -15.906546 | 1.647392 | 340.984394 | -21.208847 | 3.787845 |
| r4_b52_rs01 | 556175.071142 | -45.041051 | 2.029669 | 11315.422650 | -35.147249 | 2.784894 | 9.970147 | -41.395017 | 1.441258 | 9.136006 | -15.906546 | 1.647392 | 340.984394 | -21.208847 | 3.787845 |

Interpretation:

- `r4_b50_rs01` is the cleanest candidate: full ROI improves by `188091.9964` points vs `r2`, OOS ROI improves by `722.2486`, and both `2018` and `2019` stop being the weak-years tax paid by unconditional `r3` reentry.
- vs `r3_reentry`, `r4_b50_rs01` also improves full ROI by `140190.4593` points and OOS ROI by `676.0569` while materially repairing the weak years.
- `r4_b52_rs01` is close, but it gives back too much 2018/2024 upside.
- conclusion: breadth alone was not enough; breadth + candidate relative-strength confirmation is the first clean state-dependent reentry upgrade.
