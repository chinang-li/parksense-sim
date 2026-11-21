"""Paired comparisons over the 30 replications (common random numbers): t-test and Wilcoxon signed-rank."""
import os, sys
import pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def paired(df, a, b, kpi, key="rep"):
    x = df[df.method == a].set_index(key)[kpi]; y = df[df.method == b].set_index(key)[kpi]
    d = (x - y.reindex(x.index)).dropna(); n = len(d)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=d.std(ddof=1) / n ** 0.5)
    return d.mean(), ci, stats.ttest_rel(x, y.reindex(x.index)).pvalue, stats.wilcoxon(d).pvalue, n

def show(name, df, a, b, kpis):
    for k in kpis:
        m, ci, pt, pw, n = paired(df, a, b, k)
        print(f"{name:8s} {a}-{b:9s} {k:22s} diff {m:8.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  t p={pt:.2g}  Wilcoxon p={pw:.2g}  n={n}")

K = ["time_to_park_s_mean", "area_imbalance_sd", "hot_area_h", "walk_m_mean"]
for f, name in (("S1_main_raw.csv", "holiday"), ("S1_weekday_raw.csv", "weekday")):
    p = os.path.join(R, f)
    if os.path.exists(p):
        show(name, pd.read_csv(p), "PARKSENSE", "NEAREST", K)
p = os.path.join(R, "S9_guide_robustness_raw.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    best = d[(d.method == "GUIDE") & (d.period == 0) & (d.topk == 5)].assign(method="GUIDE_best")
    show("holiday", pd.concat([d[d.method == "PARKSENSE"].drop_duplicates("rep"), best]), "PARKSENSE", "GUIDE_best", K)
