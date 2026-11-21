"""v3 experiments: (A) enforcement ablation - non-compliant drivers with / without the space barrier;
(B) queue-threshold fallback under overload."""
import pandas as pd
from experiments import runs, save, BASE
rows = []
for p in (0.0, 0.05, 0.10, 0.20):
    rows.append(runs(dict(BASE, noncomply=p, barrier=True), ["PARKSENSE"], 30, noncomply=p, barrier="with barrier"))
    rows.append(runs(dict(BASE, noncomply=p, barrier=False), ["PARKSENSE"], 30, noncomply=p, barrier="no barrier"))
A = save(pd.concat(rows), "V3A_enforcement", ["method", "noncomply", "barrier"])
cols = ["time_to_park_s_mean", "conflicts_per_100", "stolen_per_100", "refused_per_100", "res_success_pct",
        "area_imbalance_sd", "hot_area_h", "walk_res_m", "drive_m_per_veh", "noncomp_pct"]
print(A[cols].round(2).to_string())
rows = []
for d in (1.00, 1.15):
    for qf in (0, 5, 10, 20):
        rows.append(runs(dict(BASE, demand_level=d, queue_fallback=qf), ["PARKSENSE"], 30, demand=d, qf=qf))
    rows.append(runs(dict(BASE, demand_level=d), ["GUIDE"], 30, demand=d, qf=-1))
B = save(pd.concat(rows), "V3B_queue_fallback", ["method", "demand", "qf"])
cols = ["time_to_park_s_mean", "entrance_s_mean", "max_queue", "balk_pct", "res_success_pct", "conflicts_per_100",
        "drive_m_per_veh", "area_imbalance_sd", "fallback_pct"]
print(B[cols].round(2).to_string())
