"""v9: ParkSense with prototype-level hardware (6.5 s allocation, 27% entrance read failure, 10% space check failure)."""
import pandas as pd
from experiments import runs, save, BASE
HW = dict(assign_latency_s=6.5, anpr_fail=0.27, verify_fail=0.10, anpr_latency_s=2.0)
WK = dict(profile="weekday", demand_level=BASE["demand_weekday"])
rows = []
for day, extra in (("holiday", {}), ("weekday", WK)):
    rows.append(runs(dict(BASE, **extra), ["GUIDE", "PARKSENSE"], 30, hw="deployment", day=day))
    rows.append(runs(dict(BASE, **extra, **HW), ["PARKSENSE"], 30, hw="prototype", day=day))
A = save(pd.concat(rows), "V9_prototype_level", ["method", "hw", "day"])
cols = ["time_to_park_s_mean", "entrance_s_mean", "conflicts_per_100", "res_success_pct", "area_imbalance_sd", "hot_area_h", "balk_pct", "drive_m_per_veh"]
print(A[[c for c in cols if c in A.columns]].round(2).to_string())
