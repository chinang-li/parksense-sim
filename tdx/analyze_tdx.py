"""TDX occupancy -> arrival profile for the simulation (Costco Dashun, KHB00434)."""
import json, math
import numpy as np, pandas as pd
d = pd.read_csv("availability_Kaohsiung.csv", encoding="utf-8-sig")
d["t"] = pd.to_datetime(d["DataCollectTime"]).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
d["occ"] = 1 - d.AvailableSpaces / d.TotalSpaces
c = d[d.CarParkID == "KHB00434"].set_index("t").sort_index()
cap = int(c.TotalSpaces.iloc[0])
HOLIDAY = {pd.Timestamp("2026-10-26").date()}          # Retrocession Day make-up holiday (Mon)
def daytype(dt):
    return "holiday" if (dt.dayofweek >= 5 or dt.date() in HOLIDAY) else "weekday"
grid = pd.date_range(c.index.min().ceil("10min"), c.index.max().floor("10min"), freq="10min")
occ = c["occ"].reindex(c.index.union(grid)).interpolate("time").reindex(grid)
df = pd.DataFrame({"occ": occ})
df["date"] = df.index.date; df["tod"] = df.index.hour + df.index.minute / 60
df["type"] = [daytype(t) for t in df.index]
full_days = [k for k, g in df.groupby("date") if len(g) == 144]
df = df[df.date.isin(full_days)]

def lognorm_pdf_steps(mean, cv=0.5, step_min=10, n=72):
    s2 = math.log(1 + cv ** 2); mu = math.log(mean) - s2 / 2; s = math.sqrt(s2)
    from math import erf
    cdf = lambda x: 0.5 * (1 + erf((math.log(x) - mu) / (s * math.sqrt(2)))) if x > 0 else 0
    return np.array([cdf((k + 1) * step_min) - cdf(k * step_min) for k in range(n)])

def arrivals(N, dwell):
    """N: occupied vehicles per 10-min step (one day). Returns served arrivals per step (lower bound when full)."""
    f = lognorm_pdf_steps(dwell)
    lam = np.zeros(len(N)); dep = np.zeros(len(N) + len(f))
    base = N[0]                                            # overnight vehicles leave slowly: treat as constant
    for k in range(len(N) - 1):
        a = max(0.0, N[k + 1] - N[k] + dep[k])
        lam[k] = a
        dep[k + 1:k + 1 + len(f)] += a * f[: len(dep) - k - 1][: len(f)]
    return lam

out = {"cap": cap, "days": {}, "profiles": {}}
for typ in ("weekday", "holiday"):
    days = sorted({dd for dd, t in zip(df.date, df.type) if t == typ})
    out["days"][typ] = [str(x) for x in days]
    curves = np.array([df[df.date == dd].occ.values for dd in days])
    out["profiles"][typ] = {"occ_mean": curves.mean(0).round(4).tolist(), "occ_min": curves.min(0).round(4).tolist(),
                            "occ_max": curves.max(0).round(4).tolist()}
    for dwell in (60, 90, 120):
        lam = np.array([arrivals(cv * cap, dwell) for cv in curves]).mean(0)    # veh per 10 min
        hourly = lam.reshape(24, 6).sum(1)                                       # veh per hour
        out["profiles"][typ][f"arr_per_h_dwell{dwell}"] = hourly.round(1).tolist()
json.dump(out, open("tdx_profile.json", "w"), indent=1)
for typ in ("weekday", "holiday"):
    p = out["profiles"][typ]
    print(typ, out["days"][typ])
    print(" peak occ", max(p["occ_mean"]), " arrivals/h (dwell 90):", p["arr_per_h_dwell90"][7:22])
    for dw in (60, 90, 120):
        h = np.array(p[f"arr_per_h_dwell{dw}"]); print(f"  dwell {dw}: daily {h.sum():.0f} veh, peak {h.max():.0f}/h at {h.argmax()}:00, demand level={h.max()*dw/60/cap:.2f}")

# ---------------- figure for the paper
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {"weekday": "#2a78d6", "holiday": "#eb6834"}
tod = np.arange(144) / 6
fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
for typ in ("weekday", "holiday"):
    p = out["profiles"][typ]
    axs[0].fill_between(tod, np.array(p["occ_min"]) * 100, np.array(p["occ_max"]) * 100, color=COL[typ], alpha=0.15, lw=0)
    axs[0].plot(tod, np.array(p["occ_mean"]) * 100, color=COL[typ], lw=2, label=f"{typ.capitalize()} (n={len(out['days'][typ])} days)")
    axs[1].plot(range(24), p["arr_per_h_dwell90"], color=COL[typ], lw=2, marker="o" if typ == "holiday" else "s", ms=4, label=typ.capitalize())
axs[0].set_xlim(0, 24); axs[0].set_xticks(range(0, 25, 3)); axs[0].set_ylabel("Occupancy (%)"); axs[0].set_xlabel("Time of day (h)")
axs[0].set_title("Observed occupancy, Costco Dashun lot (614 spaces)", fontsize=9, loc="left"); axs[0].legend(fontsize=7, frameon=False)
axs[1].set_xlim(6, 23); axs[1].set_ylabel("Vehicles per hour"); axs[1].set_xlabel("Time of day (h)")
axs[1].set_title("Derived arrival rate (90-min dwell)", fontsize=9, loc="left"); axs[1].legend(fontsize=7, frameon=False)
fig.suptitle("TDX calibration data, Kaohsiung, 25 Oct - 1 Nov 2026 (10-min sampling)", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("../results/fig_tdx_calibration.png")
print("figure saved")
