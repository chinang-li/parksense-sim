"""
Run the simulation experiments and draw the figures.

  python experiments.py all            # everything (S1 30 reps, others 10 reps)
  python experiments.py S1 --reps 30   # one experiment
Results -> results/*.csv, figures -> results/fig_*.png (300 dpi)

S1 main comparison (5 methods, baseline)          S4 ParkSense sensitivity (penalty, walk-in weight)
S2 demand level 0.6-1.1                            S5 assignment latency x lock (duplicate assignments, LED stops)
S3 reservation share 0-60%                         S6 booking lead time (space hold cost)
S0 one-day area occupancy curves (CONV vs PARKSENSE)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BASE, METHODS, METHOD_LABELS, MALLS
from model import build_layout, generate_demand, Sim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

COLOR = {"CONV": "#8a8984", "GUIDE": "#2a78d6", "SPLIT": "#1baf7a", "NEAREST": "#6250d6", "PARKSENSE": "#eb6834"}
MARK = {"CONV": "s", "GUIDE": "^", "SPLIT": "D", "NEAREST": "v", "PARKSENSE": "o"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

KPI_LABEL = {
    "time_to_park_s_mean": "Time to park (s)", "search_s_mean": "In-lot search time (s)",
    "entrance_s_mean": "Entrance delay (s)", "walk_m_mean": "Walking distance (m)",
    "drive_m_per_veh": "In-lot driving (m/vehicle)", "conflicts_per_100": "Space conflicts per 100 vehicles",
    "balk_pct": "Drivers who gave up (%)", "area_imbalance_sd": "Area occupancy imbalance (SD)",
    "co2_kg": "CO₂ in the lot (kg/day)", "res_success_pct": "Reservation success (%)",
    "hold_space_h": "Reserved-but-empty space-hours", "dup_assign_per_100": "Duplicate assignments per 100 vehicles",
    "led_stop_pct": "Vehicles stopping at the LED screen (%)", "search_s_p95": "Search time P95 (s)",
}


def runs(cfg, methods, reps, tag=None, **extra):
    layout = build_layout(cfg, seed=0)
    cap = int(layout["OK"].sum())
    rows = []
    for r in range(reps):
        demand = generate_demand(cfg, cap, seed=r)          # same demand for every method (common random numbers)
        for m in methods:
            k = Sim(cfg, m, layout, demand, seed=r).run().kpis()
            k.update(rep=r, **extra)
            rows.append(k)
    return pd.DataFrame(rows)


def ci95(x):
    x = pd.Series(x).dropna()
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def summary(df, by):
    num = [c for c in df.columns if c not in by + ["rep", "method"] or c == "method"]
    num = [c for c in df.select_dtypes("number").columns if c not in by + ["rep"]]
    g = df.groupby(by)[num]
    out = g.mean().round(3).add_suffix("")
    ci = g.agg(ci95).round(3).add_suffix("_ci95")
    return pd.concat([out, ci], axis=1).sort_index(axis=1)


def save(df, name, by):
    df.to_csv(os.path.join(OUT, f"{name}_raw.csv"), index=False)
    s = summary(df, by)
    s.to_csv(os.path.join(OUT, f"{name}_summary.csv"))
    return s


# ------------------------------------------------------------------ S1
def S1(reps, weekday=False):
    cfg = dict(BASE, profile="weekday", demand_level=BASE["demand_weekday"]) if weekday else BASE
    name = "S1_weekday" if weekday else "S1_main"
    df = runs(cfg, METHODS, reps)
    s = save(df, name, ["method"]).reindex(METHODS)
    kpis = ["time_to_park_s_mean", "drive_m_per_veh", "walk_m_mean", "conflicts_per_100", "area_imbalance_sd", "co2_kg"]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, k in zip(axs.flat, kpis):
        y, e = s[k].values, s[k + "_ci95"].values
        ax.bar(range(len(METHODS)), y, yerr=e, color=[COLOR[m] for m in METHODS], width=0.62,
               error_kw=dict(lw=0.8, capsize=2, ecolor="#52514e"), edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(METHODS)), ["Conv.", "Guide", "Split", "Nearest", "ParkSense"], fontsize=8)
        ax.set_title(KPI_LABEL[k], fontsize=9, loc="left")
        for i, v in enumerate(y):
            ax.text(i, v + e[i], f"{v:.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
        ax.grid(axis="x", visible=False)
    day = "weekday" if weekday else "holiday"
    fig.suptitle(f"S1  Method comparison ({day} TDX profile, demand {cfg['demand_level']}, reservation {int(cfg['res_share']*100)}%, {reps} runs, mean ± 95% CI)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{name}.png"))
    plt.close(fig)
    return s


# ------------------------------------------------------------------ line-chart helper
def lines(s, xcol, kpis, methods, title, fname, xlabel):
    fig, axs = plt.subplots(1, len(kpis), figsize=(3.4 * len(kpis), 3.0))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, kpis):
        for m in methods:
            d = s.xs(m, level="method").sort_index()
            if d[k].isna().all():
                continue
            ax.errorbar(d.index, d[k], yerr=d[k + "_ci95"], color=COLOR[m], marker=MARK[m], ms=5, lw=2,
                        capsize=2, elinewidth=0.8, label=METHOD_LABELS[m])
        ax.set_title(KPI_LABEL[k], fontsize=9, loc="left")
        ax.set_xlabel(xlabel)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


# ------------------------------------------------------------------ S2
def S2(reps):
    dfs = [runs(dict(BASE, demand_level=d), METHODS, reps, demand=d) for d in (0.6, 0.7, 0.8, 0.93, 1.0, 1.15)]
    s = save(pd.concat(dfs), "S2_demand", ["method", "demand"])
    s = s.reset_index().set_index(["method", "demand"])
    lines(s, "demand", ["time_to_park_s_mean", "walk_m_mean", "area_imbalance_sd", "balk_pct"], METHODS,
          f"S2  Effect of demand level (TDX: weekday 0.93, holiday 1.00; {reps} runs per point)", "fig_S2_demand.png", "Peak demand / capacity")
    return s


# ------------------------------------------------------------------ S3
def S3(reps):
    ms = ["SPLIT", "NEAREST", "PARKSENSE"]
    dfs = []
    for r in (0.0, 0.15, 0.30, 0.45, 0.60):
        cfg = dict(BASE, res_share=r, split_zone_share=max(r, 0.05))
        dfs.append(runs(cfg, ms, reps, res_share=r))
    s = save(pd.concat(dfs), "S3_res_share", ["method", "res_share"]).reset_index().set_index(["method", "res_share"])
    lines(s, "res_share", ["res_success_pct", "time_to_park_s_mean", "walk_m_mean"], ms,
          f"S3  Effect of reservation share (SPLIT zone = share; {reps} runs)", "fig_S3_res_share.png", "Share of reserved drivers")
    return s


# ------------------------------------------------------------------ S4
def S4(reps):
    rows = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        for f in (0.3, 0.5, 0.7, 1.0):
            d = runs(dict(BASE, penalty_threshold=th, penalty_factor=f), ["PARKSENSE"], reps, threshold=th, factor=f)
            rows.append(d)
    pen = save(pd.concat(rows), "S4a_penalty", ["threshold", "factor"])
    rows = []
    for w1 in (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        rest = np.array(BASE["w_walkin"][1:]) / sum(BASE["w_walkin"][1:]) * (1 - w1)
        cfg = dict(BASE, w_walkin=(w1, *rest))
        rows.append(runs(cfg, ["PARKSENSE"], reps, w1_walkin=w1))
    wt = save(pd.concat(rows), "S4b_walkin_weight", ["w1_walkin"])

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axs[0]
    piv = pen["area_imbalance_sd"].unstack("factor")
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), ["none" if t > 1 else f"{int(t*100)}%" for t in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v > np.nanmean(piv.values) else "#0b0b0b")
    ax.set_xlabel("Penalty factor"); ax.set_ylabel("Penalty threshold (area occupancy)")
    ax.set_title("Area imbalance (SD) — lower is better", fontsize=9, loc="left"); ax.grid(False)
    ax = axs[1]
    ax.plot(wt["walk_m_mean"], wt["area_imbalance_sd"], color=COLOR["PARKSENSE"], marker="o", ms=5, lw=2)
    for w1, r in wt.iterrows():
        ax.annotate(f"W1={w1:g}", (r["walk_m_mean"], r["area_imbalance_sd"]), fontsize=7, xytext=(4, 3), textcoords="offset points", color="#52514e")
    ax.set_xlabel("Walking distance (m)"); ax.set_ylabel("Area imbalance (SD)")
    ax.set_title("Walk-in weight on free capacity: trade-off", fontsize=9, loc="left")
    fig.suptitle(f"S4  ParkSense parameter sensitivity ({reps} runs per point; code values: 70%, x0.5, W1=0.5)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_S4_sensitivity.png")); plt.close(fig)
    return pen, wt


# ------------------------------------------------------------------ S5
def S5(reps):
    rows = []
    for lock in (True, False):
        for lat in (0.05, 0.5, 1.0, 2.0, 3.0, 5.0, 6.5, 10.0):
            rows.append(runs(dict(BASE, assign_lock=lock, assign_latency_s=lat), ["PARKSENSE"], reps, lock=lock, latency=lat))
    s = save(pd.concat(rows), "S5_latency_lock", ["lock", "latency"])
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, k in zip(axs, ["dup_assign_per_100", "led_stop_pct", "time_to_park_s_mean"]):
        for lock, col, mk, lab in ((False, "#eb6834", "o", "No lock (prototype code)"), (True, "#2a78d6", "s", "With lock")):
            d = s.xs(lock, level="lock")
            ax.errorbar(d.index, d[k], yerr=d[k + "_ci95"], color=col, marker=mk, ms=5, lw=2, capsize=2, elinewidth=0.8, label=lab)
        ax.set_title(KPI_LABEL[k], fontsize=9, loc="left"); ax.set_xlabel("Assignment latency (s)")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"S5  Database latency and locking (prototype Google Sheets: reservation ≈ 1.7 s, walk-in ≈ 6.5 s; {reps} runs)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_S5_latency_lock.png")); plt.close(fig)
    return s


# ------------------------------------------------------------------ S6
def S6(reps):
    rows = []
    for lo, hi in ((5, 15), (10, 30), (10, 60), (30, 90), (60, 120)):
        rows.append(runs(dict(BASE, lead_min=(lo, hi)), ["SPLIT", "PARKSENSE"], reps, lead=f"{lo}-{hi}"))
    return save(pd.concat(rows), "S6_lead_time", ["method", "lead"])


# ------------------------------------------------------------------ S7
def S7(reps):
    """Dwell is not identifiable from occupancy: demand re-derived from TDX (holiday) for each dwell (tdx/analyze_tdx.py)."""
    levels = {60: 0.93, 90: 1.00, 120: 1.09}       # Costco holiday, tdx/analyze_multi.py
    rows = [runs(dict(BASE, dwell_mean_min=float(dw), dwell_sd_min=dw / 2, demand_level=lv), METHODS, reps, dwell=dw) for dw, lv in levels.items()]
    s = save(pd.concat(rows), "S7_dwell", ["method", "dwell"]).reset_index().set_index(["method", "dwell"])
    lines(s, "dwell", ["time_to_park_s_mean", "drive_m_per_veh", "area_imbalance_sd"], METHODS,
          f"S7  Sensitivity to mean parking duration (demand re-derived from TDX; {reps} runs)", "fig_S7_dwell.png", "Mean parking duration (min)")
    return s


# ------------------------------------------------------------------ S8
def S8(reps):
    """What-if: entrance failure at 14:00 on a holiday. Gate methods: gate broken (nobody enters).
    Gate-free methods: entrance camera down -> vehicles drive in and search (reserved drivers use the space number from the app)."""
    rows = []
    for dur in (0, 15, 30, 45):
        cfg = dict(BASE, outage=(14, dur), outage_kpi=(14, 75))
        d = runs(cfg, METHODS, reps, outage_min=dur)                       # ParkSense as built: walk-ins wait
        f = runs(dict(cfg, outage_fallback=True), ["PARKSENSE"], reps, outage_min=dur)
        f["method"] = "PS_FALLBACK"                                        # proposed fallback mode: drive in and search
        rows += [d, f]
    s = save(pd.concat(rows), "S8_outage", ["method", "outage_min"]).reset_index().set_index(["method", "outage_min"])
    KPI_LABEL.update({"win_time_to_park_s": "Time to park, arrivals 14:00-15:15 (s)", "win_balk_pct": "Gave up, arrivals 14:00-15:15 (%)",
                      "max_queue": "Longest entrance queue (vehicles)"})
    COLOR["PS_FALLBACK"], MARK["PS_FALLBACK"] = "#0e2b40", "x"
    METHOD_LABELS["PS_FALLBACK"] = "ParkSense + fallback mode (proposed)"
    saved = METHOD_LABELS["PARKSENSE"]; METHOD_LABELS["PARKSENSE"] = "ParkSense (as built)"
    lines(s, "outage_min", ["win_time_to_park_s", "win_balk_pct", "max_queue"], ["GUIDE", "SPLIT", "PARKSENSE", "PS_FALLBACK"],
          f"S8  Entrance failure at 14:00 (ParkSense as built vs. with a fallback mode; {reps} runs)", "fig_S8_outage.png", "Failure duration (min)")
    METHOD_LABELS["PARKSENSE"] = saved
    return s


# ------------------------------------------------------------------ S0
def S0():
    layout = build_layout(BASE, seed=0)
    demand = generate_demand(BASE, int(layout["OK"].sum()), seed=0)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for ax, m in zip(axs, ["CONV", "PARKSENSE"]):
        sim = Sim(BASE, m, layout, demand, seed=0).run()
        t = np.array([x[0] for x in sim.samples]) / 3600 + BASE["open_h"]
        occ = np.array([x[1] for x in sim.samples])
        for a in range(layout["na"]):
            ax.plot(t, occ[:, a] * 100, lw=1.6, label=f"Area {layout['names'][a]}")
        ax.set_title(METHOD_LABELS[m], fontsize=9, loc="left"); ax.set_xlabel("Time of day (h)")
        ax.set_xlim(BASE["open_h"], BASE["close_h"] + 2)
    axs[0].set_ylabel("Area occupancy (%)")
    axs[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("S0  Area occupancy over one holiday (TDX profile) (A-C near the mall, D-F far)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_S0_area_occupancy.png")); plt.close(fig)


def layout_figure():
    L = build_layout(BASE, seed=0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sc = ax.scatter(L["X"], L["Y"], c=L["SE"], cmap="Blues", s=14, marker="s", vmin=0, vmax=10)
    ax.scatter(L["X"][~L["OK"]], L["Y"][~L["OK"]], s=22, marker="x", color="#d03b3b", label="Out of service")
    for a in range(L["na"]):
        ax.text(*L["cen"][a], L["names"][a], fontsize=14, fontweight="bold", ha="center", va="center", color="#0b0b0b", alpha=0.6)
    ax.plot(*L["mall"], marker="*", ms=14, color="#eb6834", ls="", label="Mall entrance")
    ax.plot(*L["ent"], marker=">", ms=10, color="#0b0b0b", ls="", label="Vehicle entrance")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Space-to-entrance score (0-10)")
    ax.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1, -0.42), ncol=3)
    ax.set_title(f"Simulated lot: {L['n']} spaces, {L['na']} areas", fontsize=9, loc="left")
    fig.savefig(os.path.join(OUT, "fig_layout.png")); plt.close(fig)


# ------------------------------------------------------------------ S9
def S9(reps):
    """Robustness of the guidance-light baseline: light update period x number of nearby free spaces
    a driver chooses from. 0 s = real-time lights. Answers 'was the baseline set too weak?'"""
    rows = [runs(BASE, ["NEAREST", "PARKSENSE"], reps, period=-1, topk=-1)]
    for period in (0.0, 10.0, 30.0, 60.0):
        for k in (1, 3, 5):
            rows.append(runs(dict(BASE, info_period_s=period, guide_topk=k), ["GUIDE"], reps, period=period, topk=k))
    s = save(pd.concat(rows), "S9_guide_robustness", ["method", "period", "topk"])
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, kpi in zip(axs, ["time_to_park_s_mean", "conflicts_per_100", "area_imbalance_sd"]):
        g = s.xs("GUIDE", level="method")
        for k, mk in ((1, "v"), (3, "o"), (5, "s")):
            d = g.xs(k, level="topk")
            ax.errorbar(d.index, d[kpi], yerr=d[kpi + "_ci95"], marker=mk, ms=5, lw=1.6, capsize=2,
                        color={1: "#5c5c5c", 3: "#2a78d6", 5: "#9bbde8"}[k], label=f"Guidance, choose from {k}")
        ps = s.loc[("PARKSENSE", -1, -1), kpi]
        ax.axhline(ps, color=COLOR["PARKSENSE"], lw=2, ls="--", label="ParkSense")
        ax.set_title(KPI_LABEL.get(kpi, kpi), fontsize=9, loc="left"); ax.set_xlabel("Light update period (s; 0 = real time)")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"S9  Guidance-light baseline under better settings ({reps} runs, holiday)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_S9_guide_robustness.png")); plt.close(fig)
    return s


# ------------------------------------------------------------------ S10
def S10(reps):
    """Gate service time sensitivity (2-8 s). Component lower bound from datasheets: plate read 0.3-1.5 s + barrier
    opening 1-4 s; the rest is driver stop/reaction/start, which datasheets do not give."""
    rows = [runs(BASE, ["PARKSENSE"], reps, gate=0.0)]
    for g in (2.0, 4.0, 6.0, 8.0):
        rows.append(runs(dict(BASE, gate_service_s=(g, 0.15 * g)), ["CONV", "GUIDE", "SPLIT"], reps, gate=g))
    return save(pd.concat(rows), "S10_gate_time", ["method", "gate"])


# ------------------------------------------------------------------ S11 dual-mode ablation
def S11(reps):
    """Is the dual-mode weighting needed? Same weights for both modes vs mode-specific weights."""
    W, R = BASE["w_walkin"], BASE["w_reservation"]
    rows = [runs(BASE, ["PARKSENSE"], reps, variant="dual (proposed)"),
            runs(dict(BASE, w_reservation=W), ["PARKSENSE"], reps, variant="walk-in weights for both"),
            runs(dict(BASE, w_walkin=R), ["PARKSENSE"], reps, variant="reservation weights for both"),
            runs(BASE, ["NEAREST"], reps, variant="nearest (no weights)")]
    return save(pd.concat(rows), "S11_dual_mode_ablation", ["method", "variant"])


# ------------------------------------------------------------------ S12 scale
def S12(reps):
    """Larger car park: 12 areas x 84 spaces = 1008 spaces, same demand level relative to capacity."""
    cfg = dict(BASE, areas_x=4, areas_y=3, rows_per_area=7, spaces_per_row=12)
    return save(runs(cfg, METHODS, reps, spaces=1008), "S12_scale_1008", ["method", "spaces"])


# ------------------------------------------------------------------ S13 per-mall
def S13(reps):
    """Main comparison repeated with the arrival profile and demand level of each of the four malls."""
    rows = []
    for key, m in MALLS.items():
        cfg = dict(BASE, profile="mall", profile_mall=m["profile"], demand_level=m["demand"])
        rows.append(runs(cfg, METHODS, reps, mall=m["name"], daytype=m["type"], demand=m["demand"]))
    return save(pd.concat(rows), "S13_per_mall", ["mall", "daytype", "method"])


# ------------------------------------------------------------------ V1 occupancy check
def V1(reps):
    """Does the calibrated model reproduce the observed occupancy curve? Total occupancy (share of capacity) by hour."""
    import json
    obs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdx", "tdx_profiles_multi.json")))
    rows = []
    for key, m in MALLS.items():
        cid, typ = key.rsplit("_", 1)
        cfg = dict(BASE, profile="mall", profile_mall=m["profile"], demand_level=m["demand"])
        layout = build_layout(cfg, seed=0); cap = int(layout["OK"].sum())
        sim_h = np.zeros(24); cnt = np.zeros(24)
        for r in range(reps):
            sim = Sim(cfg, "CONV", layout, generate_demand(cfg, cap, seed=r), seed=r).run()   # no reservation holds: physical occupancy
            for t, _, tot in sim.samples:
                h = int(t // 3600 + cfg["open_h"]) % 24; sim_h[h] += tot; cnt[h] += 1
        sim_occ = np.where(cnt > 0, sim_h / np.maximum(cnt, 1), np.nan)
        r_ = obs[cid]["types"][typ]; base = r_["base_share"]
        o = np.array(r_["occ_mean"]).reshape(24, 6).mean(1)
        obs_occ = np.clip((o - base) / (1 - base), 0, None)               # transient vehicles / effective capacity
        for h in range(8, 23):
            rows.append(dict(mall=m["name"], daytype=typ, hour=h, sim=sim_occ[h], obs=obs_occ[h]))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "V1_occupancy_check.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?", default="all")
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    todo = ["layout", "S0", "S1", "S1w", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "V1"] if a.exp == "all" else [a.exp]
    for e in todo:
        print("running", e, flush=True)
        if e == "layout":
            layout_figure()
        elif e == "S0":
            S0()
        elif e == "S1":
            print(S1(a.reps or 30).T)
        elif e == "S1w":
            S1(a.reps or 30, weekday=True)
        else:
            globals()[e](a.reps or 10)
    print("done ->", OUT)
