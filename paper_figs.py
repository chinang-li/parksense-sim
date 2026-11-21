"""Figures for Section 5 of the paper (grayscale-safe; widths for S&M: 16 cm full, 8 cm half)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = os.path.join(R, "paper"); os.makedirs(P, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "savefig.dpi": 600, "savefig.bbox": "tight", "legend.frameon": False})
CM = 1 / 2.54
ORDER = ["CONV", "GUIDE", "SPLIT", "NEAREST", "PARKSENSE"]
LBL = {"CONV": "Conventional", "GUIDE": "Guidance lights", "SPLIT": "Reserved zone", "NEAREST": "Nearest space", "PARKSENSE": "ParkSense"}
# Okabe-Ito colours ordered light -> dark so the bars stay distinguishable in grayscale; hatches add a second cue
GRAY = {"CONV": "#F0E442", "GUIDE": "#56B4E9", "SPLIT": "#E69F00", "NEAREST": "#009E73", "PARKSENSE": "#0B3C8C"}
LINE = {"Warehouse store": "#D55E00", "Department store": "#0072B2", "Shopping mall": "#009E73", "Hypermarket": "#CC79A7"}
HATCH = {"CONV": "", "GUIDE": "//", "SPLIT": "..", "NEAREST": "xx", "PARKSENSE": ""}


def save(fig, name):
    for ext in ("png", "tif", "pdf"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(P, f"{name}.{ext}"), **kw)
    # grayscale proof of the same figure
    from PIL import Image
    Image.open(os.path.join(P, f"{name}.png")).convert("L").save(os.path.join(P, f"{name}_grayscale_check.png"))


def fig3_calibration():
    v = pd.read_csv(os.path.join(R, "V1_occupancy_check.csv"))
    malls = ["Costco Dashun", "Shin Kong Mitsukoshi Zuoying", "Mitsui LaLaport Nangang", "Carrefour Xinnan"]
    short = {"Costco Dashun": "Warehouse store", "Shin Kong Mitsukoshi Zuoying": "Department store",
             "Mitsui LaLaport Nangang": "Shopping mall", "Carrefour Xinnan": "Hypermarket"}
    mk = {"Costco Dashun": "o", "Shin Kong Mitsukoshi Zuoying": "s", "Mitsui LaLaport Nangang": "^", "Carrefour Xinnan": "D"}
    fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6 * CM), sharey=True)
    for ax, typ, title in zip(axs, ["holiday", "weekday"], ["Holidays and weekends", "Weekdays"]):
        for m in malls:
            d = v[(v.mall == m) & (v.daytype == typ)]
            c = LINE[short[m]]
            ax.plot(d.hour, d.obs * 100, color=c, lw=1.1, marker=mk[m], ms=3.5, mfc="white", label=short[m])
            ax.plot(d.hour, d.sim * 100, color=c, lw=0.9, ls="--")
        ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel("Time of day (h)"); ax.set_xticks(range(8, 23, 2))
        ax.set_ylim(0, 105); ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    axs[0].set_ylabel("Occupancy (%)")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="black", ls="--", lw=0.9)); l.append("Simulated (dashed); observed (solid)")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, "fig3_calibration"); plt.close(fig)


def bars(ax, s, kpi, title, fmt="{:.0f}"):
    x = np.arange(len(ORDER))
    vals = [s.loc[m, kpi] for m in ORDER]; err = [s.loc[m, kpi + "_ci95"] for m in ORDER]
    for i, m in enumerate(ORDER):
        ax.bar(i, vals[i], yerr=err[i], color=GRAY[m], edgecolor="black", lw=0.6, hatch=HATCH[m], capsize=2, error_kw={"lw": 0.6})
        ax.text(i, vals[i] + err[i] + max(vals) * 0.02, fmt.format(vals[i]), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in ORDER], rotation=35, ha="right", fontsize=7)
    ax.set_title(title, loc="left", fontsize=8.5); ax.set_ylim(0, max(np.array(vals) + np.array(err)) * 1.18)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)


def fig4_main():
    s = pd.read_csv(os.path.join(R, "S1_main_summary.csv")).set_index("method")
    fig, axs = plt.subplots(1, 4, figsize=(16 * CM, 6.2 * CM))
    bars(axs[0], s, "time_to_park_s_mean", "(a) Time to park (s)")
    bars(axs[1], s, "drive_m_per_veh", "(b) Driving in lot (m)")
    bars(axs[2], s, "conflicts_per_100", "(c) Conflicts per 100")
    bars(axs[3], s, "hot_area_h", "(d) Area-hours ≥ 90%")
    fig.tight_layout(); save(fig, "fig4_main"); plt.close(fig)


def fig5_guidance():
    s = pd.read_csv(os.path.join(R, "S9_guide_robustness_summary.csv"))
    g = s[s.method == "GUIDE"]; ps = s[s.method == "PARKSENSE"].iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(16 * CM, 5.6 * CM))
    for ax, kpi, title in zip(axs, ["time_to_park_s_mean", "conflicts_per_100", "area_imbalance_sd"],
                              ["Time to park (s)", "Conflicts per 100 vehicles", "Area imbalance"]):
        for k, mk, ls, col in ((3, "o", "-", "#D55E00"), (5, "s", "--", "#0072B2")):
            d = g[g.topk == k].sort_values("period")
            ax.errorbar(d.period, d[kpi], yerr=d[kpi + "_ci95"], color=col, marker=mk, ms=3.5, mfc="white",
                        ls=ls, lw=1, capsize=2, elinewidth=0.6, label=f"Guidance, choose from {k}")
        ax.axhline(ps[kpi], color="#0B3C8C", lw=1.8, label="ParkSense")
        ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel("Light update period (s)")
        ax.set_xticks([0, 10, 30, 60]); ax.grid(axis="y", color="#e5e5e5", lw=0.5); ax.set_ylim(bottom=0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(); save(fig, "fig5_guidance"); plt.close(fig)


if __name__ == "__main__":
    fig3_calibration(); fig4_main(); fig5_guidance(); fig6_demand(); print("saved to", P)


def fig6_demand():
    s = pd.read_csv(os.path.join(R, "S2_demand_summary.csv"))
    sty = {"CONV": ("#7F7F7F", "x", ":"), "GUIDE": ("#0072B2", "s", "-."), "SPLIT": ("#E69F00", "D", "--"),
           "NEAREST": ("#009E73", "^", "--"), "PARKSENSE": ("#0B3C8C", "o", "-")}
    fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6 * CM))
    for ax, kpi, title in zip(axs, ["time_to_park_s_mean", "area_imbalance_sd"], ["Time to park (s)", "Area imbalance (SD of area occupancy)"]):
        for m in ORDER:
            d = s[s.method == m].sort_values("demand"); c, mk, ls = sty[m]
            ax.errorbar(d.demand, d[kpi], yerr=d[kpi + "_ci95"], color=c, marker=mk, ms=3.8, mfc="white", ls=ls,
                        lw=1.6 if m == "PARKSENSE" else 1.0, capsize=2, elinewidth=0.6, label=LBL[m])
        ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel("Demand level (peak demand / capacity)")
        ax.set_xticks([0.6, 0.7, 0.8, 0.93, 1.0, 1.15]); ax.grid(axis="y", color="#e5e5e5", lw=0.5); ax.set_ylim(bottom=0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, "fig6_demand"); plt.close(fig)
