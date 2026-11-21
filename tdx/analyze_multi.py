"""TDX occupancy -> arrival profiles and demand levels for four mall car parks (Nov 2026 calibration).

Inputs (same folder):
  availability_Kaohsiung.csv      first week, 25 Oct - 1 Nov 2026 (Costco only used)
  availability_Kaohsiung_2.csv    3 - 15 Nov 2026
  availability_Taipei_2.csv       3 - 15 Nov 2026
Cleaning: AvailableSpaces < 0 -> missing; isolated drops to 0 -> missing; a value unchanged for >= 60 min
inside 10:00-21:00 while the lot is not full -> missing (frozen feed); 13 Nov removed in Kaohsiung (all lots
frozen); a day is kept if <= 20% of its 10:00-21:00 samples are missing and no gap is longer than 150 min.
Arrivals: occupied vehicles above the overnight base are deconvolved with a lognormal dwell (Eq. 4).
"""
import json, math, sys
import numpy as np, pandas as pd

LOTS = {"KHB00434": ("Costco Dashun", "warehouse store"),
        "KHB00245": ("Shin Kong Mitsukoshi Zuoying", "department store"),
        "KHB00247": ("Carrefour Xinnan", "hypermarket"),
        "TPE1813": ("Mitsui LaLaport Nangang", "shopping mall")}
HOLIDAYS = {pd.Timestamp("2026-10-26").date()}            # make-up holiday (Mon)
BAD_DAYS = {"KHH": {pd.Timestamp("2026-11-13").date()}}    # TDX outage, all Kaohsiung lots frozen
DWELLS = (60, 90, 120)


def load():
    a = pd.read_csv("availability_Kaohsiung.csv", encoding="utf-8-sig")
    a = a[a.CarParkID == "KHB00434"]
    b = pd.read_csv("availability_Kaohsiung_2.csv", encoding="utf-8-sig")
    c = pd.read_csv("availability_Taipei_2.csv", encoding="utf-8-sig")
    d = pd.concat([a, b, c], ignore_index=True)
    d = d[d.CarParkID.isin(LOTS)]
    d["t"] = pd.to_datetime(d["DataCollectTime"].astype(str).str[:19])
    return d.drop_duplicates(["CarParkID", "t"]).sort_values(["CarParkID", "t"])


def clean(g, cap):
    v = g.AvailableSpaces.astype(float).where(g.AvailableSpaces >= 0)
    # isolated zero: 0 with both neighbours > 5% of capacity
    z = (v == 0) & (v.shift(1) > 0.05 * cap) & (v.shift(-1) > 0.05 * cap)
    v[z] = np.nan
    grp = (v != v.shift()).cumsum()
    dur = g.groupby(grp)["t"].transform(lambda s: (s.max() - s.min()).total_seconds() / 60)
    biz = g.t.dt.hour.between(10, 20)
    v[(dur >= 60) & biz & (v > 0)] = np.nan
    return v


def lognorm_steps(mean, cv=0.5, step=10, n=90):
    s2 = math.log(1 + cv ** 2); mu = math.log(mean) - s2 / 2; s = math.sqrt(s2)
    cdf = lambda x: 0.5 * (1 + math.erf((math.log(x) - mu) / (s * math.sqrt(2)))) if x > 0 else 0.0
    return np.array([cdf((k + 1) * step) - cdf(k * step) for k in range(n)])


def arrivals(N, dwell):
    """N: transient vehicles per 10-min step for one day -> arrivals per step (lower bound when full)."""
    f = lognorm_steps(dwell)
    lam = np.zeros(len(N)); dep = np.zeros(len(N) + len(f))
    for k in range(len(N) - 1):
        a = max(0.0, N[k + 1] - N[k] + dep[k])
        lam[k] = a
        dep[k + 1:k + 1 + len(f)] += a * f[: len(dep) - k - 1][: len(f)]
    return lam


def main():
    d = load()
    out, rows = {}, []
    for cid, (name, kind) in LOTS.items():
        g = d[d.CarParkID == cid].reset_index(drop=True)
        cap = int(g.TotalSpaces.mode().iloc[0])
        g["v"] = clean(g, cap)
        city = "KHH" if cid.startswith("KHB") else "TPE"
        s = g.set_index("t")["v"]
        days_out = {"weekday": [], "holiday": []}
        curves = {"weekday": [], "holiday": []}
        for day, gg in g.groupby(g.t.dt.date):
            if day in BAD_DAYS.get(city, set()):
                continue
            biz = gg[gg.t.dt.hour.between(10, 20)]
            if len(biz) == 0 or biz.v.isna().mean() > 0.2:
                continue
            ok = gg.dropna(subset=["v"])
            t0, t1 = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
            pts = pd.concat([pd.Series([np.nan], [t0]), ok.set_index("t")["v"], pd.Series([np.nan], [t1])])
            gaps = pts.index.to_series().diff().dt.total_seconds().div(60)
            if len(ok) < 20 or gaps.max() > 240 or (ok.t.min() - t0).total_seconds() > 4 * 3600 \
                    or (t1 - ok.t.max()).total_seconds() > 4 * 3600:
                continue
            biz_gap = ok[ok.t.dt.hour.between(10, 20)].t.diff().dt.total_seconds().div(60).max()
            if biz_gap > 150:
                continue
            grid = pd.date_range(t0, t1, freq="10min", inclusive="left")
            ser = ok.set_index("t")["v"]
            occv = cap - ser.reindex(ser.index.union(grid)).interpolate("time").bfill().ffill().reindex(grid).values
            typ = "holiday" if (pd.Timestamp(day).dayofweek >= 5 or day in HOLIDAYS) else "weekday"
            days_out[typ].append(str(day)); curves[typ].append(occv)
        res = {"name": name, "kind": kind, "cap": cap, "days": days_out, "types": {}}
        for typ in ("weekday", "holiday"):
            if not curves[typ]:
                continue
            C = np.array(curves[typ])                      # vehicles, days x 144
            base = np.median(C[:, 18:30], axis=1)            # 03:00-05:00 overnight base per day
            eff = cap - base.mean()
            T = np.clip(C - base[:, None], 0, None)           # transient vehicles
            r = {"n_days": len(C), "base_share": float(base.mean() / cap), "eff_cap": float(eff),
                 "occ_mean": (C.mean(0) / cap).round(4).tolist(),
                 "occ_min": (C.min(0) / cap).round(4).tolist(), "occ_max": (C.max(0) / cap).round(4).tolist(),
                 "peak_occ": float((C.max(1) / cap).mean())}
            for dw in DWELLS:
                lam = np.array([arrivals(x, dw) for x in T]).mean(0).reshape(24, 6).sum(1)   # veh/h
                r[f"arr_h_{dw}"] = lam.round(1).tolist()
                r[f"demand_{dw}"] = float(lam.max() * dw / 60 / eff)
                prof = lam[8:22] / lam[8:22].max()
                r[f"profile_{dw}"] = prof.round(3).tolist()
            res["types"][typ] = r
            rows.append([cid, name, cap, typ, len(C), round(r["base_share"], 2), round(r["peak_occ"], 2),
                         round(r["demand_60"], 2), round(r["demand_90"], 2), round(r["demand_120"], 2)])
        out[cid] = res
    json.dump(out, open("tdx_profiles_multi.json", "w"), indent=1)
    tab = pd.DataFrame(rows, columns=["id", "name", "cap", "type", "days", "night_base", "peak_occ",
                                      "demand_60", "demand_90", "demand_120"])
    tab.to_csv("../results/tdx_malls_summary.csv", index=False)
    print(tab.to_string(index=False))
    for cid, r in out.items():
        print(cid, {k: v for k, v in r["days"].items()})


if __name__ == "__main__":
    main()
