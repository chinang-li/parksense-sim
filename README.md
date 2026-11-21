# ParkSense simulation

Discrete-event simulation used in the paper "Sensor-Based License Plate Binding for Parking Space Allocation in Shopping Malls: Prototype Tests and Data-Calibrated Simulation" (C.-A. Li et al., submitted to Sensors and Materials). It compares five ways of parking in a 300-space mall car park: conventional searching, guidance lights, a fixed reserved zone, nearest-space allocation, and ParkSense (license-plate binding with dual-mode weighted allocation). The 300-space car park is the base case; S12 and s12b.py repeat the comparison in a larger car park of 1008 spaces (12 areas of 84) to check whether the results hold as the lot grows.

## Requirements
Python 3.9 or later. Pure Python; no simulation library is needed.

    pip install -r requirements.txt

## Quick start

    python test_scoring.py            # checks the scoring formula against the prototype's space table
    python experiments.py S1 --reps 30   # main comparison, holiday
    bash run_all.sh                   # everything (slow)

Each experiment writes `results/<name>_raw.csv` (one row per method and replication) and `results/<name>_summary.csv` (means and 95% confidence intervals).

## Files
| File | Content |
|---|---|
| config.py | All parameters, each tagged with its source ([CODE] prototype, [SURVEY], [TDX], [ASSUMED]) |
| model.py | Car park layout, demand generation (NHPP by thinning), and the five methods |
| experiments.py | Experiments listed below and their figures |
| s12b.py | Guidance variants in the 1008-space car park |
| s_v3.py | Driver non-compliance with and without barriers; queue-length fallback under overload |
| s_v9.py | ParkSense with prototype-level hardware values |
| paired_stats.py | Paired t-test and Wilcoxon test over replications |
| paper_figs.py | Data figures of the paper (Figs. 4–6) |
| test_scoring.py, ParkingAllList.csv | Scoring check with the prototype's space table |
| tdx/ | Occupancy data and calibration scripts |

## Experiments
| Name | Purpose |
|---|---|
| S0 | Area occupancy over one day |
| S1 / S1w | Main comparison, holiday / weekday |
| S2 | Demand level 0.6–1.15 |
| S3 | Share of reserved drivers |
| S4 | Penalty threshold and factor |
| S5 | Locking and database latency (0.05–10 s) |
| S6 | Booking lead time |
| S7 | Parking duration 60 / 90 / 120 min |
| S8 | 30-min entrance failure |
| S9 | Guidance refresh period and choice set |
| S10 | Gate service time 2–8 s |
| S11 | Dual-mode vs single-mode weights |
| S12 | Scale check: 1008-space car park, same relative demand |
| S13 | Four sites |
| V1 | Simulated vs observed occupancy |

## Data
`tdx/availability_*.csv` are free-space counts of four mall car parks (identifiers KHB00434, KHB00245, KHB00247, TPE1813) downloaded from the Transport Data eXchange (TDX) platform of the Ministry of Transportation and Communications, Taiwan, between 25 October and 15 November 2026. They are open government data; please cite TDX when reusing them. `tdx/tdx_profile*.json` are the hourly arrival profiles and demand levels derived from them by `analyze_tdx.py` and `analyze_multi.py` (run them inside `tdx/`). The scripts do not download data and need no API key.

## Reproducibility note
All runs use fixed seeds, so results are identical on the same platform. Across platforms, the guidance-light method (GUIDE) can differ slightly: many spaces have the same walking distance, and the order of tied spaces returned by `numpy.argsort` may differ between NumPy builds. In a check on Windows (Python 3.13), GUIDE's mean time to park differed by about 0.1 s at the observed demand and by about 3 s under overload (demand 1.15), in both cases inside its 95% confidence interval; all other methods matched exactly, and no conclusion changed.

## License
MIT (see LICENSE).
