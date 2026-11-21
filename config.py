"""
ParkSense simulation — all parameters in one place.

Source tags:
  [CODE]    taken from parking.py (the prototype software)
  [SURVEY]  from the online survey (214 responses, October 2026)
  [PROTO]   order of magnitude from the prototype tests (see the paper, Sect. 4)
  [TDX]     from TDX parking availability data (see tdx/)
  [LIT]     from the literature cited in the paper
  [ASSUMED] assumption; the important ones are covered by sensitivity analyses
"""

BASE = {
    # ---------------- Layout (representative mall lot) ----------------
    "areas_x": 3,              # 3 x 2 = 6 areas (A-F)                      [ASSUMED]
    "areas_y": 2,
    "rows_per_area": 5,        # 5 rows x 10 spaces = 50 spaces per area -> 300 spaces
    "spaces_per_row": 10,
    "broken_ratio": 0.02,      # spaces out of service                      [ASSUMED]

    # ---------------- Scoring (identical to parking.py) ----------------
    "w_reservation": (0.2, 0.3, 0.4, 0.1),   # free capacity, area-entrance, space-entrance, position  [CODE]
    "w_walkin": (0.5, 0.2, 0.2, 0.1),        # [CODE]
    "penalty_threshold": 0.7,                # area occupancy > 70%  [CODE]
    "penalty_factor": 0.5,                   # whole space score x 0.5  [CODE]

    # ---------------- Demand ----------------
    "open_h": 8, "close_h": 22,              # arrivals 08:00-22:00 (Costco lot fills from 08:00, store closes 21:30)  [TDX]
    # relative hourly arrival rate 08:00 ... 21:00, derived from TDX occupancy of Costco Dashun (614 spaces),
    # 25 Oct - 15 Nov 2026 (cleaned), by deconvolution with a 90-min lognormal dwell (tdx/analyze_multi.py)  [TDX]
    "profile_weekend": [0.73, 1.0, 0.967, 0.717, 0.822, 0.876, 0.899, 0.892, 0.822, 0.764, 0.75, 0.77, 0.506, 0.143],   # Costco Dashun holidays/weekends, 7 days (26 Oct-15 Nov 2026)
    "profile_weekday": [0.327, 0.845, 0.904, 0.745, 0.76, 0.836, 1.0, 0.802, 0.586, 0.434, 0.609, 0.836, 0.473, 0.007],   # Costco Dashun weekdays, 8 days
    "profile": "weekend",
    "demand_level": 1.00,      # Costco holiday: peak arrivals x dwell / capacity (lower bound: censored when full)  [TDX]
    "demand_weekday": 0.93,    # [TDX]
    "dwell_mean_min": 90.0,    # parking duration, lognormal; not identifiable from occupancy -> S7 tests 60/90/120  [ASSUMED]
    "dwell_sd_min": 45.0,

    # ---------------- Reservation ----------------
    "res_share": 0.30,         # share of drivers who reserve (survey: 90.5% willing; adoption assumed lower)  [SURVEY]+[ASSUMED]
    "lead_min": (10, 60),      # booking made 10-60 min before arrival       [ASSUMED]
    "no_show": 0.10,           # reserved drivers who do not come            [ASSUMED]
    "grace_min": 15,           # hold the space this long after expected arrival  [ASSUMED]
    "arrive_sd_min": 5.0,      # actual arrival vs booked time               [ASSUMED]
    "split_zone_share": 0.30,  # SPLIT: spaces nearest the mall kept for reservations [ASSUMED] (= res_share)

    # ---------------- Patience (max acceptable wait) ----------------
    # survey max-wait answers: <30 min 176, 30-60 24, 60-90 5, 90-120 2, >120 5   [SURVEY]
    "patience_bins_min": [(5, 30), (30, 60), (60, 90), (90, 120), (120, 150)],
    "patience_probs": [176, 24, 5, 2, 5],   # survey max-wait question, 212 of 214 answered  [SURVEY]

    # ---------------- Entrance ----------------
    "gate_service_s": (4.0, 0.6),   # stop + ANPR gate, lognormal mean, sd; observed 3-5 s at mall gates; barrier opening 1-4 s (datasheet); 2-8 s tested  [LIT]/[ASSUMED]
    "gatefree_headway_s": 2.5,      # min headway at the speed bump (no stop)    [ASSUMED]
    "anpr_fail": 0.05,              # entrance plate read fails -> stop          [PROTO] (commercial camera assumed; 27% tested in s_v9.py)
    "anpr_fail_stop_s": 15.0,       # manual handling when the read fails        [ASSUMED]
    "lane_m": 20.0,                 # entrance lane length to the LED screen     [ASSUMED]
    "anpr_latency_s": 1.5,          # camera -> plate result                     [PROTO] (0.8-2.0 s measured)
    "assign_latency_s": 0.1,        # assignment read-score-write time (DB)      [ASSUMED] local database; prototype Google Sheets 6.5 s; 0.05-10 s tested
    "assign_lock": True,            # serialise assignments (False = prototype behaviour)  [CODE]
    "bay_size": 5,                  # ParkSense waiting bay for walk-ins when full  [ASSUMED]

    # ---------------- In-lot driving ----------------
    "speed_kmh": 10.0,              # in-lot speed                               [ASSUMED]
    "space_pitch_m": 4.5,           # distance passed per space while scanning   [ASSUMED]
    "maneuver_s": 20.0,             # parking manoeuvre                          [ASSUMED]
    "verify_s": 3.0,                # space plate check + barrier down           [PROTO] (2.1 s measured)
    "verify_fail": 0.03,            # space plate check fails -> retry           [PROTO] (10% tested in s_v9.py)
    "verify_retry_s": 8.0,          #                                            [ASSUMED]
    "info_period_s": 30.0,          # GUIDE: sensor/light update period          [ASSUMED]
    "guide_topk": 3,                # GUIDE: driver picks among k nearest shown-free spaces [ASSUMED]
    "conflict_extra_s": 60.0,       # ParkSense duplicate assignment -> manual reassign  [ASSUMED]
    "cruise_retry_s": 60.0,         # full lot: loop again after this time       [ASSUMED]

    # ---------------- Emissions ----------------
    "ef_kg_per_km": 0.249,          # CO2, EPA typical passenger vehicle ~400 g/mile (average driving; low-speed in-lot is higher -> conservative)  [LIT]
    "idle_kg_per_s": 0.00066,       # CO2 idling: Argonne (Gaines et al. 2013) measured 0.279 cc/s (2.5 L car; range 0.2-0.5 gal/h) x EPA 8,887 g CO2/gal  [LIT]

    # ---------------- Run ----------------
    "sample_s": 60.0,               # area occupancy sampling
    "kpi_window_h": (12, 20),       # window for occupancy-balance KPI
}

METHODS = ["CONV", "GUIDE", "SPLIT", "NEAREST", "PARKSENSE"]
METHOD_LABELS = {
    "CONV": "Conventional (gate + search)",
    "GUIDE": "Guidance lights (gate)",
    "SPLIT": "Separate reserved zone (gate)",
    "NEAREST": "Gate-free + nearest space",
    "PARKSENSE": "ParkSense (proposed)",
}

# Per-mall calibration (tdx/analyze_multi.py, 90-min dwell; demand relative to capacity net of overnight vehicles)
MALLS = {
 "KHB00434_weekday": {
  "name": "Costco Dashun",
  "kind": "warehouse store",
  "cap": 614,
  "type": "weekday",
  "days": 8,
  "profile": [
   0.327,
   0.845,
   0.904,
   0.745,
   0.76,
   0.836,
   1.0,
   0.802,
   0.586,
   0.434,
   0.609,
   0.836,
   0.473,
   0.007
  ],
  "demand": 0.928,
  "peak_occ": 0.946,
  "base": 0.049
 },
 "KHB00434_holiday": {
  "name": "Costco Dashun",
  "kind": "warehouse store",
  "cap": 614,
  "type": "holiday",
  "days": 7,
  "profile": [
   0.73,
   1.0,
   0.967,
   0.717,
   0.822,
   0.876,
   0.899,
   0.892,
   0.822,
   0.764,
   0.75,
   0.77,
   0.506,
   0.143
  ],
  "demand": 1.001,
  "peak_occ": 0.976,
  "base": 0.04
 },
 "KHB00245_weekday": {
  "name": "Shin Kong Mitsukoshi Zuoying",
  "kind": "department store",
  "cap": 1055,
  "type": "weekday",
  "days": 8,
  "profile": [
   0.004,
   0.011,
   0.079,
   0.54,
   0.688,
   0.7,
   0.654,
   0.648,
   0.644,
   0.86,
   1.0,
   0.915,
   0.588,
   0.281
  ],
  "demand": 0.563,
  "peak_occ": 0.631,
  "base": 0.163
 },
 "KHB00245_holiday": {
  "name": "Shin Kong Mitsukoshi Zuoying",
  "kind": "department store",
  "cap": 1055,
  "type": "holiday",
  "days": 4,
  "profile": [
   0.049,
   0.146,
   0.198,
   1.0,
   0.877,
   0.837,
   0.839,
   0.825,
   0.825,
   0.814,
   0.833,
   0.71,
   0.392,
   0.202
  ],
  "demand": 1.0,
  "peak_occ": 0.897,
  "base": 0.163
 },
 "KHB00247_weekday": {
  "name": "Carrefour Xinnan",
  "kind": "hypermarket",
  "cap": 460,
  "type": "weekday",
  "days": 8,
  "profile": [
   0.051,
   0.156,
   0.255,
   0.499,
   0.645,
   0.288,
   0.335,
   0.3,
   0.37,
   0.704,
   1.0,
   0.712,
   0.319,
   0.086
  ],
  "demand": 0.376,
  "peak_occ": 0.458,
  "base": 0.165
 },
 "KHB00247_holiday": {
  "name": "Carrefour Xinnan",
  "kind": "hypermarket",
  "cap": 460,
  "type": "holiday",
  "days": 4,
  "profile": [
   0.081,
   0.21,
   0.435,
   0.736,
   0.879,
   0.741,
   0.612,
   0.433,
   0.694,
   0.974,
   1.0,
   0.736,
   0.337,
   0.138
  ],
  "demand": 0.761,
  "peak_occ": 0.806,
  "base": 0.171
 },
 "TPE1813_weekday": {
  "name": "Mitsui LaLaport Nangang",
  "kind": "shopping mall",
  "cap": 2378,
  "type": "weekday",
  "days": 9,
  "profile": [
   0.42,
   0.516,
   0.642,
   0.866,
   1.0,
   0.97,
   0.92,
   0.823,
   0.719,
   0.662,
   0.684,
   0.516,
   0.211,
   0.038
  ],
  "demand": 0.522,
  "peak_occ": 0.614,
  "base": 0.167
 },
 "TPE1813_holiday": {
  "name": "Mitsui LaLaport Nangang",
  "kind": "shopping mall",
  "cap": 2378,
  "type": "holiday",
  "days": 4,
  "profile": [
   0.014,
   0.047,
   0.259,
   0.645,
   0.839,
   0.932,
   1.0,
   0.972,
   0.946,
   0.938,
   0.869,
   0.617,
   0.307,
   0.072
  ],
  "demand": 0.778,
  "peak_occ": 0.862,
  "base": 0.162
 }
}
